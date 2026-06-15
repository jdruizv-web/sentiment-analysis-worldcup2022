"""
Lee configuración → usa label_strategy, label_output, label_validation_sample, seed desde preprocess_config.yml.

Estrategias de etiquetado →

pysentimiento (RoBERTa entrenado en español).

cardiffnlp (XLM‑RoBERTa multilingüe).

both (ejecuta ambos y calcula Cohen’s Kappa).

Añade columnas al corpus → label_sentiment, label_confidence, y opcionalmente label_cardiffnlp, label_conf_cardiffnlp.

Distribución de etiquetas → imprime conteo y porcentajes por clase.

Muestra de validación manual → genera data/label_validation_sample.csv con columna vacía label_manual.

Guarda corpus etiquetado → en data/corpus_labeled_v1.jsonl.

Log de Kappa → añade interpretación del acuerdo al preprocess_log.txt.
"""

import yaml                          # Para leer la configuración YAML
import pandas as pd                  # Para manejar el DataFrame
from pathlib import Path             # Para crear directorios de salida de forma segura

# ============================================================
# Cargar configuración desde preprocess_config.yml
# ============================================================
with open("src/preprocess_config.yml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)       # Devuelve un dict con todos los parámetros del YAML

# Leer parámetros de etiquetado; el segundo argumento es el valor por defecto
LABEL_STRATEGY  = config.get("label_strategy", "pysentimiento")   # Estrategia a usar
LABEL_OUTPUT    = config.get("label_output", "data/corpus_labeled_v1.jsonl")  # Archivo de salida
LABEL_SAMPLE    = config.get("label_validation_sample", 150)       # Tamaño de muestra de validación
SEED            = config.get("seed", 42)                           # Semilla para reproducibilidad


# ============================================================
# ESTRATEGIA A — pysentimiento
# Modelo: RoBERTa preentrenado en >40M tweets en español
# Paper: Pérez et al. (2021) arXiv:2106.09462
# Instalación: pip install pysentimiento
# ============================================================
def label_with_pysentimiento(texts: list[str]) -> tuple[list[str], list[float]]:
    """
    Clasifica una lista de textos usando pysentimiento.
    Devuelve dos listas: etiquetas y confianzas.

    pysentimiento devuelve objetos con:
      .output   → "POS" | "NEG" | "NEU"
      .probas   → dict {"POS": 0.8, "NEG": 0.1, "NEU": 0.1}
    """
    try:
        from pysentimiento import create_analyzer   # Importación tardía: solo si se usa esta estrategia
    except ImportError:
        raise ImportError(
            "pysentimiento no está instalado.\n"
            "Instálalo con: pip install pysentimiento"
        )

    # Mapa de etiquetas del modelo (inglés abreviado) → español para el corpus
    label_map = {"POS": "positivo", "NEG": "negativo", "NEU": "neutro"}

    print("🔄 Cargando pysentimiento (modelo RoBERTa-es)...")
    analyzer = create_analyzer(task="sentiment", lang="es")   # Carga el modelo en español
    print("✅ Modelo pysentimiento listo.")

    sentiments  = []   # Acumula la etiqueta de cada tweet
    confidences = []   # Acumula la probabilidad de la clase ganadora

    for i, text in enumerate(texts):
        result = analyzer.predict(str(text)[:512])              # Límite de 512 caracteres del modelo
        sentiments.append(label_map.get(result.output, "neutro"))  # Fallback a "neutro" si desconocido
        confidences.append(round(max(result.probas.values()), 3))  # Máx. probabilidad = confianza

        if (i + 1) % 500 == 0:                                  # Progreso cada 500 tweets
            print(f"   ➡️ pysentimiento: {i+1}/{len(texts)} tweets procesados...")

    return sentiments, confidences


# ============================================================
# ESTRATEGIA B — cardiffnlp/twitter-xlm-roberta-base-sentiment
# Modelo: XLM-RoBERTa fine-tuned en tweets multilingüe (incluye español)
# Paper: Barbieri et al. (2020) — TweetEval — arXiv:2010.12421
# Instalación: pip install transformers torch
# Se descarga automáticamente la primera vez (~500 MB)
# ============================================================
def label_with_cardiffnlp(texts: list[str]) -> tuple[list[str], list[float]]:
    """
    Clasifica una lista de textos con el modelo XLM-RoBERTa de Cardiff NLP.
    Devuelve dos listas: etiquetas y confianzas.

    El modelo devuelve objetos con:
      ["label"]  → "positive" | "negative" | "neutral"
      ["score"]  → probabilidad de la clase ganadora
    """
    try:
        from transformers import pipeline               # Importación tardía
    except ImportError:
        raise ImportError(
            "transformers no está instalado.\n"
            "Instálalo con: pip install transformers torch"
        )

    # Mapa de etiquetas del modelo (inglés) → español para el corpus
    label_map = {"positive": "positivo", "negative": "negativo", "neutral": "neutro"}

    print("🔄 Cargando cardiffnlp/twitter-xlm-roberta-base-sentiment...")
    print("   (Primera descarga puede tardar ~2 minutos, ~500 MB)")
    classifier = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",       # Modelo en Hugging Face Hub
        tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment",   # Tokenizador del mismo modelo
        truncation=True,    # Recortar textos que superen el límite de tokens
        max_length=128      # Máximo de tokens (suficiente para tweets)
    )
    print("✅ Modelo cardiffnlp listo.")

    sentiments  = []
    confidences = []

    for i, text in enumerate(texts):
        result = classifier(str(text)[:512])[0]                        # [0] porque pipeline devuelve lista
        sentiments.append(label_map.get(result["label"].lower(), "neutro"))  # Normalizar a minúsculas
        confidences.append(round(result["score"], 3))                  # "score" = confianza de la predicción

        if (i + 1) % 500 == 0:
            print(f"   ➡️ cardiffnlp: {i+1}/{len(texts)} tweets procesados...")

    return sentiments, confidences


# ============================================================
# Cálculo de Cohen's Kappa entre dos anotadores
# ============================================================
def compute_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """
    Calcula el coeficiente Kappa de Cohen entre dos conjuntos de etiquetas.
    Mide el acuerdo inter-anotador corrigiendo el acuerdo por azar.

    Interpretación estándar:
      κ < 0.40  → acuerdo pobre
      κ 0.40–0.60 → acuerdo moderado
      κ 0.60–0.80 → acuerdo sustancial  ← umbral mínimo recomendado para TFM
      κ > 0.80  → acuerdo casi perfecto

    Requiere scikit-learn (pip install scikit-learn).
    """
    try:
        from sklearn.metrics import cohen_kappa_score   # Importación tardía
    except ImportError:
        raise ImportError(
            "scikit-learn no está instalado.\n"
            "Instálalo con: pip install scikit-learn"
        )
    return round(cohen_kappa_score(labels_a, labels_b), 4)   # Redondear a 4 decimales


# ============================================================
# Función de validación con muestra manual
# ============================================================
def save_validation_sample(df: pd.DataFrame, n: int, seed: int) -> None:
    """
    Guarda una muestra aleatoria estratificada en CSV para anotación manual.
    La muestra contiene el texto limpio y la etiqueta automática,
    más una columna vacía 'label_manual' para que el investigador la rellene.

    El investigador anota manualmente esta muestra y luego puede calcular
    el acuerdo con las etiquetas automáticas usando compute_kappa().
    """
    # Seleccionar columnas disponibles para la muestra
    cols = ["clean_text", "label_sentiment", "label_confidence"]
    if "full_text" in df.columns:
        cols.insert(0, "full_text")               # Añadir texto original si existe
    if "label_cardiffnlp" in df.columns:
        cols.append("label_cardiffnlp")           # Añadir segundo anotador si existe

    # Trabajar solo con las columnas que necesitamos
    df_cols = df[cols].copy()

    # Muestra estratificada: n//3 tweets por cada clase (positivo/negativo/neutro)
    per_class = max(1, n // 3)
    partes = []
    for clase in ["positivo", "negativo", "neutro"]:
        subset = df_cols[df_cols["label_sentiment"] == clase]   # Filtrar por clase
        k = min(len(subset), per_class)                         # No pedir más de los que hay
        if k > 0:
            partes.append(subset.sample(k, random_state=seed))  # Muestra reproducible

    sample = pd.concat(partes, ignore_index=True)   # Unir las tres partes
    sample["label_manual"] = ""                     # Columna vacía para anotación humana

    output_path = "data/label_validation_sample.csv"
    # Si el archivo esta abierto (ej. en Excel), intentar cerrar antes de escribir
    try:
        sample.to_csv(output_path, index=False, encoding="utf-8")
    except PermissionError:
        print(f"\n  AVISO: No se pudo guardar {output_path}.")
        print("  Cierra el archivo si esta abierto en Excel u otro programa y vuelve a correr.\n")
        return
    print(f"✅ Muestra de validación guardada: {output_path} ({len(sample)} tweets)")
    print("   → Rellena la columna 'label_manual' y calcula Cohen's Kappa con compute_kappa()")


# ============================================================
# Función principal — llamada desde preprocess.py
# ============================================================
def label_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Punto de entrada principal.
    Lee la estrategia desde preprocess_config.yml y añade las columnas de sentimiento.

    Estrategias:
      "pysentimiento"  → solo pysentimiento (baseline principal)
      "cardiffnlp"     → solo cardiffnlp (segundo anotador)
      "both"           → ambos modelos + Cohen's Kappa entre ellos (recomendado para TFM)
    """
    strategy = LABEL_STRATEGY.lower()   # Normalizar a minúsculas por si hay mayúsculas en el YAML
    texts    = df["clean_text"].tolist() # Extraer la columna de texto como lista Python

    print(f"\n{'='*60}")
    print(f"📊 ETIQUETADO DE SENTIMIENTO — estrategia: {strategy.upper()}")
    print(f"   Corpus: {len(texts)} tweets")
    print(f"{'='*60}")

    # ── Estrategia A: solo pysentimiento ───────────────────
    if strategy == "pysentimiento":
        sentiments, confidences = label_with_pysentimiento(texts)
        df["label_sentiment"]  = sentiments    # Etiqueta final
        df["label_confidence"] = confidences   # Confianza de la predicción

    # ── Estrategia B: solo cardiffnlp ──────────────────────
    elif strategy == "cardiffnlp":
        sentiments, confidences = label_with_cardiffnlp(texts)
        df["label_sentiment"]  = sentiments
        df["label_confidence"] = confidences

    # ── Estrategia C: ambos modelos + Kappa ────────────────
    elif strategy == "both":
        # Correr ambos modelos sobre todos los tweets
        sent_pys, conf_pys = label_with_pysentimiento(texts)
        sent_car, conf_car = label_with_cardiffnlp(texts)

        # pysentimiento es el anotador principal (etiqueta oficial del corpus)
        df["label_sentiment"]       = sent_pys
        df["label_confidence"]      = conf_pys

        # cardiffnlp es el segundo anotador (para validación)
        df["label_cardiffnlp"]      = sent_car
        df["label_conf_cardiffnlp"] = conf_car

        # Calcular Cohen's Kappa entre los dos sistemas
        kappa = compute_kappa(sent_pys, sent_car)
        print(f"\nCohen's Kappa (pysentimiento vs cardiffnlp): k = {kappa}")

        # Interpretar el valor de Kappa
        if kappa >= 0.80:
            nivel = "casi perfecto"
        elif kappa >= 0.60:
            nivel = "sustancial (aceptable para TFM)"
        elif kappa >= 0.40:
            nivel = "moderado (reportar y justificar)"
        else:
            nivel = "pobre (revisar criterios o muestra)"
        print(f"   Interpretacion: {nivel}")

        # Guardar Kappa en el log de preprocesamiento (append, no sobreescribe)
        with open("data/preprocess_log.txt", "a", encoding="utf-8") as log:
            log.write(f"\nLabel strategy: {strategy}\n")
            log.write(f"Cohen's Kappa (pysentimiento vs cardiffnlp): {kappa}\n")
            log.write(f"Kappa interpretation: {nivel}\n")

        # Guardar tambien en archivo dedicado para referencia en el documento
        with open("data/kappa_result.txt", "w", encoding="utf-8") as kf:
            kf.write(f"Cohen's Kappa (pysentimiento vs cardiffnlp): {kappa}\n")
            kf.write(f"Kappa interpretation: {nivel}\n")
            kf.write(f"N tweets evaluated: {len(texts)}\n")
            kf.write(f"Seed: {SEED}\n")

    else:
        # Estrategia no reconocida: abortar con mensaje claro
        raise ValueError(
            f"label_strategy='{strategy}' no reconocida en preprocess_config.yml.\n"
            f"Valores válidos: 'pysentimiento' | 'cardiffnlp' | 'both'"
        )

    # ── Distribución de etiquetas ───────────────────────────
    # Mostrar cuántos tweets tiene cada clase (útil para detectar desequilibrio)
    dist = df["label_sentiment"].value_counts()
    print("\n📈 Distribución de etiquetas (label_sentiment):")
    for label, count in dist.items():
        pct = 100 * count / len(df)
        bar = "█" * int(pct / 2)                          # Barra visual proporcional
        print(f"   {label:10s}: {count:5d} ({pct:5.1f}%)  {bar}")

    # ── Muestra de validación manual ────────────────────────
    # Genera un CSV con LABEL_SAMPLE tweets para que el investigador anote manualmente
    # y pueda calcular el acuerdo con las etiquetas automáticas
    save_validation_sample(df, n=LABEL_SAMPLE, seed=SEED)

    # ── Guardar corpus etiquetado ────────────────────────────
    print(f"\n🔄 Guardando corpus etiquetado en {LABEL_OUTPUT}...")
    Path(LABEL_OUTPUT).parent.mkdir(parents=True, exist_ok=True)  # Crear carpeta si no existe
    df.to_json(LABEL_OUTPUT, orient="records", lines=True, force_ascii=False)  # JSONL, una línea por tweet
    print(f"✅ Corpus etiquetado guardado: {LABEL_OUTPUT}")

    return df   # Devolver DataFrame con las columnas nuevas para continuar el pipeline


# ============================================================
# Ejecución directa como script independiente (sin preprocess.py)
# python label_sentiment.py
# ============================================================
if __name__ == "__main__":
    # Este bloque solo corre si ejecutas: python label_sentiment.py
    # Si importas el módulo desde preprocess.py, este bloque se ignora
    INPUT_PATH = "data/corpus_processed_v1.jsonl"
    print(f"🔄 Cargando corpus desde {INPUT_PATH}...")
    df = pd.read_json(INPUT_PATH, lines=True)   # Leer JSONL como DataFrame de pandas
    print(f"✅ Corpus cargado: {len(df)} registros.")

    df = label_dataframe(df)                    # Ejecutar etiquetado completo
    print("\n🎉 Etiquetado completado con éxito.")

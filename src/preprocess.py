"""
Resumen:

Cargamos archivo corpus bruto

Aplicamos limpieza y normalización → eliminamos ruido (URLs, menciones), pasamos a minúsculas, normalizamos espacios y caracteres.

Tokenizamos.

creamos diccionario de emojis y hashtags para normalizar y remover data innecesaria

Generamos corpus procesado y guardamos corpus_processed_v1.jsonl.

Sacamos una muestra aleatoria y creamos sample_processed.csv con el tamaño definido en el parametro sample_size.

Registramos los parámetros en el archivo preprocess_log.txt  y creamos el log del preprocesamiento.

Llamamos al script label_sentiment.py, si label_strategy ≠ "none" entonces añadimos etiquetas de sentimiento automáticamente.

"""

# preprocess.py (Preprocesado avanzado con multiproceso y progreso en tokenización)

import re                  # Para expresiones regulares en limpieza de texto
import unicodedata         # Para normalizar caracteres Unicode
import pandas as pd        # Para manejar DataFrames y archivos CSV/JSONL
import spacy               # Para tokenización y lematización
import emoji               # Para tratamiento de emojis
import yaml                # Para leer archivo de configuración YAML

# ============================================================
# 1. Cargar configuración desde preprocess_config.yml
# ============================================================
with open("src/preprocess_config.yml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)   # Lee todos los parámetros definidos en el YAML

# Parámetros generales
SAMPLE_SIZE      = config.get("sample_size", 500)               # Tamaño de muestra aleatoria para CSV
SEED             = config.get("seed", 42)                        # Semilla para reproducibilidad
LEMMATIZE        = config.get("lemmatize", False)                # True = usar lemas; False = tokens originales
EMOJI_STRATEGY   = config.get("emoji_strategy", "remove")        # "remove" | "map"
HASHTAG_STRATEGY = config.get("hashtag_strategy", "normalize")   # "normalize" | "remove"

# Parámetro de etiquetado — controla si se ejecuta label_sentiment al final
LABEL_STRATEGY   = config.get("label_strategy", "none")          # "pysentimiento" | "cardiffnlp" | "both" | "none"

print("✅ Configuración cargada desde preprocess_config.yml")
print(f"   label_strategy = {LABEL_STRATEGY}")

# ============================================================
# 2. Cargarmos modelo de spaCy
# ============================================================
print("🔄 Cargando modelo de spaCy para español...")
nlp = spacy.load("es_core_news_sm")   # Modelo de lenguaje en español (tokenización y lematización)
print("✅ Modelo cargado correctamente.")

# ============================================================
# 3. Diccionario de emojis 
# ============================================================
emoji_map = {
    "🙂": "feliz",
    "😊": "feliz",
    "🤣": "feliz",
    "😂": "feliz",
    "😎": "feliz",
    "💀": "feliz",
    "😢": "triste",
    "😭": "triste",
    "😔": "triste",
    "🤬": "enojado",
    "😡": "enojado",
    "😠": "enojado",
    "😤": "enojado",
    "❤️": "amor",
    "🥰": "amor",
    "😍": "amor",
    "💕": "amor",
    "🥳": "celebracion",
    "🫠": "incomodo",
    "😅": "nervioso",
    "🙈": "verguenza",
    "😳": "verguenza",
    "🐸": "ironia",
    "💩": "apesta",
    "💅": "desinteresado",
    "🙃": "sarcasmo",
    "💪": "motivado",
    "👏": "reconocimiento",
    "🤔": "duda",
    "😜": "coqueto",
    "😏": "coqueto",
    "🤯": "sorprendido",
    "🥺": "suplica",
    "🤡": "ridiculo",
    "🔥": "popular",
    "⭐": "sobresaliente",
    "😐": "indiferencia",
    "😇": "inocente",
    "🤮": "apesta",
    "🥶": "impactado"
}

# ============================================================
# 4. Limpieza de texto
# ============================================================

# Para paises con nombres compuestos los tratamos como un solo token
# Reemplazamos espacio por guion bajo antes de tokenizar
compound_locations = [
    "Países Bajos", "Costa Rica", "Estados Unidos", "República Dominicana",
    "Nueva Zelanda", "San José", "El Salvador", "Puerto Rico",
    "Bosnia y Herzegovina", "Trinidad y Tobago", "Guinea Ecuatorial"
]

def normalize_compound_locations(text: str) -> str:
    """
    Reemplaza nombres compuestos por versiones con guion bajo
    para que spaCy los trate como un solo token.
    Ejemplo: "costa rica" → "costa_rica"
    """
    for loc in compound_locations:
        text = text.replace(loc.lower(), loc.lower().replace(" ", "_"))
    return text


def clean_text(text: str) -> str:
    """
    Pipeline de limpieza avanzada de texto de tweets:
      1. Normalizamos Unicode (NFKC: unifica variantes de caracteres)
      2. Converimos a minúsculas
      3. Eliminamos URLs y menciones (@usuario)
      4. Normalizamos o eliminamos hashtags según HASHTAG_STRATEGY
      5. Aplicamos estrategia de emojis (remove/map) según EMOJI_STRATEGY
      6. Normalizamos nombres de paises compuestos
      7. Separamos palabras unidas por guion (ej: "bien-dicho" → "bien dicho")
      8. Eliminamos espacios extra al inicio, final y entre palabras
    """
    text = unicodedata.normalize("NFKC", str(text))    # Paso 1: normalizar unicode
    text = text.lower()                                 # Paso 2: minúsculas
    text = re.sub(r"http\S+|www\S+", "", text)          # Paso 3: eliminar URLs
    text = re.sub(r"@\w+", "", text)                    # Paso 3: eliminar menciones

    # Paso 4: hashtags
    if HASHTAG_STRATEGY == "normalize":
        text = re.sub(r"#(\w+)", lambda m: m.group(1).lower(), text)   # #Mundial → mundial
    elif HASHTAG_STRATEGY == "remove":
        text = re.sub(r"#\w+", "", text)                               # Eliminar completamente

    # Paso 5: emojis
    if EMOJI_STRATEGY == "remove":
        text = emoji.replace_emoji(text, replace="")                   # Eliminar emojis
    elif EMOJI_STRATEGY == "map":
        text = emoji.replace_emoji(
            text,
            replace=lambda e, data: " " + emoji_map.get(e, "")        # Sustituir por palabra
        )

    text = normalize_compound_locations(text)           # Paso 6: nombres geográficos
    text = re.sub(r"(\w)-(\w)", r"\1 \2", text)         # Paso 7: separar guiones entre palabras
    text = re.sub(r"\s+", " ", text).strip()            # Paso 8: normalizar espacios
    return text


# ============================================================
# 5. Pipeline de preprocesamiento principal
# ============================================================
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ejecutamos el pipeline sobre el DataFrame:
      1. Limpiamos texto
      2. Tokenizamos y lematizamoscuando sea necesario
      3. Etiquetamos sentimiento, columnas label_sentiment (si label_strategy != "none")
      4. Guardamos el corpus procesado completo (JSONL)
      5. Guardadomos la muestra aleatoria (CSV)
      6. Generamos el log de preprocesamiento
    """

    # ── Paso 1: Limpieza ──────────────────────────────────────
    print("🔄 Aplicando limpieza de texto...")
    df["clean_text"] = df["full_text"].apply(clean_text)   # Aplica clean_text() a cada tweet
    print("✅ Limpieza completada.")

    # ── Paso 2: Tokenización ─────────────────────────────────
    print("🔄 Aplicando tokenización con multiproceso y progreso...")
    tokens_list = []   # Acumula la lista de tokens de cada tweet

    # nlp.pipe procesa en lotes (batch_size=100) y en paralelo (n_process=4)
    # Es mucho más eficiente que llamar nlp() tweet por tweet
    for i, doc in enumerate(nlp.pipe(df["clean_text"], batch_size=100, n_process=4)):
        if LEMMATIZE:
            # LEMMATIZE=True: usa lema de cada token (forma base)
            # Útil para modelos clásicos (TF-IDF + SVM/NB); no recomendado con BETO
            tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_space]
        else:
            # LEMMATIZE=False: usar el token tal como aparece en el texto limpio
            # Recomendado para pysentimiento y BETO (esperan texto más natural)
            tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
        tokens_list.append(tokens)

        if (i + 1) % 1000 == 0:   # Progreso cada 1000 registros
            print(f"   ➡️ Procesados {i+1} registros...")

    df["tokens"] = tokens_list   # Añadir columna de tokens al DataFrame
    print("✅ Tokenización completada.")

    # ── Paso 3: Generar log base de preprocesamiento ─────────
    # Se escribe ANTES del etiquetado para que label_sentiment.py
    # pueda hacer append del valor de kappa sin ser sobreescrito.
    print("🔄 Generando log de preprocesamiento...")
    with open("data/preprocess_log.txt", "w", encoding="utf-8") as log:
        log.write("Preprocessing completed\n")
        log.write(f"Total records: {len(df)}\n")
        log.write(f"Sample size: {SAMPLE_SIZE}\n")
        log.write(f"Random seed: {SEED}\n")
        log.write(f"Lemmatization: {LEMMATIZE}\n")
        log.write(f"Emoji strategy: {EMOJI_STRATEGY}\n")
        log.write(f"Hashtag strategy: {HASHTAG_STRATEGY}\n")
        log.write(f"Label strategy: {LABEL_STRATEGY}\n")
    print("✅ Log base guardado: data/preprocess_log.txt")

    # ── Paso 4: Etiquetado de sentimiento (opcional) ─────────
    # Solo se ejecuta si label_strategy != "none" en el YAML.
    # label_dataframe() hace append de Cohen's Kappa al log cuando strategy="both".
    if LABEL_STRATEGY.lower() != "none":
        print(f"\n🔄 Iniciando etiquetado de sentimiento (strategy={LABEL_STRATEGY})...")
        from label_sentiment import label_dataframe   # Importación tardía: evita cargar los modelos
                                                      # si label_strategy="none" y no son necesarios
        df = label_dataframe(df)                      # Añade label_sentiment, label_confidence, etc.
        print("✅ Etiquetado completado.")
    else:
        print("ℹ️  label_strategy='none' — se omite el etiquetado de sentimiento.")

    # ── Paso 5: Guardar corpus procesado completo ────────────
    output_path = "data/corpus_processed_v1.jsonl"
    print(f"🔄 Guardando corpus procesado completo en {output_path}...")
    df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    # orient="records" → un dict por fila; lines=True → una línea por registro (formato JSONL)
    # force_ascii=False → preservar tildes y ñ en el archivo
    print(f"✅ Corpus procesado guardado: {output_path}")

    # ── Paso 6: Guardar muestra aleatoria ────────────────────
    print(f"🔄 Creando muestra aleatoria de {SAMPLE_SIZE} registros...")
    sample = df.sample(n=SAMPLE_SIZE, random_state=SEED)   # Muestra reproducible gracias a SEED
    sample.to_csv("data/sample_processed.csv", index=False, encoding="utf-8")
    print("✅ Muestra guardada: data/sample_processed.csv")

    return df   # Devolver DataFrame completo con todas las columnas nuevas


# ============================================================
# 6. Ejecución directa del script
# python preprocess.py
# ============================================================
if __name__ == "__main__":
    print("🔄 Cargando corpus canónico...")
    df = pd.read_json("data/corpus_canonical_v1.jsonl", lines=True)   # Lee corpus canónico en JSONL
    print(f"✅ Corpus cargado: {len(df)} registros.")

    print("🚀 Iniciando preprocesamiento...")
    df_processed = preprocess_dataframe(df)   # Ejecuta el pipeline completo
    print("🎉 Preprocesamiento completado con éxito.")

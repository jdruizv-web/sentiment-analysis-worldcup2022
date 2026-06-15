"""
Resumen:

Cargamos corpus etiquetado, leemos corpus_labeled_v1.jsonl.

Definimos entrada y salida, limpiamos texto con clean_text como X y label_sentiment como y.

Vectorizamos texto, usamos TF‑IDF para representar los tweets.

Entrenamos modelos clásicos Naïve Bayes, Logistic Regression y SVM.

Evaluamos rendimiento y calculamos métricas (precision, recall, F1, matriz de confusión).

Guardamos resultados en results_baselines.csv.

Guardamos modelos entrenados.

Documentamos parámetros y resultados
"""

# baselines.py (Modelos clásicos con TF-IDF)

import pandas as pd                          # Para manejar DataFrames y leer el corpus procesado
from sklearn.feature_extraction.text import TfidfVectorizer  # Convierte texto en vectores TF-IDF
from sklearn.model_selection import train_test_split         # Divide datos en train/test
from sklearn.naive_bayes import MultinomialNB                # Modelo Naïve Bayes
from sklearn.linear_model import LogisticRegression          # Modelo Regresión Logística
from sklearn.svm import LinearSVC                            # Modelo SVM lineal
from sklearn.metrics import classification_report            # Calcula métricas de evaluación
import joblib                                                # Guarda modelos entrenados en disco
import os                                                    # Maneja carpetas y archivos

# ============================================================
# 1. Cargarmos corpus procesado
print("🔄 Cargando corpus procesado...")
df = pd.read_json("data/corpus_processed_v1.jsonl", lines=True)  # Lee corpus del Día 4
print(f"✅ Corpus cargado con {len(df)} registros.")

# Utilizamos la columna 'label_sentiment' previamenete creada con el script label_sentiment.py
X_texts = df["clean_text"]   # Usamos el texto limpio como entrada
y = df["label_sentiment"]              # Usamos la etiqueta como salida

# ============================================================
# 2. Vectorización TF-IDF
print("🔄 Vectorizando textos con TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000)  # Limitamos a 5000 características
X = vectorizer.fit_transform(X_texts)            # Convertimos los textos en vectores numéricos
print("✅ Vectorización completada.")

# ============================================================
# 3. Definimos procentaje de entranamiento y prueva
print("🔄 Dividiendo en train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)        # 80% entrenamiento, 20% prueba
print("✅ División completada.")

# ============================================================
# 4. Definir modelos
models = {
    "NaiveBayes": MultinomialNB(),               # Modelo Naïve Bayes multinomial
    "LogisticRegression": LogisticRegression(max_iter=1000),  # Regresión Logística
    "SVM": LinearSVC()                           # SVM lineal
}

results = []                                     # Lista para guardar métricas

# Crear carpeta dentro de data para guardar modelos entrenados
os.makedirs("data/models_baselines", exist_ok=True)

# ============================================================
# 5. Entrenamos y evaluamos cada modelo
for name, model in models.items():
    print(f"🚀 Entrenando {name}...")
    model.fit(X_train, y_train)                  # Entrenamos el modelo
    y_pred = model.predict(X_test)               # Predecimos etiquetas en test
    report = classification_report(y_test, y_pred, output_dict=True)  # Calculamos métricas
    
    # Guardar métricas en la lista
    results.append({
        "model": name,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"]
    })
    
    # Guardar modelo entrenado en disco dentro de data/models_baselines
    joblib.dump(model, f"data/models_baselines/{name}.pkl")
    print(f"✅ {name} entrenado y guardado.")

# ============================================================
# 6. Guardamos resultados
print("🔄 Guardando métricas...")
pd.DataFrame(results).to_csv("data/results_baselines.csv", index=False)  # Guardamos métricas en CSV
print("✅ Métricas guardadas en data/results_baselines.csv")

# ============================================================
# 7. Guardamos el log de experimento
print("🔄 Guardando log de experimento...")
with open("data/experiments_log.txt", "w", encoding="utf-8") as log:
    log.write("Baselines training completed\n")
    log.write("Models: NaiveBayes, LogisticRegression, SVM\n")
    log.write("Vectorizer: TF-IDF (max_features=5000)\n")
    log.write("Seed: 42\n")
print("✅ Log guardado en data/experiments_log.txt")

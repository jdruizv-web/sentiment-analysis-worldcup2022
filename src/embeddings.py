"""
Resumen
Cargamos corpus.

Generamos embeddings con BETO y Sentence‑Transformers.

Entrenamos los modelos clásicos.

Evaluamos y guardamos métricas en results_embeddings.csv.

Guarda modelos entrenados → en data/models_embeddings/.

Log del experimento → en data/experiments_embeddings_log.txt.
"""

# embeddings.py (Modelos avanzados con BETO y Sentence-Transformers)

import pandas as pd                      # Para manejar el corpus en DataFrame
from sklearn.model_selection import train_test_split   # Para dividir train/test
from sklearn.linear_model import LogisticRegression    # Modelo clásico
from sklearn.svm import LinearSVC                      # Modelo clásico
from sklearn.metrics import classification_report      # Métricas de evaluación
import joblib                           # Para guardar modelos entrenados
import os                               # Para crear carpetas si no existen

# HuggingFace y Sentence-Transformers
from transformers import AutoTokenizer, AutoModel      # Para cargar BETO
from sentence_transformers import SentenceTransformer  # Para embeddings ST

import torch                            # Para manejar tensores de BETO
import numpy as np                       # Para arrays numéricos

# ============================================================
# 1. Cargar corpus procesado
print("🔄 Cargando corpus procesado...")
df = pd.read_json("data/corpus_processed_v1.jsonl", lines=True)   # Leer corpus limpio
print(f"✅ Corpus cargado con {len(df)} registros.")

X_texts = df["clean_text"]              # Textos de entrada (tweets limpios)
y = df["label_sentiment"]               # Etiquetas de salida (positivo/negativo/neutro)

# ============================================================
# 2. Funciones para generar embeddings
def get_beto_embeddings(texts):
    """
    Extrae embeddings con BETO (BERT entrenado en español).
    Se usa la representación [CLS] como vector del texto.
    """
    print("🔄 Generando embeddings con BETO...")
    tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")  # Tokenizador BETO
    model = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")          # Modelo BETO

    embeddings = []

    # Recorremos cada texto en la lista 'texts' junto con su índice 'i'
    for i, text in enumerate(texts):
        # Convertimos el texto en tensores que entiende BETO:
        # - return_tensors="pt" → formato PyTorch
        # - truncation=True → recorta si es demasiado largo
        # - max_length=128 → límite de tokens por texto
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
        # Desactivamos el cálculo de gradientes (no estamos entrenando, solo extrayendo embeddings)
        with torch.no_grad():
            # Pasamos el texto tokenizado por el modelo BETO
            outputs = model(**inputs)
    
        # Extraemos el embedding del token [CLS] (posición 0 de la secuencia),
        # que se usa como representación global del texto
        cls_embedding = outputs.last_hidden_state[:,0,:].numpy()
    
        # Aplanamos el vector y lo añadimos a la lista de embeddings
        embeddings.append(cls_embedding.flatten())
    
        # Cada 100 textos procesados mostramos un mensaje de progreso en consola
        if i % 500 == 0:
            print(f"Procesados {i} textos con BETO...")
            
    return np.array(embeddings)                                                           # Convertir a array numpy

def get_st_embeddings(texts):
    """
    Extrae embeddings con Sentence-Transformers (LaBSE).
    Recibe una lista de textos y devuelve una matriz numpy con los embeddings.
    """
    print("🔄 Generando embeddings con Sentence-Transformers...")
    model = SentenceTransformer("sentence-transformers/LaBSE")  # Cargar modelo multilingüe LaBSE
    
    # Convertir Series de pandas a lista si es necesario
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    
    # Codificar todos los textos en batch → devuelve una matriz (n_textos x dim_embedding)
    embeddings = model.encode(texts, show_progress_bar=True)
    
    print("✅ Embeddings ST generados.")
    return embeddings  # Devolver matriz numpy

# ============================================================
# 3. Dividimos corpus en train/test
print("🔄 Dividiendo en train/test...")
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    X_texts, y, test_size=0.2, random_state=42   # 80% train, 20% test
)
print("✅ División completada.")

# ============================================================
# 4. Definimos modelos (Logistic Regression y SVM)
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),  # Regresión logística
    "SVM": LinearSVC()                                        # SVM lineal
}

results = []  # Lista para guardar métricas de todos los modelos
os.makedirs("data/models_embeddings", exist_ok=True)  # Crear carpeta de modelos si no existe

# ============================================================
# 5. Entrenamos y evaluamos con BETO
print("🔄 Generando embeddings con BETO (train/test)...")
X_train_beto = get_beto_embeddings(X_train_texts)  # Embeddings BETO para train
X_test_beto = get_beto_embeddings(X_test_texts)    # Embeddings BETO para test

for name, model in models.items():                 # Iterar sobre Logistic y SVM
    print(f"🚀 Entrenando {name} con BETO...")
    model.fit(X_train_beto, y_train)               # Entrenar modelo
    y_pred = model.predict(X_test_beto)            # Predecir en test
    report = classification_report(y_test, y_pred, output_dict=True)  # Métricas

    # Guardamos métricas en lista results
    results.append({
        "model": f"{name}_BETO",
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"]
    })

    # Guardamosr modelo entrenado en archivo .pkl
    joblib.dump(model, f"data/models_embeddings/{name}_BETO.pkl")
    print(f"✅ {name} con BETO entrenado y guardado.")

# ============================================================
# 6. Entrenamos y evaluamos con Sentence-Transformers
print("🔄 Generando embeddings con Sentence-Transformers (train/test)...")
X_train_st = get_st_embeddings(X_train_texts)  # Embeddings ST para train
X_test_st = get_st_embeddings(X_test_texts)    # Embeddings ST para test

for name, model in models.items():             # Iterar sobre Logistic y SVM
    print(f"🚀 Entrenando {name} con Sentence-Transformers...")
    model.fit(X_train_st, y_train)             # Entrenar modelo
    y_pred = model.predict(X_test_st)          # Predecir en test
    report = classification_report(y_test, y_pred, output_dict=True)  # Métricas

    # Guardamos métricas en lista results
    results.append({
        "model": f"{name}_ST",
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"]
    })

    # Guardamos modelo entrenado en archivo .pkl
    joblib.dump(model, f"data/models_embeddings/{name}_ST.pkl")
    print(f"✅ {name} con ST entrenado y guardado.")
# ============================================================
# 7. Guardamos resultados en CSV
print("🔄 Guardando métricas...")
pd.DataFrame(results).to_csv("data/results_embeddings.csv", index=False)  # Exportar métricas
print("✅ Métricas guardadas en data/results_embeddings.csv")

# ============================================================
# 8. Guardamos log de experimento
print("🔄 Guardando log de experimento...")
with open("data/experiments_embeddings_log.txt", "w", encoding="utf-8") as log:
    log.write("Embeddings training completed\n")  # Mensaje de finalización
    log.write("Models: LogisticRegression, SVM\n")  # Modelos usados
    log.write("Embeddings: BETO, Sentence-Transformers\n")  # Tipos de embeddings
    log.write("Seed: 42\n")  # Semilla para reproducibilidad
print("✅ Log guardado en data/experiments_embeddings_log.txt")

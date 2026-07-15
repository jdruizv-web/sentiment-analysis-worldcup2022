#1 Importacion de Librerias

import pandas as pd               # Manipulación y análisis de datos; DataFrame, lectura/escritura CSV/JSON
import numpy as np                # Operaciones numéricas y arrays; utilidades estadísticas y matemáticas
import matplotlib.pyplot as plt   # Visualización básica; crear figuras, ejes y guardar gráficos
import seaborn as sns             # Visualizaciones estadísticas de alto nivel; estilos y gráficos complejos
import missingno as msno          # Visualización de datos faltantes (matrices, barras, dendrogramas)
import json                       # Leer/escribir JSON nativo; parseo y serialización
from pandas import json_normalize # Normalizar estructuras JSON anidadas a tablas planas (DataFrame)
import ast                        # Parseo seguro de literales Python (ej. convertir strings que contienen listas/dicts)
import re                         # Librería para trabajar con expresiones regulares

# =======================================================================
#2 Cargar dataset (ajusta el nombre según lo que descargues)
df = pd.read_csv(
    "data/twtdata.com_tweets_by_hashtag_Fifa_Xd8b0u2i9n.csv",
    sep=",",          # usa tab como separador 
    encoding="utf-8", # prueba utf-8, si falla usa latin1
    on_bad_lines="skip" # ignora filas mal formateadas
)

# =======================================================================
#3 Vista preliminar de los datos
print("\n\n=== 3. Vista preliminar de los datos ===\n")
# Revisamos las primeras filas, últimas filas y una muestra aleatoria
# para entender la estructura general del dataset.
print(df.head())


# =======================================================================
# 4. Dimensiones y tipos de datos
print("\n\n=== 4. Dimensiones y tipos de datos ===\n")
# shape devuelve número de filas y columnas.
# dtypes muestra el tipo de dato de cada columna (numérico, texto, etc.).
print("Dimensiones:", df.shape)
print("Tipos de datos:\n", df.dtypes)

# =======================================================================
# 5. Estadísticas básicas
print("\n\n=== 5. Estadísticas básicas ===\n")
# describe() genera estadísticas descriptivas de columnas numéricas
# y con include="all" también incluye categóricas.
print(df.describe(include="all"))

# =======================================================================
# 6. Análisis de datos faltantes
print("\n\n=== 6. Análisis de datos faltantes ===\n\n")
# isnull().sum() cuenta valores nulos por columna.
print("Valores nulos por columna:\n", df.isnull().sum())

# Visualización de valores faltantes con missingno
# Esto ayuda a identificar patrones de ausencia de datos.
msno.matrix(df)
plt.title("Visualización de valores faltantes")
plt.show()


# =======================================================================
# 7. Eliminar columnas con más del 95% de valores nulos + columnas sensibles

# Calculamos el porcentaje de valores nulos por columna
threshold = 0.95
null_ratio = df.isnull().mean()

# Seleccionamos las columnas cuyo porcentaje de nulos es mayor al 95%
cols_to_drop = null_ratio[null_ratio > threshold].index.tolist()

# Definimos explícitamente columnas sensibles que contienen datos personales
cols_sensitive = ["id", "name", "screen_name", "location"]

# Unimos ambas listas (nulos + sensibles)
cols_to_drop = list(set(cols_to_drop + cols_sensitive))

# Eliminamos esas columnas del dataframe
df = df.drop(columns=cols_to_drop, errors="ignore")
print("Columnas eliminadas:", cols_to_drop)

# =======================================================================
# 8. Filtrar tweets en español y eliminar duplicados/retweets

# Nos quedamos solo con los tweets cuyo idioma es español
df_es = df[df["lang"] == "es"].copy()

# Eliminamos duplicados basados en el texto completo del tweet
df_es = df_es.drop_duplicates(subset=["full_text"])

# Eliminamos los retweets (columna booleana 'retweeted')
df_es = df_es[df_es["retweeted"] == False]

print("Tweets en español después de limpieza:", len(df_es))

# =======================================================================
# 9. Anonimización de identificadores

import hashlib  # Librería para aplicar funciones hash

# Definimos función que convierte un texto en un hash SHA-256
def hash_text(x):
    return hashlib.sha256(str(x).encode("utf-8")).hexdigest()

# Creamos un identificador interno estable combinando texto + fecha de creación
df_es["tweet_id_hashed"] = (df_es["full_text"] + df_es["created_at"]).apply(hash_text)

# =======================================================================
# 10. Guardar corpus canónico

# Definimos la ruta de salida del corpus canónico
output_path = "data/corpus_canonical_v1.jsonl"

# Guardamos el dataframe en formato JSONL (una línea por registro)
df_es.to_json(output_path, orient="records", lines=True, force_ascii=False)

# =======================================================================
# 11. Generar checksum del archivo

# Función para calcular el hash SHA-256 de un archivo completo
def file_checksum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

# Calculamos el checksum del corpus guardado
checksum = file_checksum(output_path)

# Guardamos el checksum en un archivo de texto
with open("data/checksum.txt", "w") as f:
    f.write("sha256:" + checksum)

# Mensaje final de confirmación
print("Corpus canónico guardado con", len(df_es), "tweets en español y IDs anonimizados.")

print("\n\n=== Checksum (SHA-256) ===\n")
print("sha256:",checksum)

# 12. =======================================================================
# Verificación final del corpus

# Mostrar las columnas restantes después de la limpieza
print("\n=== Columnas finales del corpus ===")
print(df_es.columns.tolist())

# Mostrar las primeras filas para verificar estructura y anonimización
print("\n=== Primeras filas del corpus limpio ===")
print(df_es.head())

# Confirmar que las columnas sensibles ya no están
cols_sensitive = ["id", "name", "screen_name", "location"]
for col in cols_sensitive:
    if col in df_es.columns:
        print(f"⚠️ La columna {col} todavía está presente.")
    else:
        print(f"✔ La columna {col} fue eliminada correctamente.")

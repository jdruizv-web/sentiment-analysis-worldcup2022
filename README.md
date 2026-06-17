# 📘 Trabajo Fin de Máster — Mundial FIFA 2022

✨ **Autor:** Juan Diego Ruiz Valverde  
📅 **Fecha:** Abril 2026  
📊 **Tema:** Análisis de sentimiento en tweets en español durante el Mundial FIFA 2022  
🏫 **Institución:** Universidad Internacional de La Rioja (UNIR)  
👨‍🏫 **Director:** Félix Ladstätter  

---

## 📂 Estructura del repositorio

- **`data/`**  
  Carpeta local para datos — **no incluida en el repositorio**.  
  Los datos procesados están disponibles en Zenodo: [DOI pendiente]  
  - `corpus_canonical_v1.jsonl` — corpus en español tras ingestión y anonimización  
  - `corpus_processed_v1.jsonl` — corpus tras preprocesado completo  
  - `corpus_labeled_v1.jsonl` — corpus con etiquetas de sentimiento  
  - `sample_processed.csv` — muestra aleatoria de 500 registros  
  - `checksum.txt` — hash SHA-256 para verificar integridad del corpus  
  - `results_baselines.csv` — métricas de Naïve Bayes, Logistic Regression y SVM  
  - `results_embeddings.csv` — métricas de modelos con BETO y Sentence-Transformers  

- **`src/`**  
  Scripts principales del pipeline, ejecutables en orden.  
  - `ingest.py` — carga, limpieza inicial y anonimización del corpus crudo  
  - `preprocess.py` — preprocesado lingüístico (normalización, tokenización, emojis, hashtags)  
  - `preprocess_config.yml` — parámetros centralizados del pipeline  
  - `label_sentiment.py` — etiquetado automático con pysentimiento y cardiffnlp  
  - `baselines.py` — modelos clásicos con TF-IDF (NB, LR, SVM)  
  - `embeddings.py` — modelos con embeddings BETO y Sentence-Transformers  

- **`notebooks/`**  
  Cuadernos Jupyter para exploración y análisis.  
  - `eda_corpus.ipynb` — análisis exploratorio del corpus  

- **`models/`**  
  Modelos entrenados (generados al ejecutar el pipeline).  
  - `models_baselines/` — modelos clásicos (.pkl)  
  - `models_embeddings/` — modelos con embeddings (.pkl)  

- **`anexos/`**  
  Documentación normativa y técnica.  
  - `Anexo_1_Registro_de_tratamiento.pdf` — cumplimiento RGPD y Ley 8968  
  - `Anexo_2_DPIA_Analisis_de_riesgos.pdf` — evaluación de impacto y mitigaciones  
  - `Anexo_3_Checklist_Reproducibilidad.md` — comandos, seeds y trazabilidad  

- **`docs/`**  
  Documentación académica.  
  - `TFM_documento_principal.pdf`  
  - `referencias_bibliograficas.bib`  

---

## 🚀 Cómo reproducir el pipeline

### 1. Requisitos previos
```bash
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

### 2. Datos
Descarga el corpus desde Zenodo [DOI pendiente] y coloca los archivos en `data/`.

### 3. Ejecutar en orden
```bash
python src/ingest.py
python src/preprocess.py
python src/baselines.py
python src/embeddings.py
```

> `preprocess.py` llama automáticamente a `label_sentiment.py` según el parámetro  
> `label_strategy` definido en `preprocess_config.yml`.

### 4. Parámetros del pipeline
Todos los parámetros configurables están en `src/preprocess_config.yml`:
- `label_strategy: "both"` — etiquetado con pysentimiento + cardiffnlp + Cohen's Kappa
- `emoji_strategy: "map"` — emojis convertidos a palabras sentimentales
- `lemmatize: false` — tokens originales (recomendado para modelos Transformer)


### 5. Exploración opcional
El notebook `notebooks/eda_corpus.ipynb` contiene el análisis exploratorio del corpus.  
Puede ejecutarse después de `ingest.py` o `preprocess.py`, pero no es requisito para reproducir los resultados.

---

## 📊 Resultados principales

| Modelo | F1 (weighted) |
|--------|--------------|
| Naïve Bayes + TF-IDF | pendiente |
| Logistic Regression + TF-IDF | pendiente |
| SVM + TF-IDF | pendiente |
| Logistic Regression + BETO | pendiente |
| SVM + BETO | pendiente |
| Logistic Regression + ST | pendiente |
| SVM + ST | pendiente |

*Tabla completa en `data/results_baselines.csv` y `data/results_embeddings.csv`*

---

## 📌 Reproducibilidad y ética

- Semilla fija `seed: 42` en todos los experimentos
- IDs de usuario anonimizados mediante hash SHA-256
- Corpus limitado a texto en español, sin datos sensibles
- Acuerdo inter-anotador automático: Cohen's Kappa κ = 0.6356 (sustancial)
- Marco normativo: RGPD, LOPDGDD y Ley 8968 (Costa Rica)

---

## 📄 Licencia

Código bajo licencia [MIT](LICENSE).  
Datos bajo licencia [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).

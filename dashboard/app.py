# ============================================================
# Dashboard interactivo — Mundial FIFA 2022
# Análisis de Sentimiento en Twitter/X
# TFM — Juan Diego Ruiz Valverde — UNIR 2026
# ============================================================

# ── Importación de librerías ──────────────────────────────────
import streamlit as st              # Framework principal del dashboard web
import pandas as pd                 # Manejo de DataFrames y lectura de archivos
import altair as alt                # Gráficos interactivos declarativos
from wordcloud import WordCloud     # Generación de nubes de palabras
import matplotlib.pyplot as plt     # Visualización base para dona y nube
import matplotlib.patches as mpatches  # Parches de color para la leyenda de la dona
import numpy as np                  # Operaciones numéricas (reservado para extensiones)
import os
import base64

# Rutas absolutas calculadas desde la ubicación del script
DASHBOARD_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_DIR        = os.path.dirname(DASHBOARD_DIR)
LOGO_PATH       = os.path.join(DASHBOARD_DIR, "logo_qatar2022.png")
MASCOTA_PATH    = os.path.join(DASHBOARD_DIR, "mascota.jpg")
DATA_PATH       = os.path.join(REPO_DIR, "data", "corpus_labeled_v1.jsonl")
BASELINES_PATH  = os.path.join(REPO_DIR, "results", "results_baselines.csv")
EMBEDDINGS_PATH = os.path.join(REPO_DIR, "results", "results_embeddings.csv")

# ============================================================
# 1. CONFIGURACIÓN INICIAL DE LA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Sentimiento Mundial 2022",   # Título que aparece en la pestaña del navegador
    layout="wide",                           # Usar todo el ancho de pantalla disponible
    initial_sidebar_state="expanded"         # Mostrar sidebar desplegado al cargar
)

# ============================================================
# 2. PALETA DE COLORES QATAR 2022
# Todos los colores se definen aquí como constantes para
# mantener consistencia y facilitar cambios globales.
# ============================================================
COLOR_GRANATE_OSC  = "#4A0E23"   # Granate oscuro — sidebar y cabeceras
COLOR_GRANATE_MED  = "#8A1538"   # Granate medio — fondo principal del dashboard
COLOR_GRANATE_CLA  = "#B01D45"   # Granate claro — acentos secundarios
COLOR_DORADO       = "#D4AF37"   # Dorado Qatar — separadores y destacados
COLOR_ARENA        = "#F2E9D8"   # Arena claro — texto principal sobre fondos oscuros
COLOR_ARENA_OSC    = "#D9C9A8"   # Arena oscuro — texto secundario y etiquetas de ejes

COLOR_POSITIVO     = "#4CAF82"   # Verde menta — sentimiento positivo (contrasta bien)
COLOR_NEGATIVO     = "#E8574A"   # Rojo coral — sentimiento negativo (diferenciado del granate)
COLOR_NEUTRO       = "#A8B8C8"   # Azul grisáceo — sentimiento neutro (visualmente neutro)

# ============================================================
# 3. CSS GLOBAL
# Se inyecta CSS personalizado en la app de Streamlit para
# sobreescribir estilos por defecto y aplicar la identidad visual.
# unsafe_allow_html=True es necesario para inyectar HTML/CSS directo.
# ============================================================
st.markdown(f"""
<style>
    /* ── Fondo general del área de contenido ── */
    div[data-testid="stAppViewContainer"],
    div[data-testid="stAppViewBlockContainer"] {{
        background-color: {COLOR_GRANATE_MED} !important;
    }}

    /* ── Sidebar: color de fondo oscuro y borde dorado ── */
    section[data-testid="stSidebar"] {{
        background-color: {COLOR_GRANATE_OSC} !important;
        border-right: 2px solid {COLOR_DORADO} !important;
    }}
    /* Todo el texto dentro del sidebar en color arena */
    section[data-testid="stSidebar"] * {{
        color: {COLOR_ARENA} !important;
    }}

    /* ── Tipografía general del área central ── */
    div[data-testid="stAppViewContainer"] p,
    div[data-testid="stAppViewContainer"] label,
    div[data-testid="stAppViewContainer"] span,
    div[data-testid="stAppViewContainer"] li,
    div[data-testid="stAppViewContainer"] h1,
    div[data-testid="stAppViewContainer"] h2,
    div[data-testid="stAppViewContainer"] h3,
    div[data-testid="stAppViewContainer"] h4 {{
        color: {COLOR_ARENA} !important;
    }}

    /* ── Línea separadora horizontal dorada ── */
    .divider {{
        border: none;
        border-top: 1px solid {COLOR_DORADO};   /* Línea dorada */
        margin: 24px 0;                          /* Espaciado vertical */
        opacity: 0.5;                            /* Semi-transparente para suavizar */
    }}

    /* ── Tarjetas KPI: gradiente oscuro con borde dorado ── */
    .kpi-card {{
        background: linear-gradient(135deg, {COLOR_GRANATE_OSC}, {COLOR_GRANATE_MED}); /* Gradiente diagonal */
        border: 1px solid {COLOR_DORADO};        /* Borde dorado fino */
        border-radius: 12px;                     /* Esquinas redondeadas */
        padding: 20px 16px;                      /* Espaciado interno */
        text-align: center;                      /* Centrar contenido */
        box-shadow: 0 4px 16px rgba(0,0,0,0.35); /* Sombra suave */
    }}
    .kpi-emoji  {{ font-size: 44px; line-height: 1.1; }}                        /* Emoji grande */
    .kpi-label  {{ font-size: 13px; color: {COLOR_ARENA_OSC}; letter-spacing: 1px;
                   text-transform: uppercase; margin: 6px 0 4px; }}             /* Etiqueta en mayúsculas */
    .kpi-value  {{ font-size: 32px; font-weight: 700; color: {COLOR_ARENA}; }}  /* Número principal grande */
    .kpi-pct    {{ font-size: 12px; color: {COLOR_DORADO}; margin-top: 2px; }}  /* Porcentaje en dorado */

    /* ── Cabecera de sección: texto dorado con borde inferior ── */
    .section-header {{
        font-size: 15px;
        font-weight: 600;
        color: {COLOR_DORADO};
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin: 32px 0 12px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid {COLOR_DORADO}44;  /* 44 = 27% opacidad en hex */
    }}

    /* ── Tarjetas de tweets virales ── */
    .viral-row {{
        background: {COLOR_GRANATE_OSC};       /* Fondo granate oscuro */
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        border-left: 4px solid {COLOR_DORADO}; /* Acento dorado a la izquierda */
    }}
    .viral-text {{ font-size: 13px; color: {COLOR_ARENA}; }}           /* Texto del tweet */
    .viral-meta {{ font-size: 11px; color: {COLOR_ARENA_OSC}; margin-top: 6px; }} /* Metadatos: RTs, likes */

    /* ── Dataframe con esquinas redondeadas ── */
    .stDataFrame {{ border-radius: 8px; overflow: hidden; }}

    /* ── Selectbox y multiselect con fondo oscuro ── */
    .stMultiSelect div, .stSelectbox div {{
        background-color: {COLOR_GRANATE_OSC} !important;
        color: {COLOR_ARENA} !important;
    }}

    /* ── Ocultar el menú hamburguesa y el footer de Streamlit ── */
    #MainMenu, footer {{ visibility: hidden; }}

    /* ── Prevenir que los SVG se expandan de forma inesperada ── */
    svg {{ width: auto !important; height: auto !important; }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 4. CARGAR DATOS
# @st.cache_data almacena los resultados en caché para que
# la app no relean los archivos cada vez que el usuario
# interactúa con un filtro — mejora el rendimiento.
# ============================================================
@st.cache_data
def load_data():
    # Lee el corpus etiquetado con sentimientos (generado por preprocess.py + label_sentiment.py)
    df =  pd.read_json(DATA_PATH, lines=True) #pd.read_json("data/corpus_labeled_v1.jsonl", lines=True)

    # Lee las métricas de los modelos clásicos TF-IDF (generado por baselines.py)
    results_base = pd.read_csv(BASELINES_PATH) #pd.read_csv("results/results_baselines.csv")

    # Lee las métricas de los modelos con embeddings BETO y ST (generado por embeddings.py)
    results_emb = pd.read_csv(EMBEDDINGS_PATH) #pd.read_csv("results/results_embeddings.csv")

    # Convierte la columna de fechas a formato datetime de pandas
    # errors="coerce" convierte valores inválidos en NaT en lugar de lanzar error
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    # Elimina filas donde la fecha no pudo convertirse (NaT)
    df = df.dropna(subset=["created_at"])

    return df, results_base, results_emb

# Ejecuta la carga de datos (desde caché si ya se cargó antes)
df, results_base, results_emb = load_data()

# ============================================================
# 5. SIDEBAR — FILTROS INTERACTIVOS
# El bloque "with st.sidebar" coloca todo su contenido
# en el panel lateral izquierdo del dashboard.
# ============================================================
with st.sidebar:
    # Cabecera del sidebar con logo emoji y título del proyecto
    st.image(LOGO_PATH, use_container_width=True)
    st.markdown(f"""
        <div style='text-align:center; padding: 8px 0 20px 0;'>
            <div style='font-size:15px; font-weight:700; color:{COLOR_DORADO}; letter-spacing:1px;'>
                MUNDIAL FIFA 2022
            </div>
            <div style='font-size:11px; color:{COLOR_ARENA_OSC}; margin-top:4px;'>
                Análisis de Sentimiento · Twitter/X
            </div>
        </div>
        <hr style='border-color:{COLOR_DORADO}44; margin-bottom:20px;'>
    """, unsafe_allow_html=True)

    st.markdown("**Filtrar por sentimiento**")
    # Multiselect: el usuario puede elegir uno, dos o los tres sentimientos
    # default= muestra los tres seleccionados al inicio
    sent_filter = st.multiselect(
        label="",
        options=["positivo", "negativo", "neutro"],
        default=["positivo", "negativo", "neutro"]
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Rango horario**")

    # Calcula el rango real de horas presentes en el corpus
    hora_min = int(df["created_at"].dt.hour.min())  # Hora más temprana del corpus
    hora_max = int(df["created_at"].dt.hour.max())  # Hora más tardía del corpus

    # Slider de rango: devuelve una tupla (hora_inicio, hora_fin)
    hora_range = st.slider("", hora_min, hora_max, (hora_min, hora_max))

    st.markdown("<br>", unsafe_allow_html=True)
    # Ficha informativa estática con datos clave del corpus y metodología
    st.markdown(f"""
        <div style='font-size:11px; color:{COLOR_ARENA_OSC}; line-height:1.6;'>
            📅 11 diciembre 2022<br>
            🐦 {len(df):,} tweets en español<br>
            🤖 pysentimiento + cardiffnlp<br>
            κ = 0.6356 (acuerdo sustancial)
        </div>
    """, unsafe_allow_html=True)

# ============================================================
# 6. APLICAR FILTROS AL DATAFRAME
# Filtra df según las selecciones del usuario en el sidebar.
# .copy() evita el SettingWithCopyWarning al modificar df_f más adelante.
# ============================================================
df_f = df[
    (df["label_sentiment"].isin(sent_filter)) &          # Solo sentimientos seleccionados
    (df["created_at"].dt.hour >= hora_range[0]) &        # Desde la hora mínima del slider
    (df["created_at"].dt.hour <= hora_range[1])          # Hasta la hora máxima del slider
].copy()

# ============================================================
# 7. HEADER PRINCIPAL DEL DASHBOARD
# Muestra logo a la izquierda y título a la derecha.
# st.columns([1, 6]) divide el ancho en proporción 1:6.
# ============================================================

st.markdown(f"""
    <div style='padding: 8px 0 24px 0;'>
        <h1 style='color:{COLOR_ARENA}; font-size:28px; margin:0; font-weight:700;'>
            Análisis de Sentimiento — Copa Mundial FIFA 2022
        </h1>
        <p style='color:{COLOR_ARENA_OSC}; font-size:14px; margin: 6px 0 0 0;'>
            TFM · Juan Diego Ruiz Valverde · UNIR 2026 &nbsp;·&nbsp;
            Datos: 11 dic 2022 · {len(df_f):,} tweets filtrados de {len(df):,} totales
        </p>
    </div>
""", unsafe_allow_html=True)

# ============================================================
# 8. KPIs — TARJETAS DE MÉTRICAS PRINCIPALES
# Calcula los valores sobre el corpus completo (sin filtros)
# para que las tarjetas siempre muestren el total real del corpus.
# ============================================================
total     = len(df)                                                    # Total de tweets del corpus
positivos = (df["label_sentiment"] == "positivo").sum()               # Tweets con sentimiento positivo
negativos = (df["label_sentiment"] == "negativo").sum()               # Tweets con sentimiento negativo
neutros   = (df["label_sentiment"] == "neutro").sum()                 # Tweets con sentimiento neutro
acuerdo   = (df["label_sentiment"] == df["label_cardiffnlp"]).mean() * 100  # % de tweets donde ambos modelos coinciden
conf_avg  = df["label_confidence"].mean() * 100                       # Confianza media de pysentimiento en %

# Divide el ancho en 5 columnas iguales para las 5 tarjetas
col1, col2, col3, col4, col5 = st.columns(5)

# Tarjeta 1: tweets positivos
with col1:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-emoji">😊</div>
        <div class="kpi-label">Positivos</div>
        <div class="kpi-value">{positivos:,}</div>
        <div class="kpi-pct">{positivos/total*100:.1f}% del corpus</div>
    </div>""", unsafe_allow_html=True)

# Tarjeta 2: tweets negativos
with col2:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-emoji">😡</div>
        <div class="kpi-label">Negativos</div>
        <div class="kpi-value">{negativos:,}</div>
        <div class="kpi-pct">{negativos/total*100:.1f}% del corpus</div>
    </div>""", unsafe_allow_html=True)

# Tarjeta 3: tweets neutros
with col3:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-emoji">😐</div>
        <div class="kpi-label">Neutros</div>
        <div class="kpi-value">{neutros:,}</div>
        <div class="kpi-pct">{neutros/total*100:.1f}% del corpus</div>
    </div>""", unsafe_allow_html=True)

# Tarjeta 4: porcentaje de acuerdo entre pysentimiento y cardiffnlp
with col4:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-emoji">🤝</div>
        <div class="kpi-label">Acuerdo modelos</div>
        <div class="kpi-value">{acuerdo:.1f}%</div>
        <div class="kpi-pct">κ = 0.6356 sustancial</div>
    </div>""", unsafe_allow_html=True)

# Tarjeta 5: confianza media del modelo principal (pysentimiento)
with col5:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-emoji">🎯</div>
        <div class="kpi-label">Confianza media</div>
        <div class="kpi-value">{conf_avg:.1f}%</div>
        <div class="kpi-pct">pysentimiento</div>
    </div>""", unsafe_allow_html=True)

# Línea separadora dorada entre secciones
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ============================================================
# 9. SERIE TEMPORAL + GRÁFICO DE DONA
# Columna ancha (3) para la serie temporal, estrecha (1) para la dona.
# ============================================================
st.markdown("<div class='section-header'>Distribución y evolución temporal</div>", unsafe_allow_html=True)

col_time, col_dona = st.columns([3, 1])   # Proporción 3:1 entre serie temporal y dona

with col_time:
    # Agrupa tweets por ventanas de 30 minutos y sentimiento
    # pd.Grouper con freq="30min" crea intervalos regulares de media hora
    df_time = df_f.groupby(
        [pd.Grouper(key="created_at", freq="30min"), "label_sentiment"]
    ).size().reset_index(name="count")   # .size() cuenta tweets por grupo; reset_index() convierte a columnas

    # Gráfico de áreas superpuestas — mark_area con stack=None superpone sin apilar
    chart_time = alt.Chart(df_time).mark_area(
        interpolate="monotone",   # Curva suavizada (no escalones angulosos)
        opacity=0.75,             # Semi-transparente para ver superposiciones
        strokeWidth=2.5           # Grosor del borde superior del área
    ).encode(
        x=alt.X("created_at:T",   # Eje X: tiempo (:T indica tipo temporal)
                 axis=alt.Axis(labelColor=COLOR_ARENA_OSC, gridColor="#FFFFFF11",
                               tickColor=COLOR_ARENA_OSC, title=None,
                               format="%H:%M")),   # Formato hora:minuto en el eje
        y=alt.Y("count:Q",        # Eje Y: conteo de tweets (:Q indica cuantitativo)
                 stack=None,       # Sin apilamiento: cada sentimiento tiene su propia escala
                 axis=alt.Axis(labelColor=COLOR_ARENA_OSC, gridColor="#FFFFFF11",
                               tickColor=COLOR_ARENA_OSC, title="Tweets / 30 min")),
        color=alt.Color("label_sentiment:N",   # Color según sentimiento (:N nominal/categórico)
                         scale=alt.Scale(
                             domain=["positivo", "negativo", "neutro"],
                             range=[COLOR_POSITIVO, COLOR_NEGATIVO, COLOR_NEUTRO]
                         ),
                         legend=alt.Legend(
                             title=None,
                             orient="top",           # Leyenda arriba del gráfico
                             labelColor=COLOR_ARENA,
                             symbolType="square"     # Cuadrados en lugar de círculos
                         )),
        tooltip=["created_at:T", "label_sentiment:N", "count:Q"]   # Info al pasar el ratón
    ).properties(
        height=260,
        background="transparent",   # Sin fondo para que se vea el granate del dashboard
        title=alt.TitleParams("Actividad por sentimiento cada 30 minutos",
                               color=COLOR_ARENA_OSC, fontSize=12)
    ).configure_view(
        strokeOpacity=0   # Elimina el borde del área del gráfico
    )

    st.altair_chart(chart_time, use_container_width=True)   # Ocupa todo el ancho de la columna

with col_dona:
    # Gráfico de dona con matplotlib (Altair no tiene tipo dona nativo)
    sizes  = [positivos, negativos, neutros]           # Tamaño de cada sector
    colors = [COLOR_POSITIVO, COLOR_NEGATIVO, COLOR_NEUTRO]  # Color de cada sector
    labels = ["Positivo", "Negativo", "Neutro"]        # Etiquetas de la leyenda

    fig_dona, ax_dona = plt.subplots(figsize=(3.5, 3.5))   # Figura cuadrada pequeña
    fig_dona.patch.set_alpha(0)     # Fondo de la figura transparente
    ax_dona.set_facecolor("none")   # Fondo del eje transparente

    # pie() con wedgeprops width<1 crea el efecto dona (hueco central)
    wedges, _ = ax_dona.pie(
        sizes, colors=colors, startangle=90,   # Empieza desde las 12 en punto
        wedgeprops=dict(width=0.55, edgecolor=COLOR_GRANATE_MED, linewidth=2)
        # width=0.55 define el grosor del anillo; edgecolor separa visualmente los sectores
    )

    # Texto en el centro de la dona: total de tweets
    ax_dona.text(0, 0, f"{total:,}\ntweets",
                  ha="center", va="center", fontsize=11,
                  color=COLOR_ARENA, fontweight="bold")

    # Leyenda manual con parches de color y porcentajes
    legend_patches = [mpatches.Patch(color=c, label=f"{l}\n{s/total*100:.1f}%")
                      for c, l, s in zip(colors, labels, sizes)]
    ax_dona.legend(handles=legend_patches, loc="lower center",
                    bbox_to_anchor=(0.5, -0.22),   # Posición debajo del gráfico
                    ncol=3,        # 3 columnas en la leyenda (una por sentimiento)
                    fontsize=8, frameon=False,      # Sin borde en la leyenda
                    labelcolor=COLOR_ARENA)

    plt.tight_layout()
    st.pyplot(fig_dona, transparent=True)   # transparent=True respeta el fondo del dashboard

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ============================================================
# 10. HEATMAP DE ACUERDO ENTRE MODELOS + HISTOGRAMA DE CONFIANZA
# Sección clave para la validación metodológica del TFM.
# ============================================================
st.markdown("<div class='section-header'>Validación del etiquetado automático</div>", unsafe_allow_html=True)

col_heat, col_conf = st.columns([1, 1])   # Dos columnas iguales

with col_heat:
    st.markdown(f"<p style='font-size:12px; color:{COLOR_ARENA_OSC};'>Matriz de acuerdo entre pysentimiento y cardiffnlp</p>", unsafe_allow_html=True)

    cats = ["positivo", "negativo", "neutro"]   # Categorías en orden fijo para ambos ejes
    matrix_data = []

    # Construye la matriz de confusión entre los dos anotadores automáticos
    # Para cada combinación (fila=pysentimiento, columna=cardiffnlp) cuenta cuántos tweets caen ahí
    for r in cats:
        for c in cats:
            n = ((df["label_sentiment"] == r) & (df["label_cardiffnlp"] == c)).sum()
            matrix_data.append({"pysentimiento": r, "cardiffnlp": c, "count": int(n)})

    df_matrix = pd.DataFrame(matrix_data)

    # Calcula el porcentaje por fila (sobre el total etiquetado por pysentimiento)
    # transform("sum") replica el total de la fila para cada celda del grupo
    total_per_row = df_matrix.groupby("pysentimiento")["count"].transform("sum")
    df_matrix["pct"] = (df_matrix["count"] / total_per_row * 100).round(1)

    # mark_rect dibuja rectángulos coloreados — base del heatmap
    heat = alt.Chart(df_matrix).mark_rect(cornerRadius=4).encode(
        x=alt.X("cardiffnlp:N",
                  sort=cats,
                  axis=alt.Axis(labelColor=COLOR_ARENA, title="cardiffnlp",
                                titleColor=COLOR_ARENA_OSC, labelAngle=0)),
        y=alt.Y("pysentimiento:N",
                  sort=cats,
                  axis=alt.Axis(labelColor=COLOR_ARENA, title="pysentimiento",
                                titleColor=COLOR_ARENA_OSC)),
        color=alt.Color("pct:Q",
                         scale=alt.Scale(scheme="goldorange"),   # Escala de dorado a naranja
                         legend=None),
        tooltip=["pysentimiento:N", "cardiffnlp:N",
                  alt.Tooltip("count:Q", title="Tweets"),
                  alt.Tooltip("pct:Q", title="% fila", format=".1f")]
    )

    # Capa de texto superpuesta al heatmap con el porcentaje en cada celda
    text_heat = alt.Chart(df_matrix).mark_text(fontSize=14, fontWeight="bold").encode(
        x=alt.X("cardiffnlp:N", sort=cats),
        y=alt.Y("pysentimiento:N", sort=cats),
        text=alt.Text("pct:Q", format=".0f"),   # Porcentaje sin decimales
        # Si el porcentaje > 60 usa texto oscuro (mejor contraste sobre celdas claras)
        # Si el porcentaje <= 60 usa texto claro (mejor contraste sobre celdas oscuras)
        color=alt.condition(
            alt.datum.pct > 60,
            alt.value(COLOR_GRANATE_OSC),
            alt.value(COLOR_ARENA)
        )
    )

    # Combina las dos capas (rectángulos + texto) con el operador +
    st.altair_chart(
        (heat + text_heat).properties(height=220, background="transparent")
        .configure_view(strokeOpacity=0),
        use_container_width=True
    )

with col_conf:
    st.markdown(f"<p style='font-size:12px; color:{COLOR_ARENA_OSC};'>Distribución de confianza por modelo</p>", unsafe_allow_html=True)

    # Combina las columnas de confianza de ambos modelos en un único DataFrame largo
    # Esto permite graficar ambas distribuciones en el mismo histograma con colores distintos
    df_conf = pd.DataFrame({
        "confianza": pd.concat([df["label_confidence"], df["label_conf_cardiffnlp"]]),
        "modelo": ["pysentimiento"] * len(df) + ["cardiffnlp"] * len(df)
    })

    # Histograma con barras semi-transparentes superpuestas por modelo
    hist_conf = alt.Chart(df_conf).mark_bar(opacity=0.75, binSpacing=1).encode(
        x=alt.X("confianza:Q",
                  bin=alt.Bin(maxbins=20),   # Divide el rango [0,1] en hasta 20 bins
                  axis=alt.Axis(labelColor=COLOR_ARENA_OSC, title="Confianza",
                                titleColor=COLOR_ARENA_OSC, format=".1f")),
        y=alt.Y("count():Q",   # count() cuenta tweets en cada bin automáticamente
                  axis=alt.Axis(labelColor=COLOR_ARENA_OSC, title="Tweets",
                                titleColor=COLOR_ARENA_OSC)),
        color=alt.Color("modelo:N",
                         scale=alt.Scale(
                             domain=["pysentimiento", "cardiffnlp"],
                             range=[COLOR_DORADO, COLOR_GRANATE_CLA]   # Dorado vs granate claro
                         ),
                         legend=alt.Legend(
                             title=None, orient="top",
                             labelColor=COLOR_ARENA
                         )),
        tooltip=["modelo:N", alt.Tooltip("count():Q", title="Tweets")]
    ).properties(height=220, background="transparent").configure_view(strokeOpacity=0)

    st.altair_chart(hist_conf, use_container_width=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ============================================================
# 11. NUBE DE PALABRAS + GRÁFICO DE DISPOSITIVOS
# ============================================================
st.markdown("<div class='section-header'>Vocabulario y plataformas</div>", unsafe_allow_html=True)

col_wc, col_src = st.columns([3, 2])   # Nube más ancha que el gráfico de dispositivos

with col_wc:
    # Selectbox para que el usuario elija qué sentimiento visualizar en la nube
    sent_sel = st.selectbox(
        "Sentimiento para la nube de palabras:",
        ["positivo", "negativo", "neutro"],
        index=1   # Por defecto muestra "negativo" (más informativo visualmente)
    )

    # Busca la primera columna de texto disponible en el DataFrame
    # Prioriza clean_text (texto limpio y normalizado) sobre full_text (texto crudo)
    col_texto = next((c for c in ["clean_text", "full_text"] if c in df.columns), None)

    if col_texto:
        # Une todos los textos del sentimiento seleccionado en una sola cadena
        textos = " ".join(df[df["label_sentiment"] == sent_sel][col_texto].astype(str).tolist())

        # Mapea cada sentimiento a una función de color fija para la nube
        # Las funciones lambda aceptan *args y **kwargs porque WordCloud pasa parámetros internos
        colores = {
            "positivo": lambda *a, **k: COLOR_POSITIVO,
            "negativo": lambda *a, **k: COLOR_NEGATIVO,
            "neutro":   lambda *a, **k: COLOR_NEUTRO
        }

        # Genera la nube de palabras con el color del sentimiento seleccionado
        wc = WordCloud(
            width=700, height=320,
            background_color=COLOR_GRANATE_MED,      # Fondo granate del dashboard
            color_func=colores[sent_sel],             # Función de color según sentimiento
            max_words=80,                             # Máximo 80 palabras en la nube
            stopwords={"rt", "https", "co", "que", "de", "el", "la", "en",
                        "los", "las", "un", "una", "es", "y", "a", "con",
                        "se", "del", "al", "por", "no", "le", "lo", "su"},
            # Lista de palabras vacías (stopwords) a excluir — palabras sin contenido semántico
            prefer_horizontal=0.85,   # 85% de las palabras en horizontal (más legible)
            collocations=False        # No agrupa bigramas automáticamente (evita ruido)
        ).generate(textos)

        # Renderiza la nube con matplotlib sobre fondo granate
        fig_wc, ax_wc = plt.subplots(figsize=(8, 3.5))
        fig_wc.patch.set_facecolor(COLOR_GRANATE_MED)   # Fondo de la figura
        ax_wc.set_facecolor(COLOR_GRANATE_MED)           # Fondo del eje
        ax_wc.imshow(wc, interpolation="bilinear")       # Muestra la imagen de la nube con suavizado
        ax_wc.axis("off")                                # Oculta ejes y bordes
        plt.tight_layout(pad=0)                          # Elimina márgenes internos
        st.pyplot(fig_wc, transparent=False)             # transparent=False conserva el fondo granate

with col_src:
    # Cuenta los 7 dispositivos más frecuentes en el corpus filtrado
    top_src = df_f["source"].value_counts().head(7).reset_index()
    top_src.columns = ["dispositivo", "tweets"]   # Renombra columnas para claridad

    # Gráfico de barras horizontales ordenadas de mayor a menor
    chart_src = alt.Chart(top_src).mark_bar(
        cornerRadiusTopRight=4,    # Esquina redondeada solo en el extremo derecho
        cornerRadiusBottomRight=4,
        color=COLOR_DORADO         # Barras doradas
    ).encode(
        y=alt.Y("dispositivo:N",
                  sort="-x",   # Ordena por valor de X descendente (mayor arriba)
                  axis=alt.Axis(labelColor=COLOR_ARENA, title=None, labelLimit=160)),
        x=alt.X("tweets:Q",
                  axis=alt.Axis(labelColor=COLOR_ARENA_OSC, title="Tweets",
                                titleColor=COLOR_ARENA_OSC, grid=False)),
        tooltip=["dispositivo:N", "tweets:Q"]
    ).properties(
        height=260,
        background="transparent",
        title=alt.TitleParams("Tweets por dispositivo",
                               color=COLOR_ARENA_OSC, fontSize=12)
    ).configure_view(strokeOpacity=0)

    st.altair_chart(chart_src, use_container_width=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ============================================================
# 12. TWEETS MÁS VIRALES POR SENTIMIENTO
# Muestra los 5 tweets con más retweets del sentimiento elegido.
# ============================================================
st.markdown("<div class='section-header'>Tweets más retuiteados</div>", unsafe_allow_html=True)

# Radio buttons horizontales para elegir el sentimiento (más compacto que un selectbox)
sent_viral = st.radio(
    "Sentimiento:",
    ["positivo", "negativo", "neutro"],
    horizontal=True,
    index=1   # Por defecto "negativo" — suele ser el más interesante en deportes
)

# Filtra por sentimiento y toma los 5 con mayor retweet_count
top_viral = (
    df[df["label_sentiment"] == sent_viral]
    .nlargest(5, "retweet_count")             # nlargest devuelve las N filas con mayor valor
    [["full_text", "retweet_count", "favorite_count", "label_confidence"]]
)

# Renderiza cada tweet como una tarjeta HTML con borde dorado
for _, row in top_viral.iterrows():
    # Trunca el texto a 220 caracteres para que las tarjetas sean uniformes
    texto = row["full_text"][:220] + "..." if len(row["full_text"]) > 220 else row["full_text"]
    conf_pct = f"{row['label_confidence']*100:.0f}%"   # Confianza como porcentaje entero
    st.markdown(f"""
        <div class="viral-row">
            <div class="viral-text">{texto}</div>
            <div class="viral-meta">
                🔁 {int(row['retweet_count']):,} retweets &nbsp;·&nbsp;
                ❤️ {int(row['favorite_count']):,} likes &nbsp;·&nbsp;
                🎯 Confianza: {conf_pct}
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ============================================================
# 13. COMPARATIVA DE RENDIMIENTO DE MODELOS
# Muestra las tablas de métricas con degradado de color en F1
# y un gráfico de barras comparativo de todos los modelos.
# ============================================================
st.markdown("<div class='section-header'>Rendimiento de modelos</div>", unsafe_allow_html=True)

col_m1, col_m2 = st.columns(2)   # Dos columnas: una para cada tipo de modelo

def style_table(df_t):
    """
    Aplica formato numérico y degradado de color a la columna F1.
    El degradado YlOrRd (amarillo→naranja→rojo) resalta visualmente
    qué modelo tiene mayor F1.
    """
    return df_t.style.format({
        "precision": "{:.3f}",   # 3 decimales para precision
        "recall":    "{:.3f}",   # 3 decimales para recall
        "f1":        "{:.3f}"    # 3 decimales para F1
    }).background_gradient(
        subset=["f1"],      # Solo aplica el degradado a la columna F1
        cmap="YlOrRd"       # Paleta: valores altos en naranja/rojo, bajos en amarillo
    )

with col_m1:
    st.markdown(f"<p style='color:{COLOR_DORADO}; font-weight:600;'>Modelos clásicos (TF-IDF)</p>", unsafe_allow_html=True)
    st.dataframe(style_table(results_base), use_container_width=True, hide_index=True)

with col_m2:
    st.markdown(f"<p style='color:{COLOR_DORADO}; font-weight:600;'>Modelos con embeddings (BETO / ST)</p>", unsafe_allow_html=True)
    st.dataframe(style_table(results_emb), use_container_width=True, hide_index=True)

# Combina ambos DataFrames añadiendo una columna "tipo" para distinguirlos en el gráfico
df_all_models = pd.concat([
    results_base.assign(tipo="TF-IDF"),       # Etiqueta los modelos clásicos
    results_emb.assign(tipo="Embeddings")     # Etiqueta los modelos con embeddings
]).reset_index(drop=True)

# Gráfico de barras horizontales ordenado por F1 descendente
chart_f1 = alt.Chart(df_all_models).mark_bar(
    cornerRadiusTopRight=4,
    cornerRadiusBottomRight=4
).encode(
    y=alt.Y("model:N",
              sort=alt.EncodingSortField(field="f1", order="descending"),  # Mejor modelo arriba
              axis=alt.Axis(labelColor=COLOR_ARENA, title=None, labelLimit=200)),
    x=alt.X("f1:Q",
              scale=alt.Scale(domain=[0.5, 0.85]),   # Eje empieza en 0.5 para ampliar diferencias
              axis=alt.Axis(labelColor=COLOR_ARENA_OSC, title="F1 (weighted)",
                             titleColor=COLOR_ARENA_OSC, grid=False, format=".2f")),
    color=alt.Color("tipo:N",
                     scale=alt.Scale(
                         domain=["TF-IDF", "Embeddings"],
                         range=[COLOR_DORADO, COLOR_POSITIVO]   # Dorado vs verde
                     ),
                     legend=alt.Legend(title=None, orient="top", labelColor=COLOR_ARENA)),
    tooltip=["model:N", "tipo:N",
              alt.Tooltip("f1:Q", format=".3f"),
              alt.Tooltip("precision:Q", format=".3f"),
              alt.Tooltip("recall:Q", format=".3f")]
).properties(
    height=280,
    background="transparent",
    title=alt.TitleParams("Comparativa F1 — todos los modelos",
                           color=COLOR_ARENA_OSC, fontSize=12)
).configure_view(strokeOpacity=0)

st.altair_chart(chart_f1, use_container_width=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ============================================================
# 14. EXPLORADOR DEL CORPUS FILTRADO
# Tabla interactiva con las primeras 50 filas del corpus
# después de aplicar los filtros del sidebar.
# ============================================================
st.markdown("<div class='section-header'>Explorador del corpus</div>", unsafe_allow_html=True)

# Solo muestra las columnas relevantes y que existan en el DataFrame
# (por si alguna columna opcional no está presente en todos los corpus)
columnas_mostrar = [c for c in [
    "created_at", "full_text", "label_sentiment",
    "label_confidence", "label_cardiffnlp",
    "retweet_count", "favorite_count", "source"
] if c in df_f.columns]

st.dataframe(
    df_f[columnas_mostrar].head(50).reset_index(drop=True),  # Primeras 50 filas, índice desde 0
    use_container_width=True,
    height=320   # Altura fija con scroll interno para no extender la página
)


# ============================================================
# 15. FOOTER
# Pie de página con autoría, DOI de Zenodo y modelos usados.
# ============================================================
st.markdown(f"""
    <div style='text-align:center; padding: 20px 0 8px 0;
                font-size:11px; color:{COLOR_ARENA_OSC}; letter-spacing:0.5px;'>
        TFM · Juan Diego Ruiz Valverde · UNIR 2026 &nbsp;·&nbsp;
        Corpus: Zenodo <a href='https://doi.org/10.5281/zenodo.20726879'
        style='color:{COLOR_DORADO};'>doi:10.5281/zenodo.20726879</a> &nbsp;·&nbsp;
        Modelos: pysentimiento · cardiffnlp · BETO · LaBSE
    </div>
""", unsafe_allow_html=True)

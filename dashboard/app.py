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
import os                           # Para construir rutas absolutas independientes del OS
import base64                       # Para convertir imágenes a base64 e incrustarlas en HTML

# ── Rutas absolutas calculadas desde la ubicación del script ──
# __file__ es la ruta absoluta del script actual (dashboard/app.py)
# os.path.dirname obtiene la carpeta que contiene el script (dashboard/)
DASHBOARD_DIR   = os.path.dirname(os.path.abspath(__file__))   # Carpeta dashboard/
REPO_DIR        = os.path.dirname(DASHBOARD_DIR)               # Raíz del repositorio

# Rutas absolutas a cada recurso
LOGO_PATH       = os.path.join(DASHBOARD_DIR, "logo_qatar2022.png")  # Logo Qatar 2022
MASCOTA_PATH    = os.path.join(DASHBOARD_DIR, "mascota.png")          # Mascota PNG
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
# ============================================================
COLOR_GRANATE_OSC  = "#4A0E23"
COLOR_GRANATE_MED  = "#8A1538"
COLOR_GRANATE_CLA  = "#B01D45"
COLOR_DORADO       = "#D4AF37"
COLOR_ARENA        = "#F2E9D8"
COLOR_ARENA_OSC    = "#D9C9A8"
COLOR_POSITIVO     = "#4CAF82"
COLOR_NEGATIVO     = "#E8574A"
COLOR_NEUTRO       = "#A8B8C8"

# ============================================================
# 3. CSS GLOBAL
# ============================================================
st.markdown(f"""
<style>
    div[data-testid="stAppViewContainer"],
    div[data-testid="stAppViewBlockContainer"] {{
        background-color: {COLOR_GRANATE_MED} !important;
    }}
    section[data-testid="stSidebar"] {{
        background-color: {COLOR_GRANATE_OSC} !important;
        border-right: 2px solid {COLOR_DORADO} !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: {COLOR_ARENA} !important;
    }}
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
    .divider {{
        border: none;
        border-top: 1px solid {COLOR_DORADO};
        margin: 24px 0;
        opacity: 0.5;
    }}
    .kpi-card {{
        background: linear-gradient(135deg, {COLOR_GRANATE_OSC}, {COLOR_GRANATE_MED});
        border: 1px solid {COLOR_DORADO};
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.35);
    }}
    .kpi-emoji  {{ font-size: 44px; line-height: 1.1; }}
    .kpi-label  {{ font-size: 13px; color: {COLOR_ARENA_OSC}; letter-spacing: 1px;
                   text-transform: uppercase; margin: 6px 0 4px; }}
    .kpi-value  {{ font-size: 32px; font-weight: 700; color: {COLOR_ARENA}; }}
    .kpi-pct    {{ font-size: 12px; color: {COLOR_DORADO}; margin-top: 2px; }}
    .section-header {{
        font-size: 15px;
        font-weight: 600;
        color: {COLOR_DORADO};
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin: 32px 0 12px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid {COLOR_DORADO}44;
    }}
    .viral-row {{
        background: {COLOR_GRANATE_OSC};
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        border-left: 4px solid {COLOR_DORADO};
    }}
    .viral-text {{ font-size: 13px; color: {COLOR_ARENA}; }}
    .viral-meta {{ font-size: 11px; color: {COLOR_ARENA_OSC}; margin-top: 6px; }}
    .stDataFrame {{ border-radius: 8px; overflow: hidden; }}
    .stMultiSelect div, .stSelectbox div {{
        background-color: {COLOR_GRANATE_OSC} !important;
        color: {COLOR_ARENA} !important;
    }}
    #MainMenu, footer {{ visibility: hidden; }}
    svg {{ width: auto !important; height: auto !important; }}
    section[data-testid="stSidebar"] span[data-baseweb="tag"] {{
        max-width: 100% !important;
        overflow: visible !important;
    }}
    section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] {{
        min-width: 0 !important;
    }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 4. CARGAR DATOS
# ============================================================
@st.cache_data
def load_data():
    df           = pd.read_json(DATA_PATH, lines=True)
    results_base = pd.read_csv(BASELINES_PATH)
    results_emb  = pd.read_csv(EMBEDDINGS_PATH)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df.dropna(subset=["created_at"])
    return df, results_base, results_emb

df, results_base, results_emb = load_data()

# ============================================================
# 5. SIDEBAR — FILTROS INTERACTIVOS
# ============================================================
with st.sidebar:
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
    sent_filter = st.multiselect(
        label="",
        options=["positivo", "negativo", "neutro"],
        default=["positivo", "negativo", "neutro"]
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Rango horario**")
    hora_min = int(df["created_at"].dt.hour.min())
    hora_max = int(df["created_at"].dt.hour.max())
    hora_range = st.slider("", hora_min, hora_max, (hora_min, hora_max))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style='font-size:11px; color:{COLOR_ARENA_OSC}; line-height:1.6;'>
            📅 11 diciembre 2022<br>
            🐦 {len(df):,} tweets en español<br>
            🤖 pysentimiento + cardiffnlp<br>
            κ = 0.6356 (acuerdo sustancial)
        </div>
    """, unsafe_allow_html=True)

# ============================================================
# 6. APLICAR FILTROS
# ============================================================
df_f = df[
    (df["label_sentiment"].isin(sent_filter)) &
    (df["created_at"].dt.hour >= hora_range[0]) &
    (df["created_at"].dt.hour <= hora_range[1])
].copy()

# ============================================================
# 7. HEADER PRINCIPAL
# ============================================================
st.markdown(f"""
    <div style='padding: 8px 0 24px 0;'>
        <h1 style='color:{COLOR_ARENA}; font-size:28px; margin:0; font-weight:700;'>
            Análisis de Sentimiento — Copa Mundial FIFA 2022
        </h1>
        <p style='color:{COLOR_ARENA_OSC}; font-size:14px; margin: 6px 0 0 0;'>
            TFM · Juan Diego Ruiz Valverde · UNIR 2026 &nbsp;·&nbsp;
            Datos: 11 dic 2022 · {len(df_f):,} tweets filtrados en español
        </p>
    </div>
""", unsafe_allow_html=True)

# ============================================================
# 8. KPIs
# ============================================================
total     = len(df)
positivos = (df["label_sentiment"] == "positivo").sum()
negativos = (df["label_sentiment"] == "negativo").sum()
neutros   = (df["label_sentiment"] == "neutro").sum()
acuerdo   = (df["label_sentiment"] == df["label_cardiffnlp"]).mean() * 100
conf_avg  = df["label_confidence"].mean() * 100

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-emoji">😊</div>
        <div class="kpi-label">Positivos</div>
        <div class="kpi-value">{positivos:,}</div>
        <div class="kpi-pct">{positivos/total*100:.1f}% del corpus</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-emoji">😡</div>
        <div class="kpi-label">Negativos</div>
        <div class="kpi-value">{negativos:,}</div>
        <div class="kpi-pct">{negativos/total*100:.1f}% del corpus</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-emoji">😐</div>
        <div class="kpi-label">Neutros</div>
        <div class="kpi-value">{neutros:,}</div>
        <div class="kpi-pct">{neutros/total*100:.1f}% del corpus</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-emoji">🤝</div>
        <div class="kpi-label">Acuerdo modelos</div>
        <div class="kpi-value">{acuerdo:.1f}%</div>
        <div class="kpi-pct">κ = 0.6356 sustancial</div>
    </div>""", unsafe_allow_html=True)

with col5:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-emoji">🎯</div>
        <div class="kpi-label">Confianza media</div>
        <div class="kpi-value">{conf_avg:.1f}%</div>
        <div class="kpi-pct">pysentimiento</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ============================================================
# 9. SERIE TEMPORAL + DONA
# ============================================================
st.markdown("<div class='section-header'>Distribución y evolución temporal</div>", unsafe_allow_html=True)

col_time, col_dona = st.columns([3, 1])

with col_time:
    df_time = df_f.groupby(
        [pd.Grouper(key="created_at", freq="30min"), "label_sentiment"]
    ).size().reset_index(name="count")

    chart_time = alt.Chart(df_time).mark_area(
        interpolate="monotone",
        opacity=0.75,
        strokeWidth=2.5
    ).encode(
        x=alt.X("created_at:T",
                 axis=alt.Axis(labelColor=COLOR_ARENA_OSC, gridColor="#FFFFFF11",
                               tickColor=COLOR_ARENA_OSC, title=None, format="%H:%M")),
        y=alt.Y("count:Q",
                 stack=None,
                 axis=alt.Axis(labelColor=COLOR_ARENA_OSC, gridColor="#FFFFFF11",
                               tickColor=COLOR_ARENA_OSC, title="Tweets / 30 min")),
        color=alt.Color("label_sentiment:N",
                         scale=alt.Scale(
                             domain=["positivo", "negativo", "neutro"],
                             range=[COLOR_POSITIVO, COLOR_NEGATIVO, COLOR_NEUTRO]
                         ),
                         legend=alt.Legend(title=None, orient="top",
                                           labelColor=COLOR_ARENA, symbolType="square")),
        tooltip=["created_at:T", "label_sentiment:N", "count:Q"]
    ).properties(
        height=260,
        background="transparent",
        title=alt.TitleParams("Actividad por sentimiento cada 30 minutos",
                               color=COLOR_ARENA_OSC, fontSize=12)
    ).configure_view(strokeOpacity=0)

    st.altair_chart(chart_time, use_container_width=True)

with col_dona:
    sizes  = [positivos, negativos, neutros]
    colors = [COLOR_POSITIVO, COLOR_NEGATIVO, COLOR_NEUTRO]
    labels = ["Positivo", "Negativo", "Neutro"]

    fig_dona, ax_dona = plt.subplots(figsize=(3.5, 3.5))
    fig_dona.patch.set_alpha(0)
    ax_dona.set_facecolor("none")

    wedges, _ = ax_dona.pie(
        sizes, colors=colors, startangle=90,
        wedgeprops=dict(width=0.55, edgecolor=COLOR_GRANATE_MED, linewidth=2)
    )
    ax_dona.text(0, 0, f"{total:,}\ntweets",
                  ha="center", va="center", fontsize=11,
                  color=COLOR_ARENA, fontweight="bold")
    legend_patches = [mpatches.Patch(color=c, label=f"{l}\n{s/total*100:.1f}%")
                      for c, l, s in zip(colors, labels, sizes)]
    ax_dona.legend(handles=legend_patches, loc="lower center",
                    bbox_to_anchor=(0.5, -0.22), ncol=3,
                    fontsize=8, frameon=False, labelcolor=COLOR_ARENA)
    plt.tight_layout()
    st.pyplot(fig_dona, transparent=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ============================================================
# 10. HEATMAP DE ACUERDO + HISTOGRAMA DE CONFIANZA
# ============================================================
st.markdown("<div class='section-header'>Validación del etiquetado automático</div>", unsafe_allow_html=True)

col_heat, col_conf = st.columns([1, 1])

with col_heat:
    st.markdown(f"<p style='font-size:12px; color:{COLOR_ARENA_OSC};'>Matriz de acuerdo entre pysentimiento y cardiffnlp</p>", unsafe_allow_html=True)

    cats = ["positivo", "negativo", "neutro"]
    matrix_data = []
    for r in cats:
        for c in cats:
            n = ((df["label_sentiment"] == r) & (df["label_cardiffnlp"] == c)).sum()
            matrix_data.append({"pysentimiento": r, "cardiffnlp": c, "count": int(n)})

    df_matrix = pd.DataFrame(matrix_data)
    total_per_row = df_matrix.groupby("pysentimiento")["count"].transform("sum")
    df_matrix["pct"] = (df_matrix["count"] / total_per_row * 100).round(1)

    heat = alt.Chart(df_matrix).mark_rect(cornerRadius=4).encode(
        x=alt.X("cardiffnlp:N", sort=cats,
                  axis=alt.Axis(labelColor=COLOR_ARENA, title="cardiffnlp",
                                titleColor=COLOR_ARENA_OSC, labelAngle=0)),
        y=alt.Y("pysentimiento:N", sort=cats,
                  axis=alt.Axis(labelColor=COLOR_ARENA, title="pysentimiento",
                                titleColor=COLOR_ARENA_OSC)),
        color=alt.Color("pct:Q", scale=alt.Scale(scheme="goldorange"), legend=None),
        tooltip=["pysentimiento:N", "cardiffnlp:N",
                  alt.Tooltip("count:Q", title="Tweets"),
                  alt.Tooltip("pct:Q", title="% fila", format=".1f")]
    )
    text_heat = alt.Chart(df_matrix).mark_text(fontSize=14, fontWeight="bold").encode(
        x=alt.X("cardiffnlp:N", sort=cats),
        y=alt.Y("pysentimiento:N", sort=cats),
        text=alt.Text("pct:Q", format=".0f"),
        color=alt.condition(alt.datum.pct > 60,
                             alt.value(COLOR_GRANATE_OSC),
                             alt.value(COLOR_ARENA))
    )
    st.altair_chart(
        (heat + text_heat).properties(height=220, background="transparent")
        .configure_view(strokeOpacity=0),
        use_container_width=True
    )

with col_conf:
    st.markdown(f"<p style='font-size:12px; color:{COLOR_ARENA_OSC};'>Distribución de confianza por modelo</p>", unsafe_allow_html=True)

    df_conf = pd.DataFrame({
        "confianza": pd.concat([df["label_confidence"], df["label_conf_cardiffnlp"]]),
        "modelo": ["pysentimiento"] * len(df) + ["cardiffnlp"] * len(df)
    })
    hist_conf = alt.Chart(df_conf).mark_bar(opacity=0.75, binSpacing=1).encode(
        x=alt.X("confianza:Q", bin=alt.Bin(maxbins=20),
                  axis=alt.Axis(labelColor=COLOR_ARENA_OSC, title="Confianza",
                                titleColor=COLOR_ARENA_OSC, format=".1f")),
        y=alt.Y("count():Q",
                  axis=alt.Axis(labelColor=COLOR_ARENA_OSC, title="Tweets",
                                titleColor=COLOR_ARENA_OSC)),
        color=alt.Color("modelo:N",
                         scale=alt.Scale(domain=["pysentimiento", "cardiffnlp"],
                                         range=[COLOR_DORADO, COLOR_GRANATE_CLA]),
                         legend=alt.Legend(title=None, orient="top", labelColor=COLOR_ARENA)),
        tooltip=["modelo:N", alt.Tooltip("count():Q", title="Tweets")]
    ).properties(height=220, background="transparent").configure_view(strokeOpacity=0)

    st.altair_chart(hist_conf, use_container_width=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ============================================================
# 11. NUBE DE PALABRAS + DISPOSITIVOS
# ============================================================
st.markdown("<div class='section-header'>Vocabulario y plataformas</div>", unsafe_allow_html=True)

col_wc, col_src = st.columns([3, 2])

with col_wc:
    sent_sel = st.selectbox(
        "Sentimiento para la nube de palabras:",
        ["positivo", "negativo", "neutro"],
        index=1
    )
    col_texto = next((c for c in ["clean_text", "full_text"] if c in df.columns), None)

    if col_texto:
        textos = " ".join(df[df["label_sentiment"] == sent_sel][col_texto].astype(str).tolist())
        colores = {
            "positivo": lambda *a, **k: COLOR_POSITIVO,
            "negativo": lambda *a, **k: COLOR_NEGATIVO,
            "neutro":   lambda *a, **k: COLOR_NEUTRO
        }
        wc = WordCloud(
            width=700, height=320,
            background_color=COLOR_GRANATE_MED,
            color_func=colores[sent_sel],
            max_words=80,
            stopwords={"rt", "https", "co", "que", "de", "el", "la", "en",
                        "los", "las", "un", "una", "es", "y", "a", "con",
                        "se", "del", "al", "por", "no", "le", "lo", "su"},
            prefer_horizontal=0.85,
            collocations=False
        ).generate(textos)

        fig_wc, ax_wc = plt.subplots(figsize=(8, 3.5))
        fig_wc.patch.set_facecolor(COLOR_GRANATE_MED)
        ax_wc.set_facecolor(COLOR_GRANATE_MED)
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(fig_wc, transparent=False)

with col_src:
    top_src = df_f["source"].value_counts().head(7).reset_index()
    top_src.columns = ["dispositivo", "tweets"]
    chart_src = alt.Chart(top_src).mark_bar(
        cornerRadiusTopRight=4,
        cornerRadiusBottomRight=4,
        color=COLOR_DORADO
    ).encode(
        y=alt.Y("dispositivo:N", sort="-x",
                  axis=alt.Axis(labelColor=COLOR_ARENA, title=None, labelLimit=160)),
        x=alt.X("tweets:Q",
                  axis=alt.Axis(labelColor=COLOR_ARENA_OSC, title="Tweets",
                                titleColor=COLOR_ARENA_OSC, grid=False)),
        tooltip=["dispositivo:N", "tweets:Q"]
    ).properties(
        height=260,
        background="transparent",
        title=alt.TitleParams("Tweets por dispositivo", color=COLOR_ARENA_OSC, fontSize=12)
    ).configure_view(strokeOpacity=0)

    st.altair_chart(chart_src, use_container_width=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ============================================================
# 12. TWEETS MÁS VIRALES
# ============================================================
st.markdown("<div class='section-header'>Tweets más retuiteados</div>", unsafe_allow_html=True)

sent_viral = st.radio(
    "Sentimiento:",
    ["positivo", "negativo", "neutro"],
    horizontal=True,
    index=1
)
top_viral = (
    df[df["label_sentiment"] == sent_viral]
    .nlargest(5, "retweet_count")
    [["full_text", "retweet_count", "favorite_count", "label_confidence"]]
)
for _, row in top_viral.iterrows():
    texto = row["full_text"][:220] + "..." if len(row["full_text"]) > 220 else row["full_text"]
    conf_pct = f"{row['label_confidence']*100:.0f}%"
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
# 13. COMPARATIVA DE MODELOS
# ============================================================
st.markdown("<div class='section-header'>Rendimiento de modelos</div>", unsafe_allow_html=True)

col_m1, col_m2 = st.columns(2)

def style_table(df_t):
    return df_t.style.format({
        "precision": "{:.3f}",
        "recall":    "{:.3f}",
        "f1":        "{:.3f}"
    }).background_gradient(subset=["f1"], cmap="YlOrRd")

with col_m1:
    st.markdown(f"<p style='color:{COLOR_DORADO}; font-weight:600;'>Modelos clásicos (TF-IDF)</p>", unsafe_allow_html=True)
    st.dataframe(style_table(results_base), use_container_width=True, hide_index=True)

with col_m2:
    st.markdown(f"<p style='color:{COLOR_DORADO}; font-weight:600;'>Modelos con embeddings (BETO / ST)</p>", unsafe_allow_html=True)
    st.dataframe(style_table(results_emb), use_container_width=True, hide_index=True)

# Gráfico F1 comparativo — sin espacio extra antes del explorador
df_all_models = pd.concat([
    results_base.assign(tipo="TF-IDF"),
    results_emb.assign(tipo="Embeddings")
]).reset_index(drop=True)

chart_f1 = alt.Chart(df_all_models).mark_bar(
    cornerRadiusTopRight=4,
    cornerRadiusBottomRight=4
).encode(
    y=alt.Y("model:N",
              sort=alt.EncodingSortField(field="f1", order="descending"),
              axis=alt.Axis(labelColor=COLOR_ARENA, title=None, labelLimit=200)),
    x=alt.X("f1:Q",
              scale=alt.Scale(domain=[0.5, 0.85]),
              axis=alt.Axis(labelColor=COLOR_ARENA_OSC, title="F1 (weighted)",
                             titleColor=COLOR_ARENA_OSC, grid=False, format=".2f")),
    color=alt.Color("tipo:N",
                     scale=alt.Scale(domain=["TF-IDF", "Embeddings"],
                                     range=[COLOR_DORADO, COLOR_POSITIVO]),
                     legend=alt.Legend(title=None, orient="top", labelColor=COLOR_ARENA)),
    tooltip=["model:N", "tipo:N",
              alt.Tooltip("f1:Q", format=".3f"),
              alt.Tooltip("precision:Q", format=".3f"),
              alt.Tooltip("recall:Q", format=".3f")]
).properties(
    height=220,                  # ← reducido de 280 a 220 para menos espacio
    background="transparent",
    title=alt.TitleParams("Comparativa F1 — todos los modelos",
                           color=COLOR_ARENA_OSC, fontSize=12)
).configure_view(strokeOpacity=0)

st.altair_chart(chart_f1, use_container_width=True)

# Sin divider aquí para reducir espacio entre Rendimiento y Explorador

# ============================================================
# 14. EXPLORADOR DEL CORPUS
# Corpus: corpus_labeled_v1.jsonl (6,437 tweets en español)
# Columnas incluidas: created_at, full_text, clean_text,
#   label_sentiment, label_confidence, label_cardiffnlp,
#   label_conf_cardiffnlp, retweet_count, favorite_count,
#   followers_count, friends_count, source, account_created_at
# Columnas excluidas por no aportar valor analítico:
#   tweet_id_hashed (hash interno), tokens (lista técnica),
#   truncated, favorited, retweeted, lang (siempre "es"),
#   in_reply_to_status_id, in_reply_to_user_id,
#   in_reply_to_screen_name, is_quote_status
# ============================================================
st.markdown("<div class='section-header'>Explorador del corpus</div>", unsafe_allow_html=True)

# Selector de cuántas filas mostrar
n_filas = st.select_slider(
    "Número de filas a mostrar:",
    options=[50, 100, 200, 500, len(df_f)],
    value=100    # Por defecto 100 filas
)

# Columnas con valor analítico para mostrar en el explorador
columnas_mostrar = [c for c in [
    "created_at",             # Fecha y hora del tweet
    "full_text",              # Texto original del tweet
    "clean_text",             # Texto limpio y normalizado
    "label_sentiment",        # Etiqueta pysentimiento (modelo principal)
    "label_confidence",       # Confianza de pysentimiento [0-1]
    "label_cardiffnlp",       # Etiqueta cardiffnlp (segundo anotador)
    "label_conf_cardiffnlp",  # Confianza de cardiffnlp [0-1]
    "retweet_count",          # Número de retweets
    "favorite_count",         # Número de likes
    "followers_count",        # Seguidores del autor
    "friends_count",          # Seguidos del autor
    "source",                 # Dispositivo usado (Android, iPhone, Web)
    "account_created_at",     # Fecha de creación de la cuenta del autor
] if c in df_f.columns]

st.dataframe(
    df_f[columnas_mostrar].head(n_filas).reset_index(drop=True),
    use_container_width=True,
    height=530    # Altura aumentada para mostrar ~15 filas de primera vista
)

# ============================================================
# 15. MASCOTA + FOOTER
# La mascota se muestra centrada entre el explorador y el footer.
# Usamos PNG o JPG — formato recomendado sobre GIF para Streamlit.
# ============================================================

# Función para convertir imagen a base64 e incrustar en HTML
def img_a_base64(ruta: str, mime: str = "image/png") -> str:
    """Convierte una imagen a base64 para incrustarla directamente en HTML."""
    with open(ruta, "rb") as f:
        datos = f.read()
    b64 = base64.b64encode(datos).decode("utf-8")
    return f"data:{mime};base64,{b64}"

# Intenta cargar la mascota — si no existe, solo muestra el footer
mascota_html = ""
if os.path.exists(MASCOTA_PATH):
    # Detecta el formato según la extensión del archivo
    ext = os.path.splitext(MASCOTA_PATH)[1].lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    mime = mime_map.get(ext, "image/png")
    src = img_a_base64(MASCOTA_PATH, mime)
    mascota_html = f"""
        <div style='text-align:center; margin: 16px 0 8px 0;'>
            <img src="{src}" style="width:100px; opacity:0.9;" />
        </div>
    """

# Footer con mascota centrada arriba del texto
st.markdown(f"""
    {mascota_html}
    <div style='text-align:center; padding: 4px 0 12px 0;
                font-size:11px; color:{COLOR_ARENA_OSC}; letter-spacing:0.5px;'>
        TFM · Juan Diego Ruiz Valverde · UNIR 2026 &nbsp;·&nbsp;
        Corpus: Zenodo <a href='https://doi.org/10.5281/zenodo.20726879'
        style='color:{COLOR_DORADO};'>doi:10.5281/zenodo.20726879</a> &nbsp;·&nbsp;
        Modelos: pysentimiento · cardiffnlp · BETO · LaBSE
    </div>
""", unsafe_allow_html=True)

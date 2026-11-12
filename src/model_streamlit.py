"""model_deploy.py: aplicación Streamlit para predecir con el modelo y monitorear data drift.

Usa tal cual model.py (modelo) y model_monitoring.py (simulación y métricas).
Ejecutar desde la carpeta del proyecto:  streamlit run model_deploy.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import linregress

st.set_page_config(page_title="Monitoreo del modelo", page_icon="📊", layout="wide")

EMOJI = {"Estable": "🟢", "Alerta": "🟡", "Crítico": "🔴"}
UMBRAL_PSI_ALERTA = 0.10  # PSI entre 0.10 y UMBRAL_PSI -> amarillo

# Bandas de riesgo de incumplimiento (1 - proba de pagar a tiempo). Valores ilustrativos.
BANDAS_RIESGO = [
    (0.20, "Bajo", "🟢"),
    (0.40, "Moderado", "🟡"),
    (0.60, "Medio-alto", "🟠"),
    (0.80, "Alto", "🔴"),
    (1.01, "Muy alto", "🟣"),
]


# =============================================================================
# RECURSOS PESADOS (se cargan UNA vez por proceso con cache_resource)
# =============================================================================
@st.cache_resource(show_spinner="Cargando modelo y datos...")
def cargar_recursos():
    """Importa el modelo ya entrenado y arma base cruda + referencia (features + proba)."""
    import model_deploy as md
    import model_monitoring as mm

    base = mm.cargar_base()               # datos crudos para muestrear y para el formulario
    ref = mm.pronostico(md.X_train)       # referencia: features de train + predict_proba
    return mm, md, base, ref


@st.cache_resource(show_spinner="Simulando drift mes a mes...")
def simular_meses(n_meses, acumulativo, n_muestra, frac_filas):
    """Corre TODOS los meses (sin cortar en el primer drift) y guarda cada muestra actual.

    acumulativo=True: en el mes k las variaciones de model_monitoring se multiplican por k,
    así el drift crece con el tiempo. Con False, cada mes repite la misma variación.
    """
    mm, md, base, ref = cargar_recursos()
    orig = (mm.VARIACION_NUM, mm.VARIACION_PROP, mm.FRAC_FILAS)
    tablas, actuales = [], {}
    try:
        for mes in range(1, n_meses + 1):
            k = mes if acumulativo else 1
            mm.VARIACION_NUM = {c: {"media": v["media"] * k, "std": v["std"] * k}
                                for c, v in orig[0].items()}
            mm.VARIACION_PROP = {c: (cat, pct * k) for c, (cat, pct) in orig[1].items()}
            mm.FRAC_FILAS = frac_filas

            rng = np.random.default_rng(mm.SEMILLA + mes)
            muestra = base.sample(n_muestra, random_state=mm.SEMILLA + mes)
            muestra = mm.simular_drift(muestra, rng)

            act = mm.pronostico(md.preprocesador.transform(muestra))
            actuales[mes] = act
            tablas.append(mm.metricas_drift(ref, act).assign(mes=mes))
    finally:  # siempre restaura la configuración original del módulo
        mm.VARIACION_NUM, mm.VARIACION_PROP, mm.FRAC_FILAS = orig
    return pd.concat(tablas, ignore_index=True), actuales


mm, md, base, ref = cargar_recursos()


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================
def nivel(f):
    """Semáforo por variable usando los umbrales de model_monitoring."""
    if f["tipo"] == "numerica":
        if f["PSI"] > mm.UMBRAL_PSI or f["JS"] > mm.UMBRAL_JS:
            return "Crítico"
        if f["PSI"] > UMBRAL_PSI_ALERTA or f["KS_p"] < mm.UMBRAL_P:
            return "Alerta"
        return "Estable"
    if f["Chi2_p"] < 0.001:
        return "Crítico"
    return "Alerta" if f["Chi2_p"] < mm.UMBRAL_P else "Estable"


def riesgo_barra(f):
    """Puntaje 0-1 para la barra de riesgo (1 = en o sobre el umbral crítico)."""
    if f["tipo"] == "numerica":
        r = max(f["PSI"] / mm.UMBRAL_PSI, f["JS"] / mm.UMBRAL_JS)
    else:
        r = -np.log10(max(f["Chi2_p"], 1e-12)) / 3
    return float(np.clip(r, 0, 1))


def enriquecer(tabla):
    t = tabla.copy()
    t["nivel"] = t.apply(nivel, axis=1)
    t["riesgo"] = t.apply(riesgo_barra, axis=1)
    t["semaforo"] = t["nivel"].map(EMOJI)
    return t


def cambios_abruptos(serie, z=2.0):
    """Meses donde el salto de la serie es atípico (z-score de la diferencia)."""
    d = serie.diff().dropna()
    if len(d) < 3 or d.std() == 0:
        return []
    zs = (d - d.mean()) / d.std()
    return zs[zs > z].index.tolist()


def tendencia(serie):
    """(pendiente, p_value) de una regresión lineal sobre los meses."""
    if len(serie) < 3:
        return 0.0, 1.0
    r = linregress(serie.index.values.astype(float), serie.values)
    return float(r.slope), (1.0 if np.isnan(r.pvalue) else float(r.pvalue))


def comparar_distribuciones(ref_s, act_s, mes):
    fig = go.Figure()
    fig.add_histogram(x=ref_s, name="Histórico (train)", histnorm="probability density",
                      opacity=0.55)
    fig.add_histogram(x=act_s, name=f"Actual (mes {mes})", histnorm="probability density",
                      opacity=0.55)
    fig.update_layout(barmode="overlay", height=380, margin=dict(t=30, b=10),
                      legend=dict(orientation="h", y=1.1), yaxis_title="densidad")
    return fig


def recomendar(tabla_mes, serie_psi, mes):
    """Lista de (nivel_streamlit, mensaje) según el estado del mes analizado."""
    msgs = []
    crit = tabla_mes.loc[tabla_mes["nivel"] == "Crítico", "variable"].tolist()
    alert = tabla_mes.loc[tabla_mes["nivel"] == "Alerta", "variable"].tolist()
    prop = len(crit) / max(len(tabla_mes), 1)

    if prop >= 0.3:
        msgs.append(("error", f"Drift crítico en {len(crit)} de {len(tabla_mes)} variables "
                              f"({prop:.0%}). Se sugiere REENTRENAR el modelo."))
    elif crit:
        msgs.append(("error", f"Drift crítico en: {', '.join(crit)}. Revisar estas variables "
                              "(cambios en la fuente, calidad de datos) y evaluar reentrenar."))
    if "proba" in crit:
        msgs.append(("error", "Cambió la distribución de las PREDICCIONES (proba): el modelo "
                              "está puntuando distinto a como fue entrenado."))
    if alert:
        msgs.append(("warning", f"Variables en alerta: {', '.join(alert)}. Monitorear de cerca."))
    if not crit and not alert:
        msgs.append(("success", "Todas las variables estables. No se requiere acción."))

    abruptos = [m for m in cambios_abruptos(serie_psi.loc[:mes]) if m <= mes]
    if abruptos:
        msgs.append(("warning", f"Cambio abrupto del drift en el/los mes(es): "
                                f"{', '.join(map(str, abruptos))}. Investigar qué cambió."))
    pend, p = tendencia(serie_psi.loc[:mes])
    if p < 0.05 and pend > 0:
        msgs.append(("warning", "El PSI medio muestra tendencia creciente: conviene planificar "
                                "un reentrenamiento preventivo."))
    return msgs


# =============================================================================
# SIDEBAR: parámetros de simulación (form = se aplican todos juntos al enviar)
# =============================================================================
with st.sidebar:
    st.header("Simulación de drift")
    with st.form("form_simulacion"):
        n_meses = st.slider("Meses a simular", 3, 36, int(mm.MAX_MESES))
        acumulativo = st.checkbox("Drift acumulativo (se acentúa mes a mes)", value=True)
        n_muestra = st.number_input("Filas por mes", min_value=200, max_value=len(base),
                                    value=min(int(mm.N_MUESTRA), len(base)), step=100)
        frac_filas = st.slider("Fracción de filas modificadas", 0.05, 1.0,
                               float(mm.FRAC_FILAS), 0.05)
        st.form_submit_button("Simular")

resultados, actuales = simular_meses(int(n_meses), bool(acumulativo), int(n_muestra),
                                     float(frac_filas))
resultados = enriquecer(resultados)

# Serie temporal global (una fila por mes)
num = resultados[resultados["tipo"] == "numerica"]
serie = num.groupby("mes")["PSI"].agg(psi_medio="mean", psi_max="max")
serie["criticas"] = resultados.groupby("mes")["nivel"].apply(lambda s: int((s == "Crítico").sum()))
serie["con_drift"] = resultados.groupby("mes")["drift"].sum()
serie["proba_media"] = [actuales[m]["proba"].mean() for m in serie.index]

# =============================================================================
# CUERPO
# =============================================================================
st.title("📊 Monitoreo del modelo y data drift")
mes = st.slider("Mes a analizar (meses desde el training)", 1, int(n_meses), int(n_meses))

tab_pred, tab_drift, tab_temp, tab_rec = st.tabs(
    ["🔮 Predicción", "📊 Drift por variable", "📈 Evolución temporal", "🚨 Recomendaciones"]
)

# ------------------------------------------------------------------ Predicción
with tab_pred:
    st.subheader("Predicción individual")
    st.caption("Completá los datos del cliente. El modelo devuelve la probabilidad de pagar a "
               "tiempo (predict_proba) y se traduce a una banda de riesgo.")
    with st.form("form_prediccion"):
        cols = st.columns(3)
        valores = {}
        for i, col in enumerate(base.columns):
            s = base[col]
            with cols[i % 3]:
                if pd.api.types.is_numeric_dtype(s) and s.dtype != object:
                    if pd.api.types.is_integer_dtype(s):
                        valores[col] = st.number_input(col, value=int(s.median()), step=1)
                    else:
                        valores[col] = st.number_input(col, value=float(s.median()))
                else:
                    opciones = sorted(s.dropna().unique(), key=str)
                    valores[col] = st.selectbox(col, opciones)
        enviar = st.form_submit_button("Calcular riesgo")

    if enviar:
        entrada = pd.DataFrame([valores]).astype(base.dtypes.to_dict())
        proba = float(md.pipe_final.predict_proba(entrada)[0, 1])   # P(pagar a tiempo)
        riesgo = 1 - proba
        _, banda, icono = next(b for b in BANDAS_RIESGO if riesgo < b[0])

        c1, c2, c3 = st.columns(3)
        c1.metric("Prob. de pagar a tiempo", f"{proba:.1%}")
        c2.metric("Riesgo de incumplimiento", f"{riesgo:.1%}")
        c3.metric("Banda de riesgo", f"{icono} {banda}")
        st.progress(min(max(riesgo, 0.0), 1.0), text="Nivel de riesgo")

    # ---- Predicción por lotes ------------------------------------------------
    st.divider()
    st.subheader("Predicción por lotes (CSV)")
    st.caption("Subí un CSV con las mismas columnas originales que el formulario. "
               "Las columnas de más se ignoran; si falta alguna, se avisa.")
    st.download_button("Descargar plantilla CSV",
                       pd.DataFrame(columns=base.columns).to_csv(index=False).encode("utf-8"),
                       file_name="plantilla_clientes.csv", mime="text/csv")

    archivo = st.file_uploader("CSV de clientes", type="csv")
    if archivo is not None:
        try:
            lote = pd.read_csv(archivo, sep=None, engine="python")  # detecta , o ;
            faltan = [c for c in base.columns if c not in lote.columns]
            if faltan:
                st.error(f"Faltan columnas en el CSV: {', '.join(faltan)}")
            else:
                entrada = mm.preparar_tipos(lote.copy())[list(base.columns)]
                proba = md.pipe_final.predict_proba(entrada)[:, 1]   # P(pagar a tiempo)
                riesgo = 1 - proba
                idx = np.digitize(riesgo, [b[0] for b in BANDAS_RIESGO[:-1]])
                etiquetas = [f"{ic} {bn}" for _, bn, ic in BANDAS_RIESGO]
                salida = lote.assign(
                    proba_pago_a_tiempo=proba.round(4),
                    riesgo_incumplimiento=riesgo.round(4),
                    banda_riesgo=[etiquetas[i] for i in idx],
                )
                c1, c2 = st.columns(2)
                c1.metric("Clientes evaluados", len(salida))
                c2.metric("Riesgo promedio", f"{riesgo.mean():.1%}")
                st.bar_chart(salida["banda_riesgo"].value_counts().reindex(etiquetas, fill_value=0))
                st.dataframe(salida, hide_index=True)
                st.download_button("Descargar resultados (CSV)",
                                   salida.to_csv(index=False).encode("utf-8"),
                                   file_name="predicciones.csv", mime="text/csv")
        except Exception as e:  # CSV mal formado, tipos inesperados, nulos, etc.
            st.error(f"No se pudo procesar el archivo: {e}")

# ----------------------------------------------------------- Drift por variable
with tab_drift:
    tabla_mes = resultados[resultados["mes"] == mes].sort_values("riesgo", ascending=False)
    n_c = int((tabla_mes["nivel"] == "Crítico").sum())
    n_a = int((tabla_mes["nivel"] == "Alerta").sum())
    n_e = int((tabla_mes["nivel"] == "Estable").sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 Críticas", n_c)
    c2.metric("🟡 En alerta", n_a)
    c3.metric("🟢 Estables", n_e)

    st.subheader(f"Métricas de drift por variable (mes {mes})")
    mostrar = tabla_mes[["variable", "semaforo", "nivel", "riesgo", "PSI", "JS", "KS_p",
                         "Chi2_p", "tipo"]].round(4)
    st.dataframe(
        mostrar, hide_index=True,
        column_config={"riesgo": st.column_config.ProgressColumn(
            "Riesgo", min_value=0.0, max_value=1.0, format="%.2f")},
    )
    st.caption("Numéricas: Crítico si PSI > 0.25 o JS > 0.10; Alerta si PSI > 0.10 o KS p < 0.05. "
               "Categóricas: Crítico si chi² p < 0.001; Alerta si p < 0.05.")

    st.subheader("Histórico vs actual")
    var = st.selectbox("Variable", tabla_mes["variable"].tolist())
    st.plotly_chart(comparar_distribuciones(ref[var], actuales[mes][var], mes))

# ---------------------------------------------------------- Evolución temporal
with tab_temp:
    st.subheader("Evolución del drift")
    fig = go.Figure()
    fig.add_scatter(x=serie.index, y=serie["psi_medio"], name="PSI medio", mode="lines+markers")
    fig.add_scatter(x=serie.index, y=serie["psi_max"], name="PSI máximo", mode="lines+markers")
    ab = cambios_abruptos(serie["psi_medio"])
    if ab:
        fig.add_scatter(x=ab, y=serie.loc[ab, "psi_medio"], name="Cambio abrupto",
                        mode="markers", marker=dict(size=14, color="red", symbol="x"))
    fig.add_hline(y=mm.UMBRAL_PSI, line_dash="dash", line_color="red",
                  annotation_text="umbral crítico")
    fig.add_hline(y=UMBRAL_PSI_ALERTA, line_dash="dot", line_color="orange",
                  annotation_text="alerta")
    fig.update_layout(height=380, xaxis_title="meses desde el training", yaxis_title="PSI",
                      margin=dict(t=30, b=10), legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig)

    pend, p = tendencia(serie["psi_medio"])
    if p < 0.05 and pend > 0:
        st.warning(f"Tendencia creciente del PSI medio (+{pend:.3f} por mes, p={p:.3f}).")
    elif p < 0.05 and pend < 0:
        st.info(f"Tendencia decreciente del PSI medio ({pend:.3f} por mes, p={p:.3f}).")
    else:
        st.info("Sin tendencia significativa en el PSI medio.")

    primer = serie.index[serie["con_drift"] > 0]
    if len(primer):
        st.error(f"Primer mes con drift según model_monitoring: mes {primer[0]}.")
    else:
        st.success("Ningún mes supera los umbrales de model_monitoring.")

    st.subheader("PSI por variable numérica")
    top = (num.groupby("variable")["PSI"].max().sort_values(ascending=False).index[:5].tolist())
    elegidas = st.multiselect("Variables", sorted(num["variable"].unique()), default=top)
    fig2 = go.Figure()
    for v in elegidas:
        d = num[num["variable"] == v]
        fig2.add_scatter(x=d["mes"], y=d["PSI"], name=v, mode="lines")
    fig2.add_hline(y=mm.UMBRAL_PSI, line_dash="dash", line_color="red")
    fig2.update_layout(height=350, xaxis_title="meses desde el training", yaxis_title="PSI",
                       margin=dict(t=30, b=10))
    st.plotly_chart(fig2)

    st.subheader("Predicciones: probabilidad media de pagar a tiempo")
    fig3 = go.Figure()
    fig3.add_scatter(x=serie.index, y=serie["proba_media"], name="Actual", mode="lines+markers")
    fig3.add_hline(y=float(ref["proba"].mean()), line_dash="dash",
                   annotation_text="referencia (train)")
    fig3.update_layout(height=300, xaxis_title="meses desde el training",
                       yaxis_title="proba media", margin=dict(t=30, b=10))
    st.plotly_chart(fig3)

# --------------------------------------------------------------- Recomendaciones
with tab_rec:
    st.subheader(f"Diagnóstico automático (mes {mes})")
    tabla_mes = resultados[resultados["mes"] == mes]
    for tipo_msg, texto in recomendar(tabla_mes, serie["psi_medio"], mes):
        getattr(st, tipo_msg)(texto)

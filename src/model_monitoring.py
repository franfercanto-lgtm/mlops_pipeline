"""
model_monitoring.py
-------------------
Simula data drift mes a mes (cada "commit" = 1 mes desde el training) y lo mide
con KS, PSI, Jensen-Shannon y chi-cuadrado. Se detiene cuando UNA métrica de
UNA variable sale de su umbral (o al llegar a MAX_MESES).
Si se indica RUTA_DATASET_NUEVO, se omite la simulación y se mide el drift de ese dataset.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp

from ft_engineering import COLS_CATEGORICAS, COLS_EXCLUIR, RUTA_CSV, SEMILLA, TARGET
from model_deploy import X_train, pipe_modelo, preprocesador

# =============================================================================
# CONFIGURACIÓN (editá estos valores a mano)
# =============================================================================
RUTA_DATASET_NUEVO = None   # ej. "dataset_nuevo.csv" -> datos reales; None -> simulación
N_MUESTRA = 1000        # filas muestreadas de toda la base en cada mes
FRAC_FILAS = 0.30       # fracción de filas que se modifican (media y std)
MAX_MESES = 24          # tope de meses simulados

# Variación en % de la media y la desviación (solo en las filas seleccionadas)
VARIACION_NUM = {
    "edad_cliente":        {"media": 5,  "std": 0},
    "capital_prestado":    {"media": 5,  "std": 10},
    "puntaje_datacredito": {"media": -3, "std": 0},
}
# Variación en % de la proporción de una categoría: {columna: (categoría, %)}
VARIACION_PROP = {"tipo_laboral": ("Independiente", 5)}

# Umbrales de drift
UMBRAL_PSI, UMBRAL_JS, UMBRAL_P = 0.25, 0.1, 0.05
MAX_CATEGORIAS = 10     # columnas con <= este nº de valores se tratan como categóricas


# =============================================================================
# 1. SIMULACIÓN DE DRIFT (sobre datos crudos)
# =============================================================================
def preparar_tipos(df):
    """Mismo tratamiento de tipos que ft_engineering (booleanas a 0/1, categóricas a texto)."""
    df = df.drop(columns=[c for c in COLS_EXCLUIR + [TARGET] if c in df.columns])
    df[df.select_dtypes("bool").columns] = df.select_dtypes("bool").astype(int)
    df[COLS_CATEGORICAS] = df[COLS_CATEGORICAS].astype("object")
    return df


def cargar_base():
    """Lee la base cruda (para muestrear en la simulación)."""
    return pd.read_csv(RUTA_CSV).pipe(lambda d: d.drop(columns=COLS_EXCLUIR)).pipe(preparar_tipos)


def simular_drift(muestra, rng):
    """Cambia medias, desviaciones (en algunas filas) y proporciones de la muestra."""
    muestra = muestra.copy()

    # Medias y desviaciones: solo en una fracción de las filas
    filas = rng.choice(muestra.index, size=int(len(muestra) * FRAC_FILAS), replace=False)
    for col, v in VARIACION_NUM.items():
        muestra[col] = muestra[col].astype(float)
        x = muestra.loc[filas, col]
        x = x + x.mean() * v["media"] / 100                      # corre la media
        x = x.mean() + (x - x.mean()) * (1 + v["std"] / 100)     # escala la dispersión
        muestra.loc[filas, col] = x

    # Proporciones: sube/baja la proporción de una categoría cambiando filas al azar
    for col, (cat, pct) in VARIACION_PROP.items():
        n_cambiar = int(round(len(muestra) * (muestra[col] == cat).mean() * abs(pct) / 100))
        if pct >= 0:
            origen, destino = muestra.index[muestra[col] != cat], cat
        else:
            origen = muestra.index[muestra[col] == cat]
            destino = muestra.loc[muestra[col] != cat, col].iloc[0]
        elegidas = rng.choice(origen, size=min(n_cambiar, len(origen)), replace=False)
        muestra.loc[elegidas, col] = destino
    return muestra


# =============================================================================
# 2. MÉTRICAS DE DATA DRIFT
# =============================================================================
def psi_js(ref, act, bins=10):
    """PSI y Jensen-Shannon (distancia, base 2) usando cuantiles de la referencia."""
    cortes = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(cortes) < 2:
        return 0.0, 0.0
    cortes[0], cortes[-1] = -np.inf, np.inf
    p = np.clip(np.histogram(ref, cortes)[0] / len(ref), 1e-6, None)
    q = np.clip(np.histogram(act, cortes)[0] / len(act), 1e-6, None)
    return float(np.sum((q - p) * np.log(q / p))), float(jensenshannon(p, q, base=2))


def chi2_p(ref, act):
    """p-valor del chi-cuadrado entre las frecuencias de referencia y actuales."""
    vals = np.union1d(ref.unique(), act.unique())
    if len(vals) < 2:
        return 1.0
    tabla = [[(ref == v).sum() for v in vals], [(act == v).sum() for v in vals]]
    return float(chi2_contingency(tabla)[1])


def metricas_drift(ref, act):
    """Tabla con una fila por variable (features transformadas + probabilidad)."""
    filas = []
    for col in ref.columns:
        if ref[col].nunique() <= MAX_CATEGORIAS:                  # categórica
            fila = {"variable": col, "tipo": "categorica", "KS_p": np.nan, "PSI": np.nan,
                    "JS": np.nan, "Chi2_p": chi2_p(ref[col], act[col])}
            drift = fila["Chi2_p"] < UMBRAL_P
        else:                                                      # numérica
            psi, js = psi_js(ref[col], act[col])
            fila = {"variable": col, "tipo": "numerica", "KS_p": ks_2samp(ref[col], act[col])[1],
                    "PSI": psi, "JS": js, "Chi2_p": np.nan}
            drift = fila["KS_p"] < UMBRAL_P or psi > UMBRAL_PSI or js > UMBRAL_JS
        filas.append({**fila, "drift": bool(drift)})
    return pd.DataFrame(filas)


# =============================================================================
# 3. MONITOREO
# =============================================================================
def pronostico(X):
    """Agrega el pronóstico (probabilidad de pagar a tiempo) como una columna más."""
    return X.assign(proba=pipe_modelo.predict_proba(X)[:, 1])


if __name__ == "__main__":
    # Referencia: features de train + pronóstico
    ref = pronostico(X_train)

    if RUTA_DATASET_NUEVO:                                        # datos reales: una sola comparación
        nuevo = preparar_tipos(pd.read_csv(RUTA_DATASET_NUEVO))
        tabla = metricas_drift(ref, pronostico(preprocesador.transform(nuevo)))
        print(tabla.round(4).to_string(index=False))
        print(f"\nVariables con drift: {int(tabla['drift'].sum())}")
    else:                                                         # simulación: 1 commit = 1 mes
        base = cargar_base()
        resultados = []
        for mes in range(1, MAX_MESES + 1):
            rng = np.random.default_rng(SEMILLA + mes)
            muestra = base.sample(N_MUESTRA, random_state=SEMILLA + mes)
            muestra = simular_drift(muestra, rng)

            tabla = metricas_drift(ref, pronostico(preprocesador.transform(muestra)))
            tabla = tabla.assign(meses_desde_training=mes)
            resultados.append(tabla)
            if tabla["drift"].any():                              # basta con 1 métrica fuera de umbral
                print(f"Data drift detectado en el mes {mes}\n")
                break

        resultados = pd.concat(resultados, ignore_index=True)
        print(tabla.round(4).to_string(index=False))             # detalle del último mes
        print("\nVariables con drift por mes:")
        print(resultados.groupby("meses_desde_training")["drift"].sum().to_string())

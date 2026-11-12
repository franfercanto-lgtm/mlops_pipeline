"""
model_deploy.py: modelo final (Regresión Logística) con el feature engineering integrado.
-----------------------------------------------------------------------------------------
Archivo autosuficiente: ya no depende de ft_engineering.py.

Incluye, todo dentro de un único Pipeline:
  - Ratios automáticos   (cuota_sobre_salario, huella_por_credito: se calculan al ingresar datos)
  - Imputación           (feature-engine: MeanMedianImputer, CategoricalImputer)
  - Codificación         (feature-engine: RareLabelEncoder, OneHotEncoder
                          sklearn: OrdinalEncoder para la variable ordinal)
  - Escalado             (sklearn: StandardScaler)
  - Modelo               (sklearn: LogisticRegression balanceada)

Requiere: pip install pandas scikit-learn feature-engine
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder

# =============================================================================
# 0. CONFIGURACIÓN
# =============================================================================
# El CSV se busca junto a este archivo (no depende de desde dónde ejecutes)
RUTA_CSV = Path(__file__).resolve().parent / "Base_de_Datos_Final.csv"
TARGET = "Pago_atiempo"
SEMILLA = 42
TEST_SIZE = 0.20

# Columnas que no entran como features:
#   - fecha_prestamo: texto de fecha; su información temporal ya se extrajo en el EDA.
#   - dias_desde_inicio: se excluye porque mide el paso del tiempo y no el riesgo del cliente.
#     Un cliente nuevo siempre tendría una fecha posterior a la de entrenamiento, y los créditos
#     recientes aún no tuvieron tiempo de incumplir (sesgo de madurez, ver EDA): el modelo
#     aprendería "fecha reciente = menos impagos", una relación que no se sostendría en producción.
COLS_EXCLUIR = ["fecha_prestamo", "dias_desde_inicio"]
COLS_CATEGORICAS = ["tipo_credito", "tipo_laboral"]
COLS_ORDINALES = {                          # columna -> orden de menor a mayor
    "tendencia_ingresos": ["Decreciente", "Estable", "Creciente"],
}
TOL_RARAS = 0.01                            # categorías con < 1% se agrupan en "Rare"


# =============================================================================
# 1. CÁLCULO AUTOMÁTICO DE RATIOS (primer paso del pipeline)
# =============================================================================
class CalculaRatios(BaseEstimator, TransformerMixin):
    """
    Calcula los dos ratios del modelo a partir de las variables base que ingresa el usuario:

        cuota_sobre_salario = cuota_pactada   / salario_cliente
        huella_por_credito  = huella_consulta / (cant_creditosvigentes + 1)

    - Al predecir, el usuario ingresa `salario_cliente` y `cant_creditosvigentes` (no los ratios).
    - Las variables base NO son features del modelo: se descartan después de calcular los ratios.
    - Si los datos ya traen los ratios y no traen las variables base (caso del CSV de
      entrenamiento), los deja como están.
    - Salario <= 0 o vacío -> ratio NaN (lo resuelve luego el imputador de medianas).
    - Reordena las columnas igual que en el entrenamiento, porque las binarias pasan por
      `remainder="passthrough"` y el modelo depende del orden.
    """
    COLS_BASE = ["salario_cliente", "cant_creditosvigentes"]

    def _calcular(self, X):
        X = X.copy()
        if "salario_cliente" in X.columns:
            salario = pd.to_numeric(X["salario_cliente"], errors="coerce")
            X["cuota_sobre_salario"] = X["cuota_pactada"] / salario.where(salario > 0)
        if "cant_creditosvigentes" in X.columns:
            vigentes = pd.to_numeric(X["cant_creditosvigentes"], errors="coerce")
            X["huella_por_credito"] = X["huella_consulta"] / (vigentes + 1)
        return X.replace([np.inf, -np.inf], np.nan)

    def fit(self, X, y=None):
        X = self._calcular(X)
        self.columnas_ = [c for c in X.columns if c not in self.COLS_BASE]
        return self

    def transform(self, X):
        X = self._calcular(X)
        faltan = [c for c in self.columnas_ if c not in X.columns]
        if faltan:
            raise ValueError(f"Faltan columnas de entrada: {faltan}")
        return X[self.columnas_]


# =============================================================================
# 2. PIPELINE DE PREPROCESAMIENTO (feature engineering)
# =============================================================================
def build_preprocessor(cols_num, cols_cat, cols_ord, cols_scale):
    """
    Arma el pipeline completo de preprocesamiento.

    Etapa 1 (feature-engine, trabaja sobre DataFrames):
        imputar numéricas -> imputar categóricas/ordinales -> agrupar raras -> one-hot
    Etapa 2 (sklearn, ColumnTransformer):
        ordinal (OrdinalEncoder) + escalado (StandardScaler) + resto sin tocar
    """
    # --- Etapa 1: imputación y codificación nominal (feature-engine) ---------
    pipe_imputacion_codificacion = Pipeline([
        ("imp_num", MeanMedianImputer(imputation_method="median", variables=cols_num)),
        ("imp_cat", CategoricalImputer(imputation_method="frequent",
                                       variables=cols_cat + cols_ord,
                                       ignore_format=True)),
        ("raras", RareLabelEncoder(tol=TOL_RARAS, n_categories=1,
                                   variables=cols_cat, ignore_format=True)),
        ("onehot", OneHotEncoder(variables=cols_cat, drop_last=False,
                                 ignore_format=True)),
    ])

    # --- Etapa 2: ordinal y escalado (sklearn) --------------------------------
    pipe_ord = Pipeline([
        ("ordinal", OrdinalEncoder(categories=list(COLS_ORDINALES.values()),
                                   handle_unknown="use_encoded_value",
                                   unknown_value=-1)),
    ])
    pipe_scale = Pipeline([
        ("scaler", StandardScaler()),
    ])

    pipe_ord_escalado = ColumnTransformer([
        ("ord", pipe_ord, cols_ord),
        ("scale", pipe_scale, cols_scale),
    ], remainder="passthrough",             # dummies del one-hot y binarias
        verbose_feature_names_out=False).set_output(transform="pandas")   # devuelve DataFrames con nombres

    # --- Pipeline de preprocesamiento: todo encadenado ------------------------
    return Pipeline([
        ("imputacion_codificacion", pipe_imputacion_codificacion),
        ("ordinal_escalado", pipe_ord_escalado),
    ])


# =============================================================================
# 3. DATOS: lectura y split
# =============================================================================
df = pd.read_csv(RUTA_CSV).drop(columns=COLS_EXCLUIR, errors="ignore")   # la base final ya no trae fecha_prestamo

# Booleanas -> 0/1 y categóricas numéricas (tipo_credito) -> texto
df[df.select_dtypes("bool").columns] = df.select_dtypes("bool").astype(int)
df[COLS_CATEGORICAS] = df[COLS_CATEGORICAS].astype("object")

X, y = df.drop(columns=TARGET), df[TARGET]

# Split 80/20 estratificado (hay desbalance: ~95% pagan a tiempo)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=SEMILLA)

# Tipos de columnas
cols_ord = list(COLS_ORDINALES)
cols_cat = list(COLS_CATEGORICAS)
cols_num = [c for c in X.columns if c not in cols_ord + cols_cat]
# Las binarias (0/1) no se escalan; el resto de numéricas sí
cols_scale = [c for c in cols_num if X_train[c].nunique(dropna=True) > 2]

# =============================================================================
# 4. PIPELINE FINAL: ratios -> features -> modelo (un solo Pipeline, ajustado SOLO con train)
# =============================================================================
preprocesador = build_preprocessor(cols_num, cols_cat, cols_ord, cols_scale)

pipe_final = Pipeline([
    ("ratios", CalculaRatios()),                                # calcula los 2 ratios a partir de los datos ingresados
    ("features", preprocesador),
    ("modelo", LogisticRegression(class_weight="balanced")),   # regresión logística balanceada (resto por defecto)
]).fit(X_train, y_train)    # fit solo con train (evita fuga de información)

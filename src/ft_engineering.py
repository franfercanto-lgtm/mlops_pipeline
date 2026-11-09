"""
Componente 1: Ingeniería de features (ft_engineering.py)
--------------------------------------------------------
Incluye, todo dentro de Pipelines:
  - Imputación           (feature-engine: MeanMedianImputer, CategoricalImputer)
  - Codificación         (feature-engine: RareLabelEncoder, OneHotEncoder
                          sklearn: OrdinalEncoder para la variable ordinal)
  - Escalado             (sklearn: StandardScaler)

Uso desde otro archivo:
    from ft_engineering import build_features
    X_train, X_test, y_train, y_test, preprocesador = build_features()

Requiere: pip install pandas scikit-learn feature-engine
"""

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from feature_engine.encoding import OneHotEncoder, RareLabelEncoder

# =============================================================================
# 0. CONFIGURACIÓN (todo lo que quieras cambiar está acá)
# =============================================================================
# El CSV se busca junto a este archivo (no depende de desde dónde ejecutes)
RUTA_CSV = Path(__file__).resolve().parent / "Base_de_datos_limpia.csv"
TARGET = "Pago_atiempo"
SEMILLA = 42
TEST_SIZE = 0.20

COLS_EXCLUIR = ["fecha_prestamo"]           # texto de fecha: no entra como feature
COLS_CATEGORICAS = ["tipo_credito", "tipo_laboral"]
COLS_ORDINALES = {                          # columna -> orden de menor a mayor
    "tendencia_ingresos": ["Decreciente", "Estable", "Creciente"],
}
TOL_RARAS = 0.01                            # categorías con < 1% se agrupan en "Rare"


# =============================================================================
# 1. PIPELINES
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

    # --- Pipeline final: todo encadenado --------------------------------------
    return Pipeline([
        ("imputacion_codificacion", pipe_imputacion_codificacion),
        ("ordinal_escalado", pipe_ord_escalado),
    ])


# =============================================================================
# 2. FEATURES
# =============================================================================
def build_features(ruta=RUTA_CSV):
    """Lee el CSV, separa train/test y devuelve los datasets transformados."""
    df = pd.read_csv(ruta).drop(columns=COLS_EXCLUIR)

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

    # fit SOLO con train (evita fuga de información), transform en ambos
    pre = build_preprocessor(cols_num, cols_cat, cols_ord, cols_scale)
    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)

    return X_train_t, X_test_t, y_train, y_test, pre


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, _ = build_features()
    print("Train:", X_tr.shape, "| Test:", X_te.shape)
    print("Nulos restantes:", int(X_tr.isna().sum().sum() + X_te.isna().sum().sum()))
    print(X_tr.head())

"""
custom_transformers.py: transformadores propios del modelo.
----------------------------------------------------
Viven en un módulo aparte (y no en model_deploy.py) por una razón técnica: joblib guarda
la *referencia* a la clase ("custom_transformers.CalculaRatios"), no su código. Tanto el script de
entrenamiento (model_deploy.py) como la API (main.py) importan la clase desde acá, así el
objeto se puede deserializar sin tener que re-ejecutar el entrenamiento.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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

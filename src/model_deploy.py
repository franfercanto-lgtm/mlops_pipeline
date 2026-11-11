"""model_deploy.py: modelo final (Regresión Logística) entrenado con el feature engineering."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ft_engineering import build_features

# 1. Feature engineering (datos transformados y preprocesador ya ajustado)
X_train, X_test, y_train, y_test, preprocesador = build_features()

# 2. Pipeline del modelo: regresión logística balanceada (resto por defecto)
pipe_modelo = Pipeline([
    ("lr", LogisticRegression(class_weight="balanced")),
]).fit(X_train, y_train)

# 3. Pipeline final: features -> modelo (ambos ya ajustados)
pipe_final = Pipeline([
    ("features", preprocesador),
    ("modelo", pipe_modelo),
])

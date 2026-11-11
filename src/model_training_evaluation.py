"""
Componente 2 del flujo de modelos operativos: Pago a tiempo
-----------------------------------------------------------
1) Importa build_features() desde ft_engineering.py (train/test ya transformados).
2) PyCaret compara modelos (AUC) sobre esos datos.
3) Gráficos comparativos de los mejores modelos.
4) GridSearch de la Regresión Logística (class_weight='balanced').
5) Odds, odds ratio y peso de los coeficientes de la Regresión Logística final.

Requiere: pycaret>=3.0 (Python 3.9-3.11), scikit-learn, pandas, matplotlib,
          feature-engine
Instalación: pip install "pycaret>=3.3" matplotlib six feature-engine
(si falta "six", PyCaret falla al importar: ModuleNotFoundError)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)

from ft_engineering import build_features, TARGET, SEMILLA

# =============================================================================
# 0. CONFIGURACIÓN
# =============================================================================
N_TOP = 3                                   # modelos "principales" a comparar en gráficos


# =============================================================================
# 1. FUNCIONES REUTILIZABLES
# =============================================================================
def _score(model, X):
    """Probabilidad de clase 1 (o decision_function si el modelo no tiene proba)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def summarize_classification(model, X, y, nombre="", umbral=0.5):
    """Métricas de clasificación de un modelo sobre (X, y). Devuelve un dict."""
    score = _score(model, X)
    pred = model.predict(X) if not hasattr(model, "predict_proba") else (score >= umbral).astype(int)
    return {
        "Modelo": nombre,
        "AUC": roc_auc_score(y, score),
        "Accuracy": accuracy_score(y, pred),
        "Balanced_Acc": balanced_accuracy_score(y, pred),
        "Recall": recall_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred),
    }


def build_model(model_id, X_train, y_train, X_test, y_test):
    """Entrena un modelo con PyCaret (CV sobre train) y lo resume en train y test."""
    from pycaret.classification import create_model
    modelo = create_model(model_id, verbose=False)
    resumen = pd.DataFrame([
        {**summarize_classification(modelo, X_train, y_train, model_id), "Set": "train"},
        {**summarize_classification(modelo, X_test, y_test, model_id), "Set": "test"},
    ])
    return modelo, resumen


# =============================================================================
# 2. GRÁFICOS COMPARATIVOS
# =============================================================================
def graficos_comparativos(modelos, resumen, X_test, y_test):
    """3 gráficos: AUC train vs test, curvas ROC superpuestas y matrices de confusión."""
    # a) Barras AUC train vs test (detecta sobreajuste)
    piv = resumen.pivot(index="Modelo", columns="Set", values="AUC")[["train", "test"]]
    piv.plot(kind="bar", figsize=(7, 4), rot=0, title="AUC: train vs test")
    plt.ylabel("AUC"); plt.ylim(0.5, 1.0); plt.tight_layout(); plt.show()

    # b) Curvas ROC superpuestas (test)
    plt.figure(figsize=(6, 5))
    for nombre, m in modelos.items():
        fpr, tpr, _ = roc_curve(y_test, _score(m, X_test))
        plt.plot(fpr, tpr, label=f"{nombre} (AUC={roc_auc_score(y_test, _score(m, X_test)):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("Falsos positivos"); plt.ylabel("Verdaderos positivos")
    plt.title("Curvas ROC (test)"); plt.legend(); plt.tight_layout(); plt.show()

    # c) Matrices de confusión lado a lado (test)
    fig, axes = plt.subplots(1, len(modelos), figsize=(4.5 * len(modelos), 4))
    for ax, (nombre, m) in zip(np.atleast_1d(axes), modelos.items()):
        cm = confusion_matrix(y_test, m.predict(X_test))
        ax.imshow(cm, cmap="Blues")
        ax.set_title(nombre); ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center")
    plt.tight_layout(); plt.show()


# =============================================================================
# 3. FLUJO PRINCIPAL
# =============================================================================
if __name__ == "__main__":
    from pycaret.classification import setup, compare_models, pull

    # --- Componente 1: features ----------------------------------------------
    X_train, X_test, y_train, y_test, preprocesador = build_features()

    print("Train:", X_train.shape, "| Test:", X_test.shape)
    print("Target train:\n", y_train.value_counts(normalize=True).round(3))
    print("Nulos restantes:", int(X_train.isna().sum().sum() + X_test.isna().sum().sum()))

    # --- PyCaret: sin preprocesamiento propio (ya viene transformado) --------
    df_train = X_train.assign(**{TARGET: y_train})
    df_test = X_test.assign(**{TARGET: y_test})
    setup(data=df_train, test_data=df_test, target=TARGET,
          preprocess=False,                      # el ColumnTransformer ya hizo el trabajo
          fold_strategy="stratifiedkfold", fold=5,
          session_id=SEMILLA, verbose=False)

    # --- Comparar todos los modelos por AUC (CV sobre train) -----------------
    compare_models(sort="AUC")
    tabla = pull()
    top_ids = tabla.index[:N_TOP].tolist()       # ids de PyCaret: 'lr', 'rf', 'lightgbm'...
    print("Top modelos:", top_ids)

    # --- Entrenar y resumir los principales ----------------------------------
    modelos, resumenes = {}, []
    for mid in top_ids:
        modelos[mid], r = build_model(mid, X_train, y_train, X_test, y_test)
        resumenes.append(r)
    resumen = pd.concat(resumenes, ignore_index=True)
    print(resumen.round(3).to_string(index=False))

    # --- Gráficos ------------------------------------------------------------
    graficos_comparativos(modelos, resumen, X_test, y_test)


# %% [markdown]
# # Conclusión: mejor modelo según AUC
#
# Resultados de ejecutar `model_training_evaluation.py` sobre `Base_de_datos_limpia.csv`
# (10.763 registros: 8.610 train / 2.153 test; 95,2 % pagan a tiempo).
#
# `compare_models` evaluó 14 algoritmos con validación cruzada (5 folds) y los 3 mejores
# se resumieron en train y test:
#
# | Modelo                        | AUC CV (5 folds) | AUC train | AUC test |
# |-------------------------------|:----------------:|:---------:|:--------:|
# | `gbc`   Gradient Boosting     | 0,669            | 0,860     | 0,679    |
# | `lr`    Logistic Regression   | 0,660            | 0,688     | **0,689** |
# | `ridge` Ridge Classifier      | 0,656            | 0,682     | 0,686    |
#
# ## Modelo elegido: Regresión Logística (`lr`)
#
# - **Mayor AUC en test (0,689)**, que es el dato que no se usó para entrenar ni ajustar.
# - **No sobreajusta**: AUC train 0,688 vs test 0,689. En cambio `gbc` pasa de 0,860 en
#   train a 0,679 en test, es decir, memoriza el train y no generaliza.
# - Queda a solo 0,009 del primer puesto en validación cruzada (`gbc`, 0,669), y es un
#   modelo más simple, más rápido y más fácil de explicar.
#
# ## Advertencias
#
# - **Las diferencias entre los 3 modelos son pequeñas** (menos de 0,015 de AUC), así que
#   `lr` y `ridge` son prácticamente equivalentes. La elección de `lr` sobre `gbc` se apoya
#   sobre todo en que `gbc` sobreajusta.
# - **El poder predictivo general es modesto** (AUC cercano a 0,69). La Balanced Accuracy
#   ronda 0,50: con umbral 0,5 los modelos predicen casi siempre "paga a tiempo", por lo
#   que Accuracy, Recall y F1 (calculados sobre la clase mayoritaria) se ven altos pero
#   no aportan información.
# - Próximos pasos sugeridos: ajustar el umbral de decisión, probar `class_weight` o
#   remuestreo, y medir Recall/Precision sobre la clase 0 (los que no pagan).


# =============================================================================
# 4. GRIDSEARCH: REGRESIÓN LOGÍSTICA (class_weight="balanced")
# =============================================================================
# class_weight queda FIJO en "balanced" (por el desbalance ~95/5); se busca el resto.
PARAM_GRID_LR = {
    "penalty": ["l1", "l2"],                 # tipo de regularización
    "C": [0.0001, 0.001, 0.01, 0.1, 1, 10],  # inversa de la fuerza de regularización
}


def _fila(nombre, modelo, X, y):
    """Métricas del modelo + Recall de la clase 0 (los que NO pagan a tiempo)."""
    fila = summarize_classification(modelo, X, y, nombre)
    pred = (_score(modelo, X) >= 0.5).astype(int)
    fila["Recall_clase0"] = recall_score(y, pred, pos_label=0)
    return fila


def grid_search_lr(X_train, y_train, X_test, y_test, modelos):
    """GridSearch (CV 5 folds sobre train, scoring AUC). El test solo se usa para reportar."""
    lr = LogisticRegression(class_weight="balanced", solver="liblinear",
                            max_iter=1000, random_state=SEMILLA)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    gs = GridSearchCV(lr, PARAM_GRID_LR, scoring="roc_auc", cv=cv, n_jobs=-1)
    gs.fit(X_train, y_train)

    tabla = (pd.DataFrame(gs.cv_results_)
             [["param_penalty", "param_C", "mean_test_score", "std_test_score", "rank_test_score"]]
             .sort_values("rank_test_score").reset_index(drop=True))

    # Comparación en test: LR de PyCaret (sin pesos) vs balanced sin tunear vs balanced tuneado
    base = modelos["lr"] if "lr" in modelos else build_model("lr", X_train, y_train, X_test, y_test)[0]
    balanced = LogisticRegression(class_weight="balanced", solver="liblinear",
                                  max_iter=1000, random_state=SEMILLA).fit(X_train, y_train)
    comparacion = pd.DataFrame([
        _fila("lr PyCaret (sin pesos)", base, X_test, y_test),
        _fila("lr balanced (C=1, l2)", balanced, X_test, y_test),
        _fila("lr balanced + GridSearch", gs.best_estimator_, X_test, y_test),
    ])
    return gs, tabla, comparacion


# Usa las variables de la sección 3 (X_train, modelos, etc.)
if __name__ == "__main__":
    gs, tabla_gs, comparacion = grid_search_lr(X_train, y_train, X_test, y_test, modelos)

    print("\n=== GridSearch Regresión Logística (AUC en CV, 5 folds) ===")
    print(tabla_gs.round(4).to_string(index=False))
    print("\nMejores hiperparámetros:", gs.best_params_)
    print(f"Mejor AUC CV: {gs.best_score_:.4f}")
    print("\n=== Comparación en test ===")
    print(comparacion.round(3).to_string(index=False))


# %% [markdown]
# # Hiperparámetros a utilizar: Regresión Logística
#
# GridSearch con validación cruzada estratificada (5 folds, métrica AUC) sobre train, con
# `class_weight="balanced"` fijo. Se probaron 12 combinaciones (2 penalizaciones x 6 valores de `C`).
#
# ## Mejor combinación del GridSearch (sólo de referencia)
#
# | Hiperparámetro | Valor            | Por qué                                                        |
# |----------------|------------------|----------------------------------------------------------------|
# | `class_weight` | `"balanced"`     | Fijado de antemano: sólo ~5 % de los casos no paga a tiempo.   |
# | `penalty`      | `"l2"`           | Mejor AUC CV en casi todos los `C`; `l1` con `C` bajo anula los coeficientes (AUC 0,54 con `C=0,001` y 0,50 con `C=0,0001`). |
# | `C`            | `0.001`          | Mejor AUC CV (0,667). Con `C=0,0001` baja a 0,665, así que el óptimo no está en el borde de la grilla. |
# | `solver`       | `"liblinear"`    | Fijo; soporta `l1` y `l2`.                                      |
# | `max_iter`     | `1000`           | Fijo; asegura la convergencia.                                  |
#
# ## Decisión para producción: sólo `class_weight="balanced"`, todo lo demás por defecto
#
# Ninguna combinación del grid mejoró el AUC de forma real:
#
# - En validación cruzada, la mejor combinación (0,667) supera a la configuración por defecto
#   con `balanced` (`C=1`, `l2`: 0,659) por menos de 0,01, muy por debajo de la desviación
#   entre folds (~0,03).
# - En test, el modelo "tuneado" quedó incluso peor: AUC 0,669 contra 0,683 con `C=1`.
#
# Como afinar `penalty` y `C` no aporta una mejora medible, en producción conviene la
# configuración más simple:
#
# ```python
# LogisticRegression(class_weight="balanced")
# # resto por defecto: penalty="l2", C=1.0, solver="lbfgs", max_iter=100
# ```
#
# Converge con los valores por defecto (90 de 100 iteraciones permitidas) y en test da
# AUC 0,684, Balanced Accuracy 0,634 y Recall de la clase 0 de 0,608.
#
# ## Qué significa esta elección
#
# - **`balanced` cambia el comportamiento del modelo:** en test, el Recall de la clase 0 (los que
#   no pagan) pasa de 0,00 a 0,61 y la Balanced Accuracy de 0,50 a 0,63. A cambio, la Accuracy
#   baja de 0,95 a 0,66, algo esperado: el modelo ya no predice "paga a tiempo" para todos.
# - **Menos hiperparámetros, menos riesgo:** una configuración por defecto es más fácil de
#   reproducir y no depende de una búsqueda cuyo resultado está dentro del ruido.
#
# ## Advertencias
#
# - **Las diferencias entre combinaciones están dentro del ruido:** la desviación estándar entre
#   folds ronda 0,03 y el AUC de las mejores combinaciones difiere en menos de 0,01.
# - **Con ~100 casos de clase 0 en test,** las diferencias de AUC entre configuraciones también
#   caen dentro del margen de error.
# - **Próximo paso con más impacto:** el modelo sigue siendo débil (AUC < 0,70). Conviene ajustar
#   el umbral de decisión según el costo de cada error y revisar si hay variables nuevas que
#   aporten señal, más que seguir afinando hiperparámetros.


# =============================================================================
# 5. COEFICIENTES: ODDS, ODDS RATIO Y PESO DE LAS FEATURES
# =============================================================================
def modelo_produccion():
    """Configuración final para producción: sólo class_weight, el resto por defecto."""
    return LogisticRegression(class_weight="balanced")


def tabla_coeficientes(modelo, X):
    """
    Una fila por feature:
      Coef          peso del coeficiente (log-odds)
      Odds_Ratio    exp(Coef): cuánto se multiplican los odds de pagar a tiempo
      Cambio_odds_% (Odds_Ratio - 1) * 100
      Peso_abs      |Coef|: sirve para ordenar por importancia
      Corr_max      mayor correlación (en valor absoluto) con OTRA feature
      Corr_con      con cuál feature (si Corr_max es alta, el coeficiente es poco confiable)
    """
    coef = pd.Series(modelo.coef_[0], index=X.columns)

    corr = X.corr().abs().to_numpy(copy=True)
    np.fill_diagonal(corr, 0)
    corr = pd.DataFrame(corr, index=X.columns, columns=X.columns)

    tabla = pd.DataFrame({
        "Coef": coef,
        "Odds_Ratio": np.exp(coef),
        "Cambio_odds_%": (np.exp(coef) - 1) * 100,
        "Peso_abs": coef.abs(),
        "Corr_max": corr.max(axis=1),
        "Corr_con": corr.idxmax(axis=1),
    })
    return tabla.sort_values("Peso_abs", ascending=False)


N_BOOT = 1000                               # remuestreos para el intervalo de confianza del odds ratio


def intervalos_odds_ratio(X, y, n_boot=N_BOOT, nivel=0.95):
    """
    Intervalo de confianza (por percentiles) del odds ratio de cada feature, por bootstrap:
    se reentrena la LR final sobre `n_boot` remuestras (con reemplazo) del conjunto de entrenamiento.
    Si el intervalo incluye 1, el odds ratio no se distingue de "sin efecto".
    (max_iter=1000 sólo aquí: con remuestras, el valor por defecto (100) suele cortar antes de converger.)
    """
    rng = np.random.default_rng(SEMILLA)
    coefs = np.empty((n_boot, X.shape[1]))
    for b in range(n_boot):
        i = rng.integers(0, len(X), len(X))
        coefs[b] = modelo_produccion().set_params(max_iter=1000).fit(X.iloc[i], y.iloc[i]).coef_[0]
    a = (1 - nivel) / 2
    inf, sup = np.exp(np.quantile(coefs, [a, 1 - a], axis=0))
    return pd.DataFrame({"OR_IC95_inf": inf, "OR_IC95_sup": sup,
                         "Incluye_1": (inf <= 1) & (sup >= 1)}, index=X.columns)


def grafico_coeficientes(tabla, top=15):
    """Barras horizontales de los `top` coeficientes con más peso."""
    t = tabla.head(top).iloc[::-1]
    colores = ["tab:green" if c > 0 else "tab:red" for c in t["Coef"]]
    plt.figure(figsize=(8, 5.5))
    plt.barh(t.index, t["Coef"], color=colores)
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel("Coeficiente (log-odds)")
    plt.title(f"Top {top} features por peso (verde: sube odds de pagar, rojo: baja)")
    plt.tight_layout(); plt.show()


# Criterios para la comprobación de más abajo
UMBRAL_PESO_BAJO = 0.05                     # |coef| < 0.05  ->  odds ratio entre 0,95 y 1,05
COLS_RATIOS = [                             # variables ratio creadas para probar
    "cuota_sobre_salario", "capital_sobre_salario", "otros_prestamos_sobre_salario",
    "cuota_sobre_ingreso_buro", "ratio_salario_vs_buro", "ratio_mora_sobre_saldo",
    "saldo_total_sobre_capital", "pct_sector_financiero", "huella_por_credito",
    "intensidad_busqueda",
]


def auc_sin(X_train, y_train, X_test, y_test, descartar):
    """AUC (CV 5 folds en train, y test) de la LR final sin las columnas `descartar`."""
    cols = [c for c in X_train.columns if c not in descartar]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    auc_cv = cross_val_score(modelo_produccion(), X_train[cols], y_train,
                             scoring="roc_auc", cv=cv).mean()
    modelo = modelo_produccion().fit(X_train[cols], y_train)
    auc_test = roc_auc_score(y_test, modelo.predict_proba(X_test[cols])[:, 1])
    return {"n_features": len(cols), "AUC_CV": auc_cv, "AUC_test": auc_test}


# Usa las variables de la sección 3 (X_train, X_test, etc.)
if __name__ == "__main__":
    lr_final = modelo_produccion().fit(X_train, y_train)
    tabla_coef = tabla_coeficientes(lr_final, X_train)

    print("\n=== Regresión Logística final (class_weight='balanced', resto por defecto) ===")
    print(pd.DataFrame([_fila("lr final (test)", lr_final, X_test, y_test)]).round(3).to_string(index=False))
    print(f"\nOdds base (intercepto, con las features en 0): {np.exp(lr_final.intercept_[0]):.3f}")
    print("Odds Ratio > 1: sube los odds de pagar a tiempo | < 1: los baja")
    print("(numéricas escaladas: por cada +1 desviación estándar; dummies/binarias: de 0 a 1)\n")
    print(tabla_coef.round(3).to_string())

    # --- Intervalo de confianza (95 %) del odds ratio, por bootstrap ---------
    ic = intervalos_odds_ratio(X_train, y_train)
    tabla_coef = tabla_coef.join(ic)
    print(f"\n=== Odds Ratio con IC 95 % (bootstrap, {N_BOOT} remuestras) ===")
    print(tabla_coef[["Odds_Ratio", "OR_IC95_inf", "OR_IC95_sup", "Incluye_1"]].round(3).to_string())
    print(f"\nFeatures cuyo IC 95 % incluye 1 (sin efecto distinguible): {ic.index[ic['Incluye_1']].tolist()}")

    # --- Comprobación: ¿qué pasa si se descartan? ---------------------------
    bajo_peso = tabla_coef.index[tabla_coef["Peso_abs"] < UMBRAL_PESO_BAJO].tolist()
    print(f"\nFeatures con |coef| < {UMBRAL_PESO_BAJO} ({len(bajo_peso)}): {bajo_peso}")
    comparacion_descarte = pd.DataFrame({
        "todas": auc_sin(X_train, y_train, X_test, y_test, []),
        "sin las de bajo peso": auc_sin(X_train, y_train, X_test, y_test, bajo_peso),
        "sin los ratios creados": auc_sin(X_train, y_train, X_test, y_test, COLS_RATIOS),
    }).T
    print(comparacion_descarte.round(4).to_string())

    grafico_coeficientes(tabla_coef)


# %% [markdown]
# # Qué dicen los coeficientes
#
# Modelo: `LogisticRegression(class_weight="balanced")`. Las numéricas están estandarizadas,
# así que sus pesos son comparables entre sí. Para cada conclusión se reentrenó el modelo
# sin esas variables (ver `auc_sin`); las diferencias de AUC están dentro del ruido
# (desviación de ~0,03 entre folds), por lo que lo que se afirma es que **descartarlas no empeora**.
#
# ## 1) Features con muy poco peso: conviene descartarlas
#
# Nueve features con |coef| < 0,05, es decir, mueven los odds menos de un 5 %:
#
# | Feature | Coef | Odds Ratio |
# |---|:-:|:-:|
# | `saldo_total_sobre_capital` | +0,048 | 1,050 |
# | `creditos_sectorCooperativo` | +0,047 | 1,048 |
# | `cant_creditosvigentes` | +0,043 | 1,044 |
# | `pct_sector_financiero` | -0,043 | 0,958 |
# | `saldo_total` | +0,033 | 1,034 |
# | `saldo_mora` | -0,020 | 0,980 |
# | `saldo_principal` | +0,015 | 1,015 |
# | `salario_cliente` | +0,008 | 1,008 |
# | `creditos_sectorFinanciero` | +0,002 | 1,002 |
#
# Sin estas 9 features el modelo no pierde: AUC CV 0,659 -> 0,669 y AUC test 0,684 -> 0,688.
#
# ## 2) Variables ratio creadas para probar: no aportan
#
# Se crearon 10 ratios (`cuota_sobre_salario`, `capital_sobre_salario`, `otros_prestamos_sobre_salario`,
# `cuota_sobre_ingreso_buro`, `ratio_salario_vs_buro`, `ratio_mora_sobre_saldo`,
# `saldo_total_sobre_capital`, `pct_sector_financiero`, `huella_por_credito`, `intensidad_busqueda`).
#
# - **Siete tienen peso bajo** (|coef| entre 0,04 y 0,17).
# - **Tres parecen pesar mucho** (`cuota_sobre_salario` +1,76, `capital_sobre_salario` -0,89,
#   `otros_prestamos_sobre_salario` +0,58), pero es engañoso: las dos primeras tienen correlación de
#   0,998 entre sí y sus pesos se cancelan, y quitar cualquiera de ellas no cambia el AUC.
# - **Resultado:** sin los 10 ratios el AUC no baja: CV 0,659 -> 0,662 y test 0,684 -> 0,688.
#
# Conclusión: los ratios no mejoran el modelo, se pueden descartar.

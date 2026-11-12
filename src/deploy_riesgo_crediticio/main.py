"""
main.py: API de scoring crediticio (FastAPI).
---------------------------------------------
Núcleo del despliegue. Responsabilidades:
  1. Cargar los 2 joblibs entrenados con model_deploy.py (preprocesador + modelo).
  2. Validar lo que llega, aplicar las transformaciones y predecir.
  3. Exponer el servicio:  GET /health   y   POST /predict  (CSV, predicción por lotes).

Ejecutar local:  uvicorn main:app --reload      (docs interactivas en http://localhost:8000/docs)
"""

import csv
import io
import os
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, ValidationError
from sklearn.exceptions import InconsistentVersionWarning

import custom_transformers  # noqa: F401  (necesario: joblib importa CalculaRatios desde este módulo)
from schemas import BOOLEANAS, CATEGORICAS, COLUMNAS_ENTRADA, RATIOS, Registro

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
DIR_ARTEFACTOS = Path(os.getenv("DIR_ARTEFACTOS", Path(__file__).resolve().parent / "artifacts"))
UMBRAL = float(os.getenv("UMBRAL", "0.5"))     # P(paga a tiempo) >= UMBRAL  ->  clase 1
MAX_ERRORES_MOSTRADOS = 50

# Pago_atiempo = 1 -> el cliente PAGÓ A TIEMPO ; Pago_atiempo = 0 -> NO pagó a tiempo.
# (El ~95% de la base es clase 1.) Si se necesitara invertir la lectura, se cambia SOLO acá.
MENSAJES = {
    1: "El cliente tiene una alta probabilidad de pagar a tiempo",
    0: "El cliente posee un perfil de ALTO RIESGO crediticio",
}

# NOTA - FORMATO CSV:
#   Se eligió CSV (y no JSON) como único formato de entrada porque es un formato mucho más
#   conocido por personas ajenas al mundo del ML (se arma y se revisa desde Excel), que es
#   quien va a consumir este servicio. Excel en español suele guardar con ";" como separador:
#   ambos separadores ("," y ";") están soportados. Los decimales deben ir con punto.

estado = {}   # acá viven el preprocesador y el modelo, cargados una sola vez al arrancar


# =============================================================================
# CARGA DEL MODELO (una vez, al iniciar el servidor)
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Si la versión de scikit-learn del contenedor difiere de la del entrenamiento, el modelo
    # podría dar resultados incorrectos sin avisar: preferimos que el servidor NO arranque.
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        preprocesador = joblib.load(DIR_ARTEFACTOS / "preprocesador.joblib")
        modelo = joblib.load(DIR_ARTEFACTOS / "modelo.joblib")

    # Chequeo de contrato: las columnas que pide el modelo deben coincidir con schemas.py
    ratios_ = preprocesador.named_steps["ratios"]
    esperadas = [c for c in ratios_.columnas_ if c not in RATIOS] + ratios_.COLS_BASE
    if set(esperadas) != set(COLUMNAS_ENTRADA):
        raise RuntimeError("schemas.py no coincide con el modelo entrenado. "
                           f"Sobran: {sorted(set(COLUMNAS_ENTRADA) - set(esperadas))}. "
                           f"Faltan: {sorted(set(esperadas) - set(COLUMNAS_ENTRADA))}.")

    estado["preprocesador"] = preprocesador
    estado["modelo"] = modelo
    estado["idx_clase_1"] = list(modelo.classes_).index(1)
    yield
    estado.clear()


app = FastAPI(
    title="API de riesgo crediticio",
    description="Predice, por lote (CSV), si un cliente pagará a tiempo su crédito.",
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# MODELOS DE RESPUESTA
# =============================================================================
class Resultado(BaseModel):
    fila: int                          # número de fila en el archivo (como se ve en Excel; la 1 es el encabezado)
    prediccion: int                    # 1 = paga a tiempo | 0 = alto riesgo
    probabilidad_pago_atiempo: float
    mensaje: str


class RespuestaBatch(BaseModel):
    n_registros: int
    umbral: float
    resultados: list[Resultado]


# =============================================================================
# LÓGICA: lectura, validación y predicción
# =============================================================================
def _error(mensaje: str, **extra):
    return HTTPException(status_code=422, detail={"mensaje": mensaje, **extra})


def _leer_csv(contenido: bytes) -> pd.DataFrame:
    """Lee el CSV como TEXTO (sin inferir tipos: los tipos los valida Pydantic fila por fila)."""
    try:
        texto = contenido.decode("utf-8-sig")        # utf-8-sig tolera el BOM que agrega Excel
    except UnicodeDecodeError:
        texto = contenido.decode("latin-1")          # CSV guardado como "ANSI" desde Excel
    if not texto.strip():
        raise _error("El archivo está vacío.")

    primera = texto.splitlines()[0]
    sep = ";" if primera.count(";") > primera.count(",") else ","
    try:
        df = pd.read_csv(io.StringIO(texto), sep=sep, dtype=str, keep_default_na=False,
                         skip_blank_lines=False, quoting=csv.QUOTE_MINIMAL)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        raise _error(f"No se pudo leer el archivo como CSV: {e}")

    df.columns = [c.strip() for c in df.columns]
    df.index = df.index + 2                          # índice = nº de fila de Excel (1 = encabezado)
    df = df[~(df.apply(lambda col: col.str.strip()) == "").all(axis=1)]   # descarta filas totalmente vacías
    if df.empty:
        raise _error("El archivo no tiene filas de datos.")
    return df


def _validar_columnas(df: pd.DataFrame) -> None:
    faltan = [c for c in COLUMNAS_ENTRADA if c not in df.columns]
    sobran = [c for c in df.columns if c not in COLUMNAS_ENTRADA]
    if faltan or sobran:
        raise _error(
            "Las columnas del archivo no coinciden con las esperadas (validación estricta). "
            "Recordá que los ratios NO se envían (se calculan solos) y que no debe incluirse el target.",
            columnas_faltantes=faltan, columnas_no_reconocidas=sobran)


_TRADUCCION = {
    "missing": "campo obligatorio",
    "float_type": "campo obligatorio: no puede estar vacío",
    "int_type": "campo obligatorio: no puede estar vacío",
    "bool_type": "campo obligatorio: no puede estar vacío",
    "string_type": "campo obligatorio: no puede estar vacío",
    "float_parsing": "debe ser un número (decimales con punto)",
    "int_parsing": "debe ser un número entero",
    "bool_parsing": "debe ser True/False (o 1/0)",
}


_LIMITES = {"greater_than": ("gt", "debe ser mayor a {}"),
            "greater_than_equal": ("ge", "debe ser mayor o igual a {}"),
            "less_than": ("lt", "debe ser menor a {}"),
            "less_than_equal": ("le", "debe ser menor o igual a {}")}


def _describir(err: dict) -> str:
    tipo = err["type"]
    if tipo in _LIMITES:
        clave, plantilla = _LIMITES[tipo]
        return plantilla.format(err["ctx"][clave])
    if tipo == "literal_error":
        return f"valor no permitido (opciones: {err['ctx']['expected']})"
    if tipo == "value_error":
        return err["msg"].removeprefix("Value error, ")        # reglas de coherencia de schemas.py
    if tipo == "finite_number":
        return "debe ser un número finito"
    return _TRADUCCION.get(tipo, err["msg"])


def _validar_filas(df: pd.DataFrame) -> list[tuple[int, Registro]]:
    """Valida TODAS las filas. Si hay algún error, no se predice nada y se informa fila y campo."""
    validos, errores = [], []
    for fila, rec in zip(df.index, df.to_dict("records")):
        try:
            validos.append((int(fila), Registro.model_validate(rec)))
        except ValidationError as e:
            for err in e.errors():
                campo = err["loc"][0] if err["loc"] else "(fila completa)"
                errores.append({"fila": int(fila), "campo": campo, "error": _describir(err)})
    if errores:
        raise _error(
            f"El archivo tiene errores en {len({e['fila'] for e in errores})} fila(s). "
            "No se realizó ninguna predicción (validación estricta).",
            total_errores=len(errores),
            errores=errores[:MAX_ERRORES_MOSTRADOS])
    return validos


def _a_dataframe(registros: list[Registro]) -> pd.DataFrame:
    """Arma el DataFrame con los mismos tipos que usó el entrenamiento (ver model_deploy.py)."""
    X = pd.DataFrame([r.model_dump() for r in registros], columns=COLUMNAS_ENTRADA)
    X["tiene_mora"] = X["tiene_mora"].map({True: 1.0, False: 0.0})        # vacío -> NaN
    X[BOOLEANAS] = X[BOOLEANAS].astype(int)
    numericas = [c for c in X.columns if c not in CATEGORICAS]
    X[numericas] = X[numericas].apply(pd.to_numeric, errors="coerce")
    X["tendencia_ingresos"] = X["tendencia_ingresos"].astype(object).where(
        X["tendencia_ingresos"].notna(), np.nan)                           # vacío -> NaN
    X["tipo_credito"] = X["tipo_credito"].astype("object")                # igual que en el entrenamiento
    return X


def predecir(registros: list[tuple[int, Registro]]) -> list[Resultado]:
    X = _a_dataframe([r for _, r in registros])
    X_feat = estado["preprocesador"].transform(X)                          # ratios + imputación + codificación + escalado
    proba = estado["modelo"].predict_proba(X_feat)[:, estado["idx_clase_1"]]
    clases = (proba >= UMBRAL).astype(int)
    return [Resultado(fila=fila, prediccion=int(c), probabilidad_pago_atiempo=round(float(p), 4),
                      mensaje=MENSAJES[int(c)])
            for (fila, _), p, c in zip(registros, proba, clases)]


# =============================================================================
# ENDPOINTS
# =============================================================================
@app.get("/health", tags=["Estado"])
def health():
    """Para monitoreo / healthcheck del contenedor."""
    return {"status": "ok", "modelo_cargado": "modelo" in estado}


@app.post("/predict", response_model=RespuestaBatch, tags=["Predicción"])
def predict(archivo: UploadFile = File(..., description="CSV con un cliente por fila")):
    """
    Predicción por lotes: recibe un CSV (un cliente por fila) y devuelve una predicción por fila.

    - Columnas obligatorias y orden sugerido: ver `ejemplo_entrada.csv` y el README.
    - Los campos del buró (puntaje, saldo mora, promedio de ingresos, tendencia, tiene_mora) pueden ir vacíos.
    - Validación estricta: ante cualquier error se responde 422 indicando fila y campo.
    """
    df = _leer_csv(archivo.file.read())
    _validar_columnas(df)
    registros = _validar_filas(df)
    resultados = predecir(registros)
    return RespuestaBatch(n_registros=len(resultados), umbral=UMBRAL, resultados=resultados)

"""
schemas.py: contrato de entrada de la API (una fila del CSV = un cliente).
---------------------------------------------------------------------------
NOTA - VALIDACIÓN ESTRICTA:
  De momento el modelo entrenado cuenta con pocos datos (~10.800 créditos y solo ~510 casos
  de clase 0), por lo que no es prudente hacer algo más flexible. Por eso la API NO completa
  campos faltantes, NO ignora columnas desconocidas y NO predice "lo que se pueda": si hay un
  solo error en el archivo, se rechaza todo con un 422 que indica fila y campo.
  Única excepción: los campos del buró (ver OPCIONALES) pueden venir vacíos, porque un
  cliente sin historial en el buró es un caso real (en el entrenamiento también había nulos).

Los ratios (cuota_sobre_salario, huella_por_credito) NO se envían: el pipeline los calcula a
partir de `salario_cliente` y `cant_creditosvigentes`.
"""

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, BeforeValidator, Field, model_validator


# --- Conversores previos: el CSV llega como texto -------------------------------------------
_VERDADERO = {"1", "1.0", "true", "t", "si", "sí", "s", "yes", "y"}
_FALSO = {"0", "0.0", "false", "f", "no", "n"}


def _vacio_a_none(v):
    """'' (celda vacía) -> None."""
    if isinstance(v, str):
        v = v.strip()
        return None if v == "" else v
    return v


def _a_entero(v):
    """'4' o '4.0' -> 4. Si tiene decimales reales lo deja como está para que falle la validación."""
    v = _vacio_a_none(v)
    if isinstance(v, str):
        try:
            f = float(v)
        except ValueError:
            return v
        return int(f) if f.is_integer() else v
    return v


def _a_bool(v):
    v = _vacio_a_none(v)
    if isinstance(v, str):
        k = v.lower()
        if k in _VERDADERO:
            return True
        if k in _FALSO:
            return False
    return v


# --- Tipos reutilizables (fábricas: las restricciones van sobre el tipo interno) ---------------
def entero(**restricciones):
    return Annotated[int, BeforeValidator(_a_entero), Field(**restricciones)]


def decimal(**restricciones):
    return Annotated[float, BeforeValidator(_vacio_a_none), Field(allow_inf_nan=False, **restricciones)]


def decimal_opcional(**restricciones):
    """Acepta celda vacía (None); si hay valor, aplica las restricciones."""
    return Annotated[Optional[Annotated[float, Field(allow_inf_nan=False, **restricciones)]],
                     BeforeValidator(_vacio_a_none)]


Booleano = Annotated[bool, BeforeValidator(_a_bool)]
BooleanoOpcional = Annotated[Optional[bool], BeforeValidator(_a_bool)]


class Registro(BaseModel):
    """Un cliente. El orden de los campos define el orden esperado de las columnas del CSV."""

    # --- Crédito ---
    tipo_credito: entero(ge=0)              # código numérico del producto
    capital_prestado: decimal(gt=0)
    plazo_meses: entero(gt=0)
    cuota_pactada: decimal(gt=0)

    # --- Cliente ---
    edad_cliente: entero(ge=18, le=100)
    tipo_laboral: Literal["Empleado", "Independiente"]
    salario_cliente: decimal(gt=0)          # insumo del ratio cuota_sobre_salario
    cant_creditosvigentes: entero(ge=0)     # insumo del ratio huella_por_credito
    total_otros_prestamos: decimal(ge=0)

    # --- Buró (pueden venir vacíos) ---
    puntaje_datacredito: decimal_opcional(ge=0, le=999)
    saldo_mora_codeudor: decimal_opcional(ge=0)
    promedio_ingresos_datacredito: decimal_opcional(ge=0)
    tendencia_ingresos: Annotated[Optional[Literal["Decreciente", "Estable", "Creciente"]],
                                  BeforeValidator(_vacio_a_none)]
    tiene_mora: BooleanoOpcional

    # --- Buró (obligatorios) ---
    huella_consulta: entero(ge=0)
    creditos_sectorReal: entero(ge=0)
    total_creditos_sectores: decimal(ge=0)
    sin_historial_ingresos_buro: Booleano
    sin_score_buro: Booleano
    sin_saldos_buro: Booleano

    # --- Variables derivadas del desembolso (se validan contra sus fuentes, ver abajo) ---
    plazo_largo: Booleano
    dia_semana: entero(ge=0, le=6)          # 0 = lunes ... 6 = domingo
    hora_desembolso: decimal(ge=0, lt=24)
    es_fin_de_semana: Booleano

    @model_validator(mode="after")
    def _coherencia(self):
        # En los datos de entrenamiento estas relaciones se cumplen en el 100% de las filas,
        # así que una incoherencia es casi seguro un error de carga.
        if self.plazo_largo != (self.plazo_meses > 12):
            raise ValueError("plazo_largo debe ser True si plazo_meses > 12 y False en caso contrario")
        if self.es_fin_de_semana != (self.dia_semana >= 5):
            raise ValueError("es_fin_de_semana debe ser True si dia_semana es 5 o 6 (sábado/domingo)")
        if self.sin_score_buro != (self.puntaje_datacredito is None):
            raise ValueError("sin_score_buro debe ser True si y solo si puntaje_datacredito está vacío")
        return self


COLUMNAS_ENTRADA = list(Registro.model_fields)
BOOLEANAS = ["sin_historial_ingresos_buro", "sin_score_buro", "sin_saldos_buro",
             "plazo_largo", "es_fin_de_semana"]
CATEGORICAS = ["tipo_credito", "tipo_laboral", "tendencia_ingresos"]
RATIOS = ["cuota_sobre_salario", "huella_por_credito"]

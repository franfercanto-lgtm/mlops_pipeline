# API de riesgo crediticio (FastAPI + Docker)

Predice por lote (CSV) si un cliente pagará a tiempo su crédito.

## Estructura

```
├── model_deploy.py          # ENTRENA y genera los 2 joblibs (necesita Base_de_Datos_Final.csv al lado)
├── custom_transformers.py   # clase CalculaRatios (compartida por el entrenamiento y la API)
├── schemas.py               # contrato de entrada: columnas, tipos y reglas de validación
├── main.py                  # API: carga los joblibs, valida, predice (/health y /predict)
├── artifacts/
│   ├── preprocesador.joblib # ratios + imputación + codificación + escalado
│   └── modelo.joblib        # regresión logística balanceada
├── ejemplo_entrada.csv      # 12 clientes de ejemplo, listo para probar
├── requirements.txt · Dockerfile · .dockerignore
```

## Paso a paso

```bash
# 1) (solo si hay que reentrenar) con Python 3.11.9 y requirements.txt instalado
python model_deploy.py                      # regenera artifacts/*.joblib

# 2) construir la imagen (los joblibs ya están en artifacts/)
docker build -t riesgo-crediticio-api .

# 3) levantar el contenedor
docker run -d --name riesgo-api -p 8000:8000 riesgo-crediticio-api

# 4) probar
curl http://localhost:8000/health
curl -F "archivo=@ejemplo_entrada.csv" http://localhost:8000/predict
```
Documentación interactiva (permite subir el CSV desde el navegador): http://localhost:8000/docs

Sin Docker: `pip install -r requirements.txt && uvicorn main:app --reload`.

## Endpoints

| Método | Ruta       | Descripción |
|--------|------------|-------------|
| GET    | `/health`  | Estado del servicio y del modelo |
| POST   | `/predict` | Recibe un CSV (campo de formulario `archivo`) y devuelve una predicción por fila |

Respuesta de `/predict`:

```json
{
  "n_registros": 2,
  "umbral": 0.5,
  "resultados": [
    {"fila": 2, "prediccion": 1, "probabilidad_pago_atiempo": 0.599,
     "mensaje": "El cliente tiene una alta probabilidad de pagar a tiempo"},
    {"fila": 3, "prediccion": 0, "probabilidad_pago_atiempo": 0.3865,
     "mensaje": "El cliente posee un perfil de ALTO RIESGO crediticio"}
  ]
}
```
`fila` es el número de fila **como se ve en Excel** (la 1 es el encabezado).

## Formato del CSV

- Una fila por cliente. Separador `,` o `;` (Excel en español). Decimales con **punto**. UTF-8.
- Deben estar **exactamente** estas columnas (el orden no importa). No se envían los ratios ni el target.

| Grupo | Columnas |
|---|---|
| Crédito | `tipo_credito` (entero), `capital_prestado` (>0), `plazo_meses` (>0), `cuota_pactada` (>0) |
| Cliente | `edad_cliente` (18-100), `tipo_laboral` (`Empleado` / `Independiente`), `salario_cliente` (>0), `cant_creditosvigentes` (≥0), `total_otros_prestamos` (≥0) |
| Buró **(pueden ir vacías)** | `puntaje_datacredito` (0-999), `saldo_mora_codeudor` (≥0), `promedio_ingresos_datacredito` (≥0), `tendencia_ingresos` (`Decreciente` / `Estable` / `Creciente`), `tiene_mora` (True/False) |
| Buró (obligatorias) | `huella_consulta` (≥0), `creditos_sectorReal` (≥0), `total_creditos_sectores` (≥0), `sin_historial_ingresos_buro`, `sin_score_buro`, `sin_saldos_buro` (True/False) |
| Desembolso | `plazo_largo` (True/False), `dia_semana` (0=lunes … 6=domingo), `hora_desembolso` (0-23), `es_fin_de_semana` (True/False) |

Reglas de coherencia (se cumplen en el 100% de los datos de entrenamiento): `plazo_largo` = `plazo_meses > 12`;
`es_fin_de_semana` = `dia_semana` ≥ 5; `sin_score_buro` = True si y solo si `puntaje_datacredito` está vacío.

Los ratios se calculan solos: `cuota_sobre_salario = cuota_pactada / salario_cliente` y
`huella_por_credito = huella_consulta / (cant_creditosvigentes + 1)`.

## Validación estricta

De momento el modelo entrenado tiene pocos datos (~510 casos de clase 0) para hacer algo más flexible: no se
completan faltantes ni se ignoran columnas. Ante **cualquier** error se responde `422` y no se predice nada:

```json
{"detail": {"mensaje": "El archivo tiene errores en 2 fila(s). No se realizó ninguna predicción (validación estricta).",
            "total_errores": 2,
            "errores": [{"fila": 2, "campo": "salario_cliente", "error": "debe ser mayor a 0"},
                        {"fila": 3, "campo": "tipo_laboral", "error": "valor no permitido (opciones: 'Empleado' or 'Independiente')"}]}}
```

## Configuración (variables de entorno)

| Variable | Default | Uso |
|---|---|---|
| `UMBRAL` | `0.5` | P(paga a tiempo) ≥ umbral → clase 1. Ej.: `docker run -e UMBRAL=0.6 ...` |

## Notas

- **Versiones**: Python 3.11.9, scikit-learn 1.4.2, feature-engine 1.6.2 (fijadas). Si la versión de scikit-learn del
  entorno no coincide con la del entrenamiento, la API **no arranca** (a propósito: evita predicciones silenciosamente
  incorrectas). `pandas<3` y `numpy<2` son necesarios por compatibilidad con esas versiones.
- La imagen **no** incluye el CSV ni `model_deploy.py`: solo el código de la API y los 2 joblibs.
- Si se reentrena, hay que volver a construir la imagen (`docker build`).

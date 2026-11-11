"""
Objetivo: crear la base final para el modelo, dejando solo las variables que aportan
a la predicción (según los coeficientes y odds ratio de la Regresión Logística, ver
model_training_evaluation.py) y sin la fecha, para reducir el tamaño del dataset.
Resultado: "Base_de_Datos_Final.csv".
"""

from pathlib import Path

import pandas as pd

CARPETA = Path(__file__).resolve().parent   # carpeta donde está este archivo

# Columnas a eliminar
DESCARTAR = [
    "fecha_prestamo",                                          # fecha: no se usa como variable
    # Peso casi nulo (|coef| < 0,05)
    "saldo_total_sobre_capital", "creditos_sectorCooperativo", "cant_creditosvigentes",
    "pct_sector_financiero", "saldo_total", "saldo_mora", "saldo_principal",
    "salario_cliente", "creditos_sectorFinanciero",
    # Ratios que son ruido (su odds ratio puede ser 1: el intervalo de confianza al 95 % lo incluye)
    "capital_sobre_salario", "otros_prestamos_sobre_salario", "cuota_sobre_ingreso_buro",
    "ratio_salario_vs_buro", "ratio_mora_sobre_saldo", "intensidad_busqueda",
]

# Leer el CSV original, quitar esas columnas y guardar el CSV final (sin índice)
(pd.read_csv(CARPETA / "Base_de_datos_limpia.csv")
   .drop(columns=DESCARTAR)
   .to_csv(CARPETA / "Base_de_Datos_Final.csv", index=False))

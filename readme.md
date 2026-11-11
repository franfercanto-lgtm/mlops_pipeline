Proyecto Integrador de Riesgo Crediticio
Objetivo

El objetivo de este trabajo es desarrollar un modelo de Machine Learning supervisado capaz de anticipar el comportamiento crediticio de nuevos usuarios a partir de datos históricos.

Con este modelo se busca apoyar la toma de decisiones en el otorgamiento de créditos, identificando de forma temprana a los solicitantes con mayor probabilidad de incumplimiento.

## Versionado en Git

Para garantizar el orden, la trazabilidad y la reproducibilidad del proyecto, se utilizó **Git** como sistema de control de versiones, siguiendo una estrategia de ramas que separa el desarrollo, la validación y las versiones estables.

### Avance v1.0.0: "Estructura del repositorio"

Se generó la siguiente estructura de carpetas, la cual se mantiene sin cambios a lo largo de todo el proyecto:

```
.
├── .gitignore
├── Base_de_datos.csv
├── mlops_pipeline/
│   └── src/
├── readme.md
└── requirements.txt
```

| Elemento | Descripción |
|---|---|
| `.gitignore` | Define los archivos y carpetas que Git debe ignorar (por ejemplo, el entorno virtual). |
| `Base_de_datos.csv` | Conjunto de datos históricos utilizado para entrenar el modelo. |
| `mlops_pipeline/src/` | Código fuente del pipeline de Machine Learning. |
| `readme.md` | Documentación del proyecto. |
| `requirements.txt` | Listado de dependencias necesarias para ejecutar el proyecto. |

### Configuración del entorno

1. **Clonado del repositorio:** se clonó localmente el repositorio creado previamente para trabajar con él.

   ```bash
   git clone <url-del-repositorio>
   cd <nombre-del-repositorio>
   ```

2. **Archivo `requirements.txt`:** se utilizó el archivo provisto por Henry, que contiene las librerías necesarias para el desarrollo del proyecto.

3. **Entorno virtual:** se configuró un entorno virtual con **Python 3.11.9**, versión requerida por compatibilidad con algunas de las librerías que se utilizan en etapas posteriores.

   ```bash
   python -m venv venv
   source venv/bin/activate        # En Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Estrategia de ramas

Se adoptó un flujo de trabajo con tres ramas, cada una con un propósito definido:

| Rama | Propósito |
|---|---|
| `main` | Rama principal. Contiene las versiones estables y aprobadas del proyecto. |
| `developer` | Rama de desarrollo. Se utiliza para nuevas funcionalidades, modificaciones y mejoras. Los cambios se integran en `main` una vez finalizados. |
| `certification` | Rama de validación. Se utiliza para evaluar las versiones antes de su incorporación definitiva a `main`. |

```bash
git checkout -b developer
git checkout -b certification
```

### Flujo de trabajo

Las modificaciones se realizaron siguiendo este recorrido:

```
developer  →  certification  →  main
(desarrollo)   (validación)     (versión estable)
```

1. Los cambios se desarrollan y se registran mediante *commits* en `developer`.
2. Una vez finalizados, se integran en `certification`, donde se validan.
3. Si la validación es satisfactoria, se incorporan a `main`.

De esta forma, `main` conserva únicamente versiones revisadas y funcionales del proyecto.

## Archivo .gitignore

En este paso se completó el archivo `.gitignore`, que indica a Git qué archivos y carpetas no deben incluirse en el control de versiones. De esta forma se evita subir al repositorio archivos temporales, datos sensibles o elementos que cada usuario puede regenerar localmente.

Se agregaron las siguientes entradas:

```ignore
.venv/
__pycache__/
.ipynb_checkpoints/
.env
Base_de_datos.xlsx
set_up.bat
```

| Entrada | Motivo |
|---|---|
| `.venv/` | Entorno virtual local. Cada usuario lo genera a partir de `requirements.txt`. |
| `__pycache__/` | Archivos temporales que genera Python al ejecutar el código. |
| `.ipynb_checkpoints/` | Puntos de guardado automáticos de Jupyter Notebook. |
| `.env` | Variables de entorno y credenciales, que nunca deben publicarse. |
| `Base_de_datos.xlsx` | Archivo de datos original. Se conserva únicamente la versión `.csv` en el repositorio. |
| `set_up.bat` | Script de configuración de uso local. |

### Archivo readme.md

En este mismo paso se creó el archivo `readme.md`, aunque su redacción se realizó en etapas posteriores del proyecto.

## Archivo Auxiliar_Zero.ipynb

Este archivo no formaba parte del entregable solicitado. Dado que la consigna permite crear archivos adicionales, se incorporó este notebook de Jupyter con un único propósito: convertir la base de datos original, que se encuentra en formato `.xlsx`, al formato `.csv` requerido para la entrega.

Si bien esta conversión podría realizarse directamente desde Excel, se optó por hacerla en Jupyter para dejar constancia del paso y garantizar su reproducibilidad.

| Elemento | Detalle |
|---|---|
| **Archivo** | `Auxiliar_Zero.ipynb` |
| **Entrada** | Base de datos original (`.xlsx`) |
| **Salida** | `Base_de_datos.csv` |
| **Función** | Conversión de formato, sin transformaciones sobre los datos |

> **Nota:** este notebook se agrega al `.gitignore` en pasos posteriores, ya que cumple una función puntual y no forma parte del pipeline del proyecto.


## Avance V1.1.1: Versionamiento y Colaboración 

### Versionado de esta etapa

El desarrollo de los siguientes notebooks se realizó en la rama `developer`. Para integrar estos cambios a la rama `main`, se solicitó la aprobación de un compañero antes de hacer el *merge*, de modo que `main` conserve únicamente versiones revisadas.

## Archivo Comprension_eda.ipynb

En este notebook se realiza el análisis exploratorio de datos (EDA) de la base de créditos. El objetivo es entender qué información contiene, detectar problemas de calidad, decidir cómo tratarlos y evaluar qué variables podrían aportar al modelo.

La **variable objetivo** es `Pago_atiempo` (1 = pagó a tiempo, 0 = no pagó).

Los datos se obtienen importando la variable `data` del notebook `Cargar_datos` y almacenándola en una nueva variable llamada `raw`. Esta base original **no se modifica**: todo el trabajo posterior se hace sobre copias.

**Librerías adicionales.** Para ejecutar este notebook se agregaron las librerías `import_ipynb` y `scipy` al archivo `requirements.txt` y se instalaron en el entorno virtual. La primera permite importar un notebook desde otro (en este caso, `Cargar_datos`) y la segunda se utiliza en las pruebas estadísticas del análisis.

Para asegurar la reproducibilidad, la variable objetivo (`TARGET`) y la semilla aleatoria (`SEED = 42`) se definen como constantes al inicio del notebook.

### 1. Exploración inicial

- La base tiene **10.763 registros y 23 columnas**.
- Hay valores nulos en **7 columnas**. Las más afectadas son `promedio_ingresos_datacredito` y `tendencia_ingresos` (~27 %). Los saldos tienen entre 1,4 % y 5,5 %, y `puntaje_datacredito` apenas 0,06 %.
- Las variables se clasificaron por su **tipo estadístico** y no por el tipo que les asigna pandas. Por ejemplo, `tipo_credito` llega como número entero pero es un código (el 68 no es "mayor" que el 4), y `fecha_prestamo` llega como texto pero es una fecha.

| Tipo | Variables |
|---|---|
| Objetivo (dicotómica) | `Pago_atiempo` |
| Categóricas nominales | `tipo_laboral`, `tipo_credito` |
| Categórica ordinal | `tendencia_ingresos` (Decreciente < Estable < Creciente) |
| Fecha | `fecha_prestamo` |
| Numéricas continuas | `capital_prestado`, `cuota_pactada`, `salario_cliente`, `total_otros_prestamos`, `promedio_ingresos_datacredito`, `saldo_mora`, `saldo_total`, `saldo_principal`, `saldo_mora_codeudor`, `puntaje`, `puntaje_datacredito` |
| Numéricas discretas | `plazo_meses`, `edad_cliente`, `cant_creditosvigentes`, `huella_consulta`, `creditos_sectorFinanciero`, `creditos_sectorCooperativo`, `creditos_sectorReal` |

Además se analizó si los nulos ocurren juntos y si se relacionan con el resultado. Los nulos de `promedio_ingresos_datacredito` y `tendencia_ingresos` aparecen casi siempre en los mismos registros, probablemente clientes sin historial de ingresos en el buró. Quienes tienen datos faltantes pagan algo menos (≈ 93–94 % frente a ≈ 95–96 %), por lo que **el nulo es en sí mismo informativo**.

### 2. Limpieza y preparación

**Valores imposibles o sospechosos.** Además de los nulos explícitos, se buscaron valores que no pueden ser reales, con umbrales definidos por plausibilidad de negocio:

| Variable | Problema detectado | Registros | Tratamiento |
|---|---|---|---|
| `tendencia_ingresos` | Números en una columna que solo debería tener 3 categorías | 58 | Se convierten a nulo |
| `edad_cliente` | Edades de 121 a 123 años (valor centinela) | 150 | Fuera del rango [18, 90] → nulo |
| `salario_cliente` | Ceros (dato no informado) | 24 | Se convierten a nulo |
| `salario_cliente` | Valores mayores a 100 millones (errores de escala o digitación) | 74 | Se convierten a nulo |
| `puntaje_datacredito` | Valores menores o iguales a 0 (sin calificación) | 146 | Se convierten a nulo |
| `total_otros_prestamos` | Valores mayores a 100 millones | 44 | Solo se señalan, no se modifican (pendiente de decisión) |

Los salarios entre 50 y 100 millones (23 casos) se conservaron por ser altos pero plausibles.

**Unificación de nulos.** Todos los valores faltantes se representan con un único estándar (`NaN`). Cada regla aplicada queda registrada con su cantidad de filas afectadas en una bitácora, para poder auditarla y replicarla.

**Indicadores de ausencia.** Convertir un dato a `NaN` borra la señal de que faltaba, y esa señal tiene efecto sobre el pago. Por eso, antes de imputar, se crearon tres indicadores: `sin_historial_ingresos_buro`, `sin_score_buro` y `sin_saldos_buro`.

**Imputación.** La decisión se tomó variable por variable con una sola pregunta: ¿el valor existe en la realidad pero no se capturó, o el nulo significa algo (por ejemplo, "sin historial")? Solo en el primer caso tiene sentido estimarlo.

| Variable | Significado del nulo | Decisión |
|---|---|---|
| `edad_cliente` | Error de captura | **Se imputa** con la media del grupo |
| `salario_cliente` | Dato no informado o error de escala | **Se imputa** con la mediana del grupo (por sus valores extremos) |
| `promedio_ingresos_datacredito`, `tendencia_ingresos`, `puntaje_datacredito` | El cliente no tiene historial o calificación | **No se imputa**: se inventaría un dato que nadie reportó |
| Saldos (`saldo_total`, `saldo_principal`, `saldo_mora`, `saldo_mora_codeudor`) | Por definir: "sin deuda" o "no reportado" | **Pendiente de negocio**: se dejan en `NaN` |

Los grupos para imputar combinan `tipo_credito` (agrupado en 4, 9, 10 y "Otros"), `tipo_laboral` y `tendencia_ingresos`, considerando el nulo de tendencia como un grupo más. Los códigos de crédito 6, 7 y 68 suman solo 24 registros y, como grupos propios, darían estimaciones inestables. Si un grupo no tiene datos, se usa el estadístico de toda la base.

**Tipos de dato.** Se convirtieron a su tipo correcto: `fecha_prestamo` a fecha, `tipo_credito` y `tipo_laboral` a categóricas, `tendencia_ingresos` a categórica ordenada, `edad_cliente` a entero que admite nulos, los conteos a enteros y los indicadores `sin_*` a booleanos. `Pago_atiempo` se mantuvo como 0/1 para que su promedio sea directamente la tasa de pago. Se agregaron verificaciones (`assert`) que hacen fallar el notebook si alguna conversión sale mal, en lugar de continuar con datos corruptos.

### 3. Análisis univariable

- **Montos muy asimétricos:** la media supera a la mediana y el máximo está muy lejos del percentil 99. El caso más extremo es `total_otros_prestamos` (mediana de 1 millón, máximo de 6.788 millones).
- **Distribuciones:** `capital_prestado`, `cuota_pactada` y `salario_cliente` son log-normales, por lo que `log1p` funciona bien. `total_otros_prestamos`, `promedio_ingresos_datacredito` y los saldos tienen colas aún más pesadas y ceros. `puntaje_datacredito` es casi gaussiana. `edad_cliente` es la más cercana a una distribución normal (entre 19 y 69 años, media ≈ 43).
- **`saldo_mora`:** cerca del 99,5 % de los valores informados son 0, por lo que conviene separarla en un indicador de mora y un monto.
- **`plazo_meses`:** toma pocos valores propios de los productos (6, 12 y 10 meses concentran la mayoría).
- **Variable objetivo muy desbalanceada:** 95,25 % pagó a tiempo y 4,75 % no (511 casos). Por eso usar *accuracy* no sería adecuado: se priorizan métricas como AUC-PR, KS o recall de la clase 0, y particiones estratificadas.
- **Categóricas:** `tipo_credito` está dominado por los códigos 4 (72 %) y 9 (27 %); `tipo_laboral` se reparte en 63 % empleados y 37 % independientes; `tendencia_ingresos` tiene 27,8 % de nulos.
- **Distribución temporal:** los desembolsos caen de ≈ 1.900 créditos por mes (enero 2025) a menos de 100 por mes desde febrero 2026. No parece una caída real del negocio sino un **efecto de corte de la muestra**, con posible sesgo de madurez (los créditos recientes aún no tuvieron tiempo de incumplir). Esto motiva validar con una partición temporal.

### 4. Análisis bivariable (frente a `Pago_atiempo`)

**Variables numéricas.** Se compararon pagadores y no pagadores con la prueba de Mann-Whitney (robusta a valores extremos) y con un AUC univariable. Como con ~10 mil filas casi todo resulta "significativo", se priorizó el tamaño del efecto sobre el p-valor. Sin `puntaje`, **ninguna variable discrimina fuerte por sí sola**: el riesgo es difuso y multifactorial.

| Variable | Poder (\|AUC − 0,5\| × 2) | Relación con el pago |
|---|---|---|
| `puntaje_datacredito` | 0,24 | Mejor score, más pago |
| `huella_consulta` | 0,20 | Más consultas, menos pago |
| `edad_cliente` | 0,15 | Más edad, más pago |
| `promedio_ingresos_datacredito` | 0,145 | Más ingreso en el buró, más pago |
| `capital_prestado` | 0,11 | Mayor monto, menos pago |

El `salario_cliente` casi no discrimina (0,01), mientras que el ingreso del buró sí: **el ingreso declarado es poco confiable**.

**Tasa de pago por tramos.** Revela relaciones no lineales. El plazo muestra un **efecto umbral**: hasta 12 meses la tasa de pago es 95–96,8 %, baja a 93,2 % entre 13 y 24 meses y a 87,4 % por encima de 24. Los clientes más jóvenes (hasta 33 años) pagan peor (≈ 92–93 % frente a 95–97 % desde los 38).

**Variables categóricas.** Se usaron chi² y V de Cramér. Las asociaciones son significativas pero pequeñas (V ≤ 0,07). Empleados pagan 95,7 % frente a 94,5 % de los independientes. En `tendencia_ingresos`, Creciente (96,1 %) > Estable (95,4 %) > Decreciente (93,7 %). El grupo "Otros" de `tipo_credito` tiene la menor tasa (62,5 %), pero con solo 24 casos, por lo que no se sobreinterpreta.

**⚠ Fuga de información en `puntaje`.** Esta variable tiene una correlación de 0,92 con el objetivo. Tres pruebas indican que se calcula a partir del resultado y no antes:

1. **Separación perfecta:** el máximo de los no pagadores (62,7) es menor que el mínimo de los pagadores (63,8), y el AUC es de 1,00, algo imposible en un caso real de riesgo.
2. **Valores repetidos:** entre los impagos aparece 47,6139, que es exactamente la mitad de 95,2278, el valor máximo.
3. **Distribución:** más del 87 % de los registros tienen el valor tope (95,2278).

**Decisión:** `puntaje` se eliminó de la base de trabajo para que ninguna etapa posterior la use por accidente. Se conserva en `raw` por si el área de negocio confirma que existe antes del desembolso.

**Nulos del buró y mora.** Tener mora vigente (`saldo_mora > 0`) es la señal binaria más fuerte (≈ 64 % de pago frente a ≈ 95 %), aunque cubre solo 55 créditos. No tener historial de ingresos o score se asocia con 1 a 3 puntos menos de pago, lo que justifica conservar los indicadores `sin_*`.

**Evolución temporal.** La tasa de pago sube de ≈ 94 % (4T-2024) a ≈ 97,5 % (4T-2025) y baja en 2026 (con muestras pequeñas). Puede reflejar mejor originación o el sesgo de madurez mencionado antes, por lo que se confirma la necesidad de una validación temporal.

### 5. Análisis multivariable

- **Correlación de Spearman** (por las colas pesadas). Hay pares de predictores muy correlacionados: `saldo_total` con `saldo_principal` (0,95), `cant_creditosvigentes` con `creditos_sectorFinanciero` (0,76), `capital_prestado` con `cuota_pactada` (0,75) y `salario_cliente` con `total_otros_prestamos` (0,69). Esto importa para modelos lineales, no tanto para árboles.
- **Correlación con el objetivo:** todas son bajas (|ρ| ≤ 0,11). No hay una variable dominante, salvo la fuga ya eliminada.
- **Pairplot y gráficos por categoría:** los no pagadores no forman un grupo claro en la mayoría de los pares de variables. Los independientes tienen cuotas más altas para el mismo capital y mucha menos información en el buró (41,5 % sin dato de tendencia frente a 19,7 % de los empleados).
- **Efectos acumulativos:** el mejor segmento (plazo de 7–12 meses y 0–2 consultas) paga 97,6 %; el peor (plazo mayor a 12 meses y más de 5 consultas) paga 87,0 %. Los efectos se suman, por lo que conviene usar interacciones o modelos basados en árboles.
- **Consistencia interna:** `saldo_total ≥ saldo_principal` y `saldo_mora ≤ saldo_total` se cumplen siempre. La suma de créditos por sector es casi siempre menor o igual a los créditos vigentes (99,96 %), pero no igual. Además, `cuota × plazo` es menor que el capital en el 91 % de los registros, lo que no es esperable con intereses.

### 6. Transformaciones sugeridas y atributos derivados

**Transformaciones recomendadas:**

| Variable(s) | Problema | Transformación sugerida |
|---|---|---|
| `capital_prestado`, `cuota_pactada`, `salario_cliente` | Log-normales | `log1p` (en modelos lineales o redes) |
| `total_otros_prestamos`, `promedio_ingresos_datacredito`, saldos | Colas muy pesadas | `log1p` y winsorización, ajustada solo con datos de entrenamiento |
| `saldo_mora` | ~99,5 % de ceros | Indicador `tiene_mora` más monto en log |
| `tipo_credito` | Tres niveles casi vacíos | Agrupar 6, 7 y 68 como "Otros"; *one-hot* o *target encoding* |
| `tendencia_ingresos` | Ordinal con 27,8 % de nulos | Categoría "Sin dato" o indicador de ausencia |
| `plazo_meses` | Efecto umbral | Numérica y/o tramos (≤ 6, 7–12, > 12) |
| Variables con nulos | Nulos informativos | Árboles: dejar el `NaN`. Modelos lineales: imputar dentro del pipeline y solo con entrenamiento |
| Variable objetivo | Desbalance 95 / 5 | Partición estratificada y temporal; ponderación de clases o SMOTE solo en entrenamiento |

**Atributos derivados.** Se construyeron 17 variables nuevas en cuatro grupos, y se midió el poder de cada una con el mismo AUC univariable:

| Grupo | Ejemplos |
|---|---|
| Capacidad de pago y endeudamiento | `cuota_sobre_salario`, `cuota_sobre_ingreso_buro`, `ratio_salario_vs_buro`, `otros_prestamos_sobre_salario` |
| Estructura de la deuda | `ratio_mora_sobre_saldo`, `saldo_total_sobre_capital`, `tiene_mora`, `pct_sector_financiero` |
| Comportamiento crediticio | `huella_por_credito`, `intensidad_busqueda`, `plazo_largo` |
| Temporales | `dia_semana`, `hora_desembolso`, `es_fin_de_semana`, `dias_desde_inicio` |

Resultados principales:

- Ninguna derivada supera a la mejor variable original (`puntaje_datacredito`, 0,24), pero `huella_por_credito` (0,21) se acerca.
- `ratio_salario_vs_buro` (0,14) y `cuota_sobre_ingreso_buro` (0,12) son informativas: la tasa de pago baja de ≈ 97 % a ≈ 94 % cuando el salario declarado supera ampliamente al ingreso del buró, lo que sugiere sobredeclaración o ingresos informales.
- `dias_desde_inicio` (0,10) refleja la deriva temporal: sirve para diagnóstico pero **no se recomienda como predictora**.
- `es_fin_de_semana` (0,10) muestra menor pago (93,7 % frente a 95,9 %), pero **no es causal**: puede ser un proxy del canal o del producto, por lo que se debe investigar antes de usarla.

### 7. Exportación del dataset limpio

Al final del notebook se exportó el archivo `Base_de_datos_limpia.csv`, en la misma carpeta donde se encuentra este notebook. Contiene los datos ya depurados, **sin `puntaje`** y **con los atributos derivados**. Solo se excluyeron las columnas auxiliares creadas para los gráficos.

## Avance V1.1.0: Ingeniería de Variables

En esta etapa se construyó el componente de ingeniería de variables (`ft_engineering.py`), que toma el dataset limpio generado en el EDA (`Base_de_datos_limpia.csv`) y lo deja listo para entrenar modelos. Todo el preprocesamiento está encapsulado en *pipelines*, de modo que se aplica siempre de la misma forma y es reproducible. La ruta del CSV se resuelve de forma relativa al propio script, por lo que el código funciona sin importar desde dónde se ejecute.

**Librerías adicionales.** Se agregó la librería `feature-engine` al archivo `requirements.txt` y se instaló en el entorno virtual. El componente utiliza además `pandas` y `scikit-learn`.

### Configuración

Los parámetros que definen el comportamiento del componente están agrupados al inicio del archivo, para poder modificarlos sin tocar la lógica:

| Parámetro | Valor | Descripción |
|---|---|---|
| `TARGET` | `Pago_atiempo` | Variable objetivo |
| `SEMILLA` | `42` | Semilla aleatoria, para que la partición sea reproducible |
| `TEST_SIZE` | `0.20` | Proporción de datos reservada para prueba |
| `COLS_EXCLUIR` | `fecha_prestamo`, `dias_desde_inicio` | Columnas que no entran como variables predictoras (ver explicación más abajo) |
| `COLS_CATEGORICAS` | `tipo_credito`, `tipo_laboral` | Variables nominales |
| `COLS_ORDINALES` | `tendencia_ingresos` | Variable ordinal, con orden Decreciente < Estable < Creciente |
| `TOL_RARAS` | `0.01` | Las categorías con menos del 1 % de los registros se agrupan como `Rare` |

**Columnas excluidas:**

- **`fecha_prestamo`:** es una fecha en formato texto, y su información temporal ya se extrajo en el EDA (por ejemplo, `dia_semana` y `hora_desembolso`).
- **`dias_desde_inicio`:** mide el paso del tiempo y no el riesgo del cliente. Un cliente nuevo siempre tendría una fecha posterior a la de entrenamiento, y los créditos recientes aún no tuvieron tiempo de incumplir (sesgo de madurez detectado en el EDA). El modelo aprendería que "fecha reciente significa menos impagos", una relación que no se sostendría con clientes nuevos.

### Preparación de los datos

Antes de construir los *pipelines*, la función `build_features()` realiza los siguientes pasos:

1. **Lectura del CSV** y eliminación de las columnas excluidas.
2. **Conversión de tipos:** las columnas booleanas (como los indicadores `sin_*`) pasan a 0/1, y `tipo_credito` se trata como texto, ya que es un código y no una cantidad.
3. **Separación entre variables predictoras y objetivo.**
4. **Partición entrenamiento/prueba 80/20, estratificada.** Como el objetivo está muy desbalanceado (≈ 95 % paga a tiempo), la estratificación asegura que ambas partes mantengan la misma proporción de impagos.
5. **Clasificación de las columnas** en categóricas, ordinales, numéricas y numéricas a escalar. Las variables binarias (0/1) no se escalan; el resto de las numéricas sí.

### Pipelines de preprocesamiento

El preprocesamiento se organiza en dos etapas encadenadas:

| Etapa | Paso | Herramienta | Variables | Qué hace |
|---|---|---|---|---|
| 1 | Imputación numérica | `MeanMedianImputer` (feature-engine) | Numéricas | Completa los nulos con la **mediana**, robusta ante los valores extremos detectados en el EDA |
| 1 | Imputación categórica | `CategoricalImputer` (feature-engine) | Categóricas y ordinal | Completa los nulos con la **categoría más frecuente** |
| 1 | Agrupación de raras | `RareLabelEncoder` (feature-engine) | Categóricas | Agrupa en `Rare` las categorías con menos del 1 % |
| 1 | Codificación nominal | `OneHotEncoder` (feature-engine) | Categóricas | Crea una columna 0/1 por cada categoría |
| 2 | Codificación ordinal | `OrdinalEncoder` (scikit-learn) | `tendencia_ingresos` | La convierte a 0/1/2 respetando el orden; un valor desconocido recibe `-1` |
| 2 | Escalado | `StandardScaler` (scikit-learn) | Numéricas no binarias | Las lleva a media 0 y desviación 1 |
| 2 | Resto | `passthrough` | Dummies y binarias | Se dejan sin modificar |

La salida de todo el *pipeline* es un `DataFrame` de pandas con los nombres de las columnas, lo que facilita inspeccionar y entender cada variable.

**Nulos que trata el pipeline.** El dataset limpio conserva a propósito los nulos cuyo significado es "sin historial" o "sin calificación" (ver EDA). El pipeline los completa dentro del ajuste, solo con estadísticas del conjunto de entrenamiento:

| Tipo | Variable | Nulos |
|---|---|---|
| Original | `tendencia_ingresos` | 2.990 |
| Original | `promedio_ingresos_datacredito` | 2.930 |
| Original | `puntaje_datacredito` | 152 |
| Original | `saldo_mora` | 156 |
| Original | `saldo_total` | 156 |
| Original | `saldo_principal` | 405 |
| Original | `saldo_mora_codeudor` | 590 |
| Derivada | `cuota_sobre_ingreso_buro` | 2.937 |
| Derivada | `ratio_salario_vs_buro` | 2.937 |
| Derivada | `ratio_mora_sobre_saldo` | 945 |
| Derivada | `pct_sector_financiero` | 704 |
| Derivada | `saldo_total_sobre_capital` | 156 |
| Derivada | `tiene_mora` | 156 |

**Decisiones principales:**

- **Agrupación de categorías raras:** coincide con lo observado en el EDA, donde los códigos de `tipo_credito` 6, 7 y 68 tenían muy pocos registros (21, 2 y 1). Tras la agrupación quedan los niveles 4, 9, 10 y `Rare`.
- **Imputación e indicadores de ausencia:** en el EDA los nulos informativos se conservaron en lugar de rellenarse. Como los modelos lineales y las redes no admiten nulos, el pipeline los completa solo con datos de entrenamiento. Los indicadores `sin_historial_ingresos_buro`, `sin_score_buro` y `sin_saldos_buro` atraviesan el pipeline sin modificarse, de modo que la señal de que el dato faltaba no se pierde.
- **Escalado selectivo:** solo se escalan las numéricas no binarias, para no alterar variables que ya están en 0/1.
- **Sin fuga de información:** el preprocesador se ajusta (`fit`) **únicamente con el conjunto de entrenamiento** y luego se aplica (`transform`) a entrenamiento y prueba. Así, ninguna estadística (medianas, categorías frecuentes, medias para el escalado) se calcula con datos de prueba.

### Uso

```python
from ft_engineering import build_features

X_train, X_test, y_train, y_test, preprocesador = build_features()
```

La función devuelve los conjuntos de entrenamiento y prueba ya transformados, las variables objetivo correspondientes y el preprocesador ajustado, que puede reutilizarse para transformar datos nuevos.

### Verificación

Al ejecutar el archivo directamente se obtiene:

```
Train: (8610, 43) | Test: (2153, 43)
Nulos restantes: 0
```

| Aspecto | Resultado |
|---|---|
| Registros de entrenamiento / prueba | 8.610 / 2.153 |
| Columnas finales | 43 (las dos categóricas nominales se expanden en 6 columnas: `tipo_credito_4`, `_9`, `_10`, `_Rare`, `tipo_laboral_Empleado`, `_Independiente`) |
| Impagos en entrenamiento / prueba | 409 / 102 (≈ 4,75 % en ambos, gracias a la estratificación) |
| Nulos restantes | 0 |

### Versionado de esta etapa

Esta etapa se desarrolló en la rama `certification`. Una vez comprobado que todo funcionaba correctamente, los cambios se integraron a la rama `main` mediante un *merge*.

## Avance 1.1.1: Evaluación de Modelos

En esta etapa se construyeron dos componentes. El primero, `model_training_evaluation.py`, toma los datos ya transformados por `ft_engineering.py` y se encarga de comparar algoritmos, elegir el modelo, ajustar sus hiperparámetros e interpretar sus resultados. El segundo, `model_final_database.py`, genera la base de datos final simplificada para entrenar el modelo. El objetivo es anticipar si un cliente pagará a tiempo (`Pago_atiempo`).

### Librerías y entorno

- **Librerías nuevas:** se agregaron al archivo `requirements.txt` las librerías necesarias para esta etapa: `pycaret` (versión 3.3 o superior), `matplotlib` y `six`. Esta última es requerida por PyCaret: sin ella falla al importarse.
- **Python 3.11.9:** se trabaja con esta versión específicamente para poder utilizar PyCaret, que solo es compatible con versiones de Python hasta la 3.11. Además, a algunas librerías se les especificó una versión exacta en `requirements.txt` para que sean compatibles con esta versión de Python y con PyCaret.

### Flujo del componente de evaluación

| Paso | Qué hace |
|---|---|
| 1 | Importa `build_features()` desde `ft_engineering.py` y obtiene los conjuntos de entrenamiento y prueba ya transformados |
| 2 | PyCaret compara decenas de algoritmos con validación cruzada y los ordena por AUC |
| 3 | Se entrenan los 3 mejores y se comparan con gráficos |
| 4 | Se ajusta la Regresión Logística con *GridSearch* |
| 5 | Se interpretan los coeficientes del modelo final (odds, odds ratio e intervalos de confianza) |

**Configuración de PyCaret:**

| Parámetro | Valor | Motivo |
|---|---|---|
| `preprocess` | `False` | Los datos ya vienen transformados por el *pipeline* de `ft_engineering.py` |
| `test_data` | Conjunto de prueba de `build_features()` | PyCaret usa exactamente la misma partición que el resto del proyecto |
| `fold_strategy` / `fold` | `stratifiedkfold` / 5 | Validación cruzada estratificada, necesaria por el desbalance |
| `session_id` | 42 | Reproducibilidad |
| `sort` (en `compare_models`) | `AUC` | Métrica de comparación |
| `N_TOP` | 3 | Cantidad de modelos que se analizan en detalle |

**Métricas.** Para cada modelo se calculan AUC, Accuracy, Balanced Accuracy, Recall, Precision y F1, con umbral de 0,5, y también el Recall de la clase 0 (los clientes que no pagan). Como solo ≈ 5 % de los casos no paga a tiempo, la **métrica principal es el AUC**: la Accuracy resulta engañosa, porque un modelo que predice "paga a tiempo" para todos ya acierta ≈ 95 %.

**Funciones reutilizables:** `summarize_classification`, `build_model`, `graficos_comparativos`, `grid_search_lr`, `modelo_produccion`, `tabla_coeficientes`, `intervalos_odds_ratio`, `grafico_coeficientes` y `auc_sin`. Los gráficos comparativos incluyen AUC en entrenamiento frente a prueba (para detectar sobreajuste), curvas ROC superpuestas y matrices de confusión.

### Comparación de modelos

`compare_models` evaluó 14 algoritmos con validación cruzada de 5 folds sobre el conjunto de entrenamiento. Los 3 mejores se entrenaron y se resumieron también en entrenamiento y prueba:

| Modelo | AUC CV (5 folds) | AUC train | AUC test |
|---|:-:|:-:|:-:|
| `gbc` Gradient Boosting | 0,669 | 0,860 | 0,679 |
| `lr` Regresión Logística | 0,660 | 0,688 | **0,689** |
| `ridge` Ridge Classifier | 0,656 | 0,682 | 0,686 |

**Modelo elegido: Regresión Logística (`lr`).**

- **No sobreajusta:** su AUC es 0,688 en entrenamiento y 0,689 en prueba. En cambio, `gbc` pasa de 0,860 en entrenamiento a 0,679 en prueba, es decir, memoriza los datos y no generaliza.
- **Es más simple, rápido y fácil de explicar**, y en validación cruzada queda a solo 0,009 del primer puesto.
- Además, tiene el mayor AUC en el conjunto de prueba.

**Advertencias:**

- Las diferencias entre los 3 modelos son pequeñas (menos de 0,015 de AUC): `lr` y `ridge` son prácticamente equivalentes. La elección de `lr` sobre `gbc` se apoya sobre todo en que `gbc` sobreajusta.
- El poder predictivo general es modesto (AUC cercano a 0,69). Con umbral 0,5 los modelos predicen casi siempre "paga a tiempo", por lo que la Balanced Accuracy ronda 0,50 y las métricas calculadas sobre la clase mayoritaria (Accuracy, Recall, F1) se ven altas pero no aportan información.

### Ajuste de hiperparámetros: Regresión Logística

Se aplicó *GridSearch* con validación cruzada estratificada (5 folds, métrica AUC) sobre el conjunto de entrenamiento, con `class_weight="balanced"` fijado de antemano por el desbalance. Se probaron 12 combinaciones (2 tipos de penalización por 6 valores de `C`). El conjunto de prueba se usó únicamente para reportar.

| Hiperparámetro | Valor | Motivo |
|---|---|---|
| `class_weight` | `"balanced"` | Fijo: solo ≈ 5 % de los casos no paga a tiempo |
| `penalty` | `"l2"` | Mejor AUC en casi todos los `C`; `l1` con `C` bajo anula los coeficientes (AUC 0,54 con `C=0,001` y 0,50 con `C=0,0001`) |
| `C` | `0.001` | Mejor AUC en validación cruzada (0,667); con `C=0,0001` baja a 0,665, así que el óptimo no está en el borde de la grilla |
| `solver` | `"liblinear"` | Fijo; admite `l1` y `l2` |
| `max_iter` | `1000` | Fijo; asegura la convergencia |

**Decisión para producción: solo `class_weight="balanced"`, todo lo demás por defecto.** Ninguna combinación del grid mejoró el AUC de forma real:

- En validación cruzada, la mejor combinación (0,667) supera a la configuración por defecto con `balanced` (0,659) por menos de 0,01, muy por debajo de la desviación entre folds (entre 0,02 y 0,03).
- En prueba, el modelo ajustado quedó incluso peor: AUC 0,669 frente a 0,683 con la configuración por defecto.

Como afinar `penalty` y `C` no aporta una mejora medible, se eligió la configuración más simple y fácil de reproducir:

```python
LogisticRegression(class_weight="balanced")
# resto por defecto: penalty="l2", C=1.0, solver="lbfgs", max_iter=100
```

Converge con los valores por defecto (90 de las 100 iteraciones permitidas) y en prueba da AUC 0,684, Balanced Accuracy 0,634 y Recall de la clase 0 de 0,608.

**Qué cambia al usar `balanced`:**

| Métrica en prueba | Sin pesos | Con `balanced` |
|---|:-:|:-:|
| Recall de la clase 0 (los que no pagan) | 0,00 | 0,61 |
| Balanced Accuracy | 0,50 | 0,63 |
| Accuracy | 0,95 | 0,66 |

La Accuracy baja, algo esperado: el modelo deja de predecir "paga a tiempo" para todos y empieza a identificar a quienes no pagan.

### Interpretación de los coeficientes

Se calculó, para cada variable, el coeficiente (log-odds), el *odds ratio* (cuánto se multiplican los odds de pagar a tiempo) y su correlación máxima con otra variable, para saber qué tan confiable es cada peso. Además, para saber si cada *odds ratio* se distingue de 1 (es decir, de "sin efecto"), se calculó su **intervalo de confianza al 95 %** por *bootstrap*: se reentrena el modelo final sobre 1.000 remuestras del conjunto de entrenamiento y se toman los percentiles 2,5 % y 97,5 % (función `intervalos_odds_ratio`). Como muchas variables están correlacionadas entre sí y el modelo está regularizado, varios intervalos son amplios: que un intervalo incluya 1 significa que el efecto no se distingue de "sin efecto", no que la variable sea irrelevante.

Las numéricas están estandarizadas, así que sus pesos son comparables entre sí. Para cada conclusión se reentrenó el modelo sin esas variables (función `auc_sin`); como las diferencias de AUC están dentro del ruido, lo que se afirma es que **descartarlas no empeora el modelo**.

**1) Variables con muy poco peso.** Nueve variables tienen |coeficiente| < 0,05, es decir, mueven los odds menos de un 5 %:

| Variable | Coeficiente | Odds Ratio |
|---|:-:|:-:|
| `saldo_total_sobre_capital` | +0,048 | 1,050 |
| `creditos_sectorCooperativo` | +0,047 | 1,048 |
| `cant_creditosvigentes` | +0,043 | 1,044 |
| `pct_sector_financiero` | -0,043 | 0,958 |
| `saldo_total` | +0,033 | 1,034 |
| `saldo_mora` | -0,020 | 0,980 |
| `saldo_principal` | +0,015 | 1,015 |
| `salario_cliente` | +0,008 | 1,008 |
| `creditos_sectorFinanciero` | +0,002 | 1,002 |

El intervalo de confianza del *odds ratio* de las nueve incluye 1. Además, sin estas variables el modelo no pierde: el AUC en validación cruzada pasa de 0,659 a 0,669 y el de prueba, de 0,684 a 0,688.

**2) Variables ratio creadas en el EDA.** De los 10 ratios probados, 7 tienen peso bajo (|coeficiente| entre 0,04 y 0,17). Tres parecen pesar mucho (`cuota_sobre_salario` +1,76, `capital_sobre_salario` -0,89 y `otros_prestamos_sobre_salario` +0,58), pero es engañoso: las dos primeras tienen una correlación de 0,998 entre sí y sus pesos se cancelan, y quitar cualquiera de ellas no cambia el AUC. Sin los 10 ratios, el AUC no baja (validación cruzada de 0,659 a 0,662 y prueba de 0,684 a 0,688), por lo que **la mayoría se pueden descartar**.

De los 10 ratios, la base final conserva dos: `cuota_sobre_salario` y `huella_por_credito`. De la pareja `cuota_sobre_salario` / `capital_sobre_salario` (correlación 0,998) se conserva solo la primera, cuyo intervalo de confianza no incluye 1.

### Variabilidad de los resultados

Las cifras de AUC de esta sección pueden variar levemente entre ejecuciones, configuraciones o entornos (por ejemplo, por las versiones de las librerías o la convergencia del optimizador). Estas diferencias son **despreciables**, y la razón es el tamaño de la muestra:

- **Pocos casos de la clase minoritaria:** el conjunto de prueba tiene 2.153 registros, pero solo ≈ 100 son clientes que no pagan. Con esa cantidad, el AUC en prueba tiene un margen de error amplio.
- **Dispersión entre folds:** en validación cruzada, el AUC varía entre 0,02 y 0,03 de un fold a otro.

Frente a ese margen, diferencias de AUC de hasta ≈ 0,02 no permiten distinguir una ejecución, un modelo o una configuración de otra, y **no alteran ninguna de las conclusiones**: ni la elección de la Regresión Logística, ni la decisión de no ajustar los hiperparámetros, ni la de descartar variables.

Dos precisiones:

- Las métricas que dependen del umbral de 0,5 (Balanced Accuracy y Recall de la clase 0) son más sensibles: con ≈ 100 casos, cada cliente mal o bien clasificado mueve el Recall de la clase 0 en ≈ 0,01. Deben leerse como un orden de magnitud. Lo que sí es robusto es el cambio cualitativo de usar `balanced`: el Recall de la clase 0 pasa de 0 a alrededor de 0,6.
- Los coeficientes de las variables de bajo peso están cerca del umbral de 0,05, por lo que la composición exacta de ese grupo puede variar levemente entre ejecuciones. La conclusión de fondo se mantiene: son variables que casi no mueven los odds y se pueden descartar sin perder capacidad predictiva.

### Limitación: enfoque de serie temporal

Dado que los créditos tienen una fecha de desembolso y que el EDA detectó un posible sesgo temporal (los créditos recientes aún no tuvieron tiempo de incumplir), lo ideal para este caso sería un enfoque de **serie temporal**, con una validación que respete el orden cronológico. Sin embargo, no se obtuvieron buenos resultados con esa alternativa: al dividir los datos por períodos quedan muy pocos registros de la clase minoritaria (los clientes que no pagan), lo que impide entrenar y evaluar el modelo de forma estable.

Por este motivo se mantuvo la partición aleatoria estratificada, que se reconoce como una limitación del trabajo. Una validación temporal podría retomarse si se dispone de más registros de clientes que no pagan.

### Base de datos final (`model_final_database.py`)

Como cierre de esta etapa se creó el script `model_final_database.py`, cuyo propósito es simplificar la base de datos con la que se entrena el modelo. A partir de `Base_de_datos_limpia.csv` elimina las variables que, según los coeficientes y los *odds ratio* de la Regresión Logística, no aportan a la predicción, y guarda el resultado como `Base_de_Datos_Final.csv` (sin índice).

| Grupo | Variables eliminadas | Criterio |
|---|---|---|
| Fecha | `fecha_prestamo` | No se usa como variable |
| Peso casi nulo (9) | `saldo_total_sobre_capital`, `creditos_sectorCooperativo`, `cant_creditosvigentes`, `pct_sector_financiero`, `saldo_total`, `saldo_mora`, `saldo_principal`, `salario_cliente`, `creditos_sectorFinanciero` | \|coeficiente\| < 0,05 |
| Ratios sin aporte (6) | `capital_sobre_salario`, `otros_prestamos_sobre_salario`, `cuota_sobre_ingreso_buro`, `ratio_salario_vs_buro`, `ratio_mora_sobre_saldo`, `intensidad_busqueda` | Su *odds ratio* no se distingue de 1: el intervalo de confianza al 95 % (calculado por *bootstrap* en `model_training_evaluation.py`) lo incluye |

De los 10 ratios creados en el EDA, 8 se eliminan (2 por su bajo peso y 6 por no aportar) y 2 se conservan: `cuota_sobre_salario` y `huella_por_credito`.

Con esto la base pasa de 42 a 26 columnas (manteniendo los 10.763 registros) y su tamaño se reduce a la mitad aproximadamente (de ≈ 3,1 MB a ≈ 1,5 MB).

**¿Por qué se eliminan las variables en lugar de omitirlas?** Con el diseño de `ft_engineering.py`, estas variables simplemente podrían haberse omitido del preprocesamiento, por ejemplo agregándolas a la lista de columnas excluidas. Se decidió, en cambio, eliminarlas del propio archivo para reducir el tamaño del CSV y, por consiguiente, el del modelo en producción, que trabaja con menos columnas.

### Conclusiones y próximos pasos

- El modelo elegido es una **Regresión Logística con `class_weight="balanced"`** y el resto de los hiperparámetros por defecto.
- El poder predictivo sigue siendo modesto (AUC < 0,70): las señales disponibles son débiles, como ya anticipaba el EDA.
- La depuración de variables ya se realizó en la base final. Los próximos pasos con más impacto son ajustar el umbral de decisión según el costo de cada tipo de error y revisar si existen variables nuevas que aporten señal, más que seguir afinando hiperparámetros.

### Versionado de esta etapa

Esta etapa se desarrolló en la rama `developer`. Una vez comprobado que todo estaba bien, los cambios se integraron a la rama `main` mediante un *merge*.

## Avance 1.3.0: Monitoreo de Data Drift

En esta etapa se construyó el componente de monitoreo (`model_monitoring.py`), que permite detectar el *data drift*: el cambio en la distribución de los datos con los que se usa el modelo respecto de los datos con los que fue entrenado. Si los datos nuevos se parecen cada vez menos a los de entrenamiento, el modelo puede perder capacidad predictiva, por lo que conviene detectarlo a tiempo.

El componente tiene dos modos de uso:

- **Simulación:** genera datos con un cambio controlado, mes a mes, y mide el drift hasta que se detecta o se llega al tope de meses.
- **Datos reales:** si se indica la ruta de un dataset nuevo, se omite la simulación y se mide el drift de ese dataset frente al de entrenamiento.

### Componentes que reutiliza

- **`model_deploy.py`:** construye el modelo final, una Regresión Logística con `class_weight="balanced"` entrenada con las variables transformadas por `ft_engineering.py`. De este archivo, el monitoreo toma los datos de entrenamiento transformados (`X_train`), el modelo (`pipe_modelo`) y el preprocesador ya ajustado (`preprocesador`). Este archivo se llamó `model_deploy.py` hasta el avance V1.3.1 (ver el aviso de esa sección).
- **`ft_engineering.py`:** aporta la configuración compartida (columnas categóricas y excluidas, ruta de la base, semilla y variable objetivo).

Utiliza además `scipy`, `numpy` y `pandas`, ya instaladas en el entorno virtual.

### ¿Qué se compara?

- **Referencia:** las variables de entrenamiento ya transformadas por el *pipeline* (28 variables), más el **pronóstico del modelo** (la probabilidad de pagar a tiempo, `proba`). Incluir el pronóstico permite detectar también un cambio en lo que el modelo predice, y no solo en las variables de entrada. En total se monitorean **29 variables**.
- **Datos actuales:** los datos simulados (o el dataset nuevo), transformados con el mismo preprocesador que se ajustó en el entrenamiento.

Las variables con **10 valores distintos o menos** (15 de las 29: dummies, binarias, la variable ordinal y las discretas con pocos valores) se tratan como categóricas; las otras 14 se tratan como numéricas.

### Métricas de drift

| Métrica | Se aplica a | Qué mide | Umbral de alerta |
|---|---|---|---|
| **KS** (Kolmogorov-Smirnov) | Numéricas | Diferencia máxima entre las distribuciones acumuladas de referencia y actual | p-valor < 0,05 |
| **PSI** (Population Stability Index) | Numéricas | Cuánto cambió la proporción de datos en cada uno de los 10 tramos definidos con los cuantiles de la referencia | PSI > 0,25 |
| **Jensen-Shannon** | Numéricas | Distancia entre las distribuciones por tramos (de 0 a 1, base 2) | JS > 0,1 |
| **Chi-cuadrado** | Categóricas | Compara las frecuencias de cada categoría entre referencia y actual | p-valor < 0,05 |

Una variable numérica se marca con drift si **cualquiera** de sus tres métricas supera su umbral, y una categórica si su p-valor del chi-cuadrado es menor a 0,05. Para cada variable, el resultado es una fila con sus métricas y un indicador `drift`.

### Simulación de drift

Cada iteración representa un mes desde el entrenamiento (un "commit" de simulación). En cada mes:

1. Se toma una **muestra de 1.000 filas** de la base de datos final, con una semilla distinta por mes (`SEMILLA + mes`), de modo que la simulación es reproducible.
2. Se le aplica una **perturbación controlada**, definida en la configuración.
3. Se transforma con el preprocesador, se calcula el pronóstico del modelo y se miden las métricas de drift contra la referencia.
4. Si **una métrica de una variable** supera su umbral, se informa el mes de detección y se detiene la simulación; si no, continúa hasta el tope de meses.

**Perturbación aplicada** (solo sobre el 30 % de las filas de la muestra para las variables numéricas):

| Variable | Tipo de cambio | Variación |
|---|---|---|
| `edad_cliente` | Media | +5 % |
| `capital_prestado` | Media y desviación estándar | +5 % y +10 % |
| `puntaje_datacredito` | Media | -3 % |
| `tipo_laboral` | Proporción de la categoría `Independiente` | +5 % |

**Parámetros de configuración:**

| Parámetro | Valor | Descripción |
|---|---|---|
| `RUTA_DATASET_NUEVO` | `None` | Si se indica un CSV, se mide su drift y se omite la simulación |
| `N_MUESTRA` | 1.000 | Filas muestreadas de la base en cada mes |
| `FRAC_FILAS` | 0,30 | Fracción de filas a las que se les modifica la media y la desviación |
| `MAX_MESES` | 24 | Tope de meses simulados |
| `UMBRAL_PSI` / `UMBRAL_JS` / `UMBRAL_P` | 0,25 / 0,1 / 0,05 | Umbrales de alerta |
| `MAX_CATEGORIAS` | 10 | Máximo de valores distintos para tratar una variable como categórica |

### Resultado

Al ejecutar el script se imprime la tabla de métricas del último mes simulado (el de la detección, si la hubo) y la cantidad de variables con drift en cada mes. Con la configuración por defecto, **el drift se detectó en el mes 1**, en una variable:

| Variable | KS (p-valor) | PSI | JS |
|---|:-:|:-:|:-:|
| `puntaje_datacredito` | 0,0013 | 0,027 | 0,070 |

- **El drift lo detectó la prueba KS:** el PSI y el Jensen-Shannon se mantuvieron muy por debajo de sus umbrales (0,25 y 0,1), lo que indica que el cambio es pequeño. KS es la métrica más sensible: con muestras de 1.000 filas detecta desplazamientos leves.
- **`puntaje_datacredito` es la variable cuya media se redujo** en la perturbación (-3 %).
- **El pronóstico del modelo (`proba`) no superó su umbral en ese mes**, aunque quedó cerca (KS con p-valor de 0,061, frente al umbral de 0,05).
- **Las demás variables perturbadas no superaron sus umbrales** (`edad_cliente`, `capital_prestado` y `tipo_laboral`, con p-valores de 0,57, 0,52 y 0,48), porque el cambio aplicado es pequeño para el tamaño de la muestra.

### Uso con datos reales

Para medir el drift de un dataset real, se indica su ruta en `RUTA_DATASET_NUEVO`. El archivo debe tener las mismas columnas que la base original. El script lo prepara igual que en el entrenamiento, lo compara contra la referencia y muestra la tabla de métricas y la cantidad de variables con drift.

## Avance V1.3.1: Disponibilización del Modelo por Streamlit

En esta etapa se construyó una aplicación web con **Streamlit** (`model_streamlit.py`) que permite usar el modelo y monitorear su data drift desde el navegador, sin tocar el código. La aplicación tiene dos funciones principales:

- **Predecir:** calcula la probabilidad de que un cliente pague a tiempo, de forma individual o por lotes (CSV), y la traduce a una banda de riesgo.
- **Monitorear:** simula el data drift mes a mes con las métricas de `model_monitoring.py` y muestra el estado de cada variable, la evolución en el tiempo y recomendaciones.

### Librerías adicionales

Se agregaron `streamlit` y `plotly` al archivo `requirements.txt` y se instalaron en el entorno virtual. La aplicación utiliza además `scipy`, `numpy` y `pandas`, ya instaladas.

### Archivos involucrados

| Archivo | Rol |
|---|---|
| `model.py` | Modelo final (Regresión Logística). Es el archivo que antes se llamaba `model_deploy.py`. Expone los datos de entrenamiento transformados (`X_train`), el preprocesador, el modelo y el *pipeline* completo (`pipe_final`, que va de los datos originales a la probabilidad) |
| `model_monitoring.py` | Simulación de drift y métricas (KS, PSI, Jensen-Shannon y chi-cuadrado). Su única modificación fue importar el modelo desde `model` en lugar de `model_deploy` |
| `model_streamlit.py` | Aplicación Streamlit. Reutiliza los dos archivos anteriores tal como están |

### Ejecución

Desde la carpeta del proyecto, con el entorno virtual activado:

```bash
streamlit run model_sreamlit.py
```

La aplicación carga el modelo y los datos **una sola vez** por sesión (con `st.cache_resource`) y vuelve a simular solo cuando cambian los parámetros de la barra lateral.

### Barra lateral: parámetros de la simulación

Los parámetros se aplican todos juntos al presionar **Simular**. Sus valores iniciales salen de la configuración de `model_monitoring.py`:

| Parámetro | Rango | Valor inicial | Descripción |
|---|---|---|---|
| Meses a simular | 3 a 36 | 24 | Cantidad de meses desde el entrenamiento |
| Drift acumulativo | Sí / No | Sí | Si está activado, en el mes *k* las variaciones se multiplican por *k*, de modo que el drift crece con el tiempo. Si no, cada mes repite la misma variación |
| Filas por mes | 200 hasta el tamaño de la base | 1.000 | Muestra que se toma de la base cada mes |
| Fracción de filas modificadas | 0,05 a 1,0 | 0,30 | Proporción de filas a las que se les cambia la media y la desviación |

La opción acumulativa resuelve una limitación de `model_monitoring.py` (que repite la misma variación cada mes): permite ver cómo evoluciona el drift cuando se acentúa con el tiempo. Un selector adicional, **Mes a analizar**, define para qué mes se muestran las pestañas de detalle.

### Pestañas de la aplicación

**1. Predicción**

- **Individual:** un formulario con un campo por cada variable de entrada (24 en total, las de la base final, incluidas las derivadas). Las numéricas parten de la mediana de la base y las categóricas ofrecen un selector con las opciones existentes. Al presionar **Calcular riesgo**, la aplicación muestra la probabilidad de pagar a tiempo, el riesgo de incumplimiento (1 menos esa probabilidad) y la banda de riesgo.
- **Por lotes:** se descarga una plantilla CSV con las columnas esperadas, se completa y se sube. Se aceptan separadores coma o punto y coma, las columnas de más se ignoran y, si falta alguna, la aplicación avisa. El resultado agrega tres columnas (`proba_pago_a_tiempo`, `riesgo_incumplimiento` y `banda_riesgo`), muestra la cantidad de clientes evaluados, el riesgo promedio y un gráfico de clientes por banda, y se puede descargar como CSV.

Las bandas de riesgo (valores ilustrativos) se definen sobre el riesgo de incumplimiento:

| Riesgo de incumplimiento | Banda |
|---|---|
| Menos de 20 % | 🟢 Bajo |
| 20 % a 40 % | 🟡 Moderado |
| 40 % a 60 % | 🟠 Medio-alto |
| 60 % a 80 % | 🔴 Alto |
| 80 % o más | 🟣 Muy alto |

**2. Drift por variable**

Muestra, para el mes elegido, cuántas variables están en cada estado, una tabla con las métricas de cada una (PSI, JS, KS y chi-cuadrado) y una barra de riesgo. Además, permite comparar el histograma del entrenamiento con el del mes actual para cualquier variable. El estado de cada variable es un semáforo:

| Estado | Variables numéricas | Variables categóricas |
|---|---|---|
| 🔴 Crítico | PSI > 0,25 o JS > 0,10 | Chi-cuadrado con p < 0,001 |
| 🟡 Alerta | PSI > 0,10 o KS con p < 0,05 | Chi-cuadrado con p < 0,05 |
| 🟢 Estable | Ninguna de las anteriores | Ninguna de las anteriores |

Este semáforo es más gradual que el criterio de `model_monitoring.py`, donde cualquier métrica fuera de umbral marca drift: aquí, un KS significativo por sí solo solo genera una alerta.

**3. Evolución temporal**

- **PSI medio y máximo por mes**, con las líneas de umbral crítico (0,25) y de alerta (0,10).
- **Cambios abruptos:** se marcan los meses en que el salto del PSI medio respecto del mes anterior es atípico (más de 2 desvíos estándar).
- **Tendencia:** se ajusta una regresión lineal al PSI medio y se informa si crece o decrece de forma significativa (p < 0,05).
- **Primer mes con drift** según el criterio de `model_monitoring.py`.
- **PSI por variable numérica** a lo largo del tiempo (por defecto, las 5 de mayor PSI).
- **Probabilidad media de pagar a tiempo** por mes, frente a la de referencia del entrenamiento.

### Versionado de esta etapa

Esta etapa se desarrolló en la rama `developer`. Una vez comprobado que todo estaba bien, los cambios se integraron a la rama `main` mediante un *merge*.
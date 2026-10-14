# Ejercicios de S7 y S8

Los dos ejercicios usan el mismo problema, los mismos datos sintéticos y el
mismo validador. Lo que se valora es una comparación reproducible y honesta:
no se exige que gane ningún modelo.

## S7 · Adaptar un modelo abierto

Ajusta un modelo abierto para clasificar incidencias ficticias de soporte y
devolver un JSON con categoría, prioridad y una respuesta breve. Compara
entrenarlo con mejorar únicamente las instrucciones del modelo.

### Trabajo obligatorio

1. Usa Qwen3-0.6B con Transformers, Datasets, TRL y PEFT. Configura
   `OPEN_MODEL_ID` y registra versiones. Separa 150–300 ejemplos de
   entrenamiento, 30 de validación y 30 de prueba, sin duplicados; define
   categorías, prioridades y JSON antes de comenzar. Solo datos sintéticos, sin
   información personal ni claves.
2. Compara el modelo base con instrucciones básicas, el base con few-shot y
   LoRA. Configuración LoRA fija, sin búsqueda de hiperparámetros. Base y
   ajustado reciben las mismas instrucciones básicas. Few-shot toma ejemplos de
   entrenamiento; mejóralo solo con validación y congélalo antes del test.
3. Mismo chat template en entrenamiento e inferencia, `enable_thinking=False`,
   sin duplicar tokens especiales. Mismos parámetros y límite de salida en las
   tres condiciones.
4. JSON como texto, sin JSON mode, herramientas que impongan formato, reparación
   ni reintentos correctores. Un único validador. Calcula JSON válido y aciertos
   de categoría y prioridad sobre los 30 casos. Una salida no interpretable es
   un error. Conserva protocolo y datos para S8.
5. Sube el adaptador al Hub, privado si prefieres, con una model card que
   indique modelo, datos, configuración, resultados, licencia y límites.
   Recárgalo desde el Hub y haz una predicción; no hace falta un endpoint de pago.

### Entrega y valoración

Código, datos sintéticos separados, respuestas originales, tabla de resultados
y enlace al adaptador. Se valora la separación de datos, la comparación justa,
el entrenamiento reproducible y la recarga. Dedicación orientativa: 5–7 horas
activas con GPU o entorno preparado, aparte de las esperas. Opcional: analizar
tres errores sin reajustar con el test.

### Cómo lo resuelve el repo

| Paso | Comando o archivo |
|---|---|
| Datos 240/30/30 y desarrollo | `datos/`, comprobados por `curso/common.py:validate_splits` |
| Original, few-shot y LoRA | `scripts/demo.py --proveedor qwen [--fewshot] [--adaptador …]` |
| Entrenamiento | `scripts/entrenar_lora.py --salida modelos/…` |
| Evaluación de las tres condiciones | `scripts/evaluar_qwen.py` y `scripts/resumir.py` |
| Publicación y recarga desde el Hub | `scripts/publicar_adaptador.py --repo TU_USUARIO/NOMBRE` |

El repo incluye 240/30/30 y una configuración fija. Una variante necesita sus
propios datos, manifiesto y resultados; no la mezcles con la referencia.
Publicar requiere tu cuenta de Hugging Face. Si usas CPU o MPS, declara el
hardware y sus tiempos.

## S8 · Modelos cerrados

Realiza un fine-tuning real de Gemini con el problema y los datos de S7. Añade
Claude sin ajustar y compara modelos abiertos y cerrados. Claude es una
referencia adicional y no sustituye el entrenamiento.

### Trabajo obligatorio

1. Usa el laboratorio de Google Cloud del curso. Modelo de referencia:
   `gemini-3.1-flash-lite`. Modelo de comparación: **Claude Haiku 5.5**
   (`claude-haiku-5-5`) por la **API de Anthropic**. IDs de modelos, proyecto y
   ubicaciones en variables (`.env`), sin credenciales publicadas.
2. Reutiliza exactamente train, las 30 de validación y las 30 de test de S7.
   Conversor a JSONL `contents`/`parts` con correspondencia de IDs. El test queda
   fuera del entrenamiento y del few-shot.
3. Ajuste en `us-central1` o `europe-west4` y uso del modelo ajustado en la
   ubicación `us` o `eu` correspondiente. Thinking MINIMAL y mismas instrucciones
   y configuración para Gemini base y ajustado. Conserva el ID del trabajo y del
   modelo y sus predicciones. Cambiar solo el prompt no cumple.
4. Seis condiciones: Qwen básico, few-shot y LoRA; Gemini base y ajustado; Claude
   con el prompt y los ejemplos de entrenamiento de S7. Reutiliza los resultados
   de Qwen solo si datos y configuración siguen iguales.
5. JSON pedido como texto, sin JSON mode, herramientas que impongan formato,
   reparación ni reintentos correctores. Respuestas originales, un único
   validador y los mismos 30 casos; las respuestas no interpretables son errores.
6. Registra calidad, latencia, tokens y coste cuando existan. Separa
   entrenamiento e inferencia y declara hardware y proveedor. No inventes
   consumos ni supongas la misma tarifa para base y ajustado. Justifica tu
   elección por calidad, coste y control.

### Entrega y valoración

Conversor, scripts, evidencia del entrenamiento y tabla de seis condiciones. Se
valora un fine-tuning verificable, incluir Claude, la separación del test y el
análisis. Dedicación orientativa: 4–6 horas activas con datos y accesos
preparados, aparte de las esperas. Las salidas estructuradas son una ampliación
aparte.

### Cómo lo resuelve el repo

| Condición | Comando |
|---|---|
| Qwen básico, few-shot y LoRA | Los de S7 (`scripts/evaluar_qwen.py`) |
| Gemini base | `scripts/evaluar_remotos.py --condicion gemini_base --split test --salida …` |
| Gemini ajustado | `scripts/evaluar_remotos.py --condicion gemini_ajustado --modelo RECURSO_AJUSTADO --split test --salida …` |
| Claude con prompt y ejemplos | `scripts/evaluar_remotos.py --condicion claude_prompt --split test --salida …` |
| Tabla de seis condiciones | `scripts/resumir.py --carpeta … --destino …` |
| Conversión y ajuste | `scripts/convertir_gemini.py` y `scripts/gemini_tuning.py preparar / subir-datos / crear / estado` |

`claude_prompt` usa la ruta de `.env`: por defecto, Claude Haiku 5.5 por la API de
Anthropic. Haiku 5.5 no admite una temperatura distinta de la predeterminada; el
resto del protocolo se mantiene. Si usas otra ruta o modelo, el informe muestra
cuál y debes declararlo. Para explorar otros modelos sin sustituir ninguna
condición, usa `--condicion remoto_basico`. Las condiciones de Gemini van siempre
por Google Cloud: otra ruta a Gemini no sustituye el ajuste.
[Proveedores y procedimiento](modelos-remotos.md).

El acceso al laboratorio de Google Cloud lo facilita el curso; hasta tenerlo,
puedes preparar conversión, especificación y el resto de condiciones.

## Si no tienes GPU o laboratorio

Para practicar el ajuste con un negocio ficticio en tu cuenta, tienes el
[laboratorio Nébula Hogar](gemini-negocio.md): dataset, scripts y comparación
base/ajustado/control completos. Es una práctica independiente del protocolo
de tickets anterior; sus entregables específicos están en la guía. No mezcles
sus cifras con la tabla de seis condiciones de S7/S8.

Se puede estudiar sin cuenta o saldo con los 378 originales docentes publicados,
marcando esa procedencia. Entrenar en una cuenta propia requiere proyecto,
permisos, bucket, cuota, facturación y presupuesto del alumno.

Se puede utilizar un entorno remoto con GPU si se dispone de acceso. También se
admite una preparación completa cuando no hay ese acceso: datos comprobados,
código, configuración, comandos de entrenamiento, recarga y evaluación, y la
explicación de la limitación.

Marca las métricas no obtenidas como pendientes. No inventes entrenamientos ni
resultados. No es necesario contratar recursos de pago para esta modalidad. Se
valora la comprensión, la separación de datos, la comparación y la preparación;
en la modalidad ejecutada, la evidencia del ajuste, la recarga y el análisis.

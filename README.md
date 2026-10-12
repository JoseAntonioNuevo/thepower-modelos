# Modelos abiertos y cerrados · S7 y S8 de AI Engineer

Repositorio de las sesiones **S7 · Modelos abiertos** y **S8 · Modelos cerrados y
APIs** del programa AI Engineer de thePower. Un mismo caso práctico recorre las
dos clases: convertir una incidencia ficticia de una plataforma educativa en un
JSON con `categoria`, `prioridad` y `respuesta`.

- En **S7** elegimos, descargamos y ejecutamos un modelo abierto (Qwen3-0.6B),
  lo comparamos con instrucciones, con ejemplos few-shot y con un adaptador LoRA
  entrenado por nosotros, y lo evaluamos con casos separados.
- En **S8** conectamos modelos por API —de cualquier proveedor, elegido en `.env`;
  en clase, Claude Haiku 5.5 por la API de Anthropic—, leemos su consumo, los
  comparamos con los mismos datos y el mismo validador, y vemos cómo se ajusta
  un modelo que no podemos descargar (Gemini en Google Cloud).

El código, los scripts, los tests y los notebooks están comentados en español.
Si trabajas con un asistente de IA, puedes preguntarle cualquier cosa del repo:
[AGENTS.md](AGENTS.md) le da el mapa y
[las preguntas frecuentes](docs/preguntas-frecuentes.md) le dan las respuestas.

## Contenido

- [Las dos clases](#las-dos-clases)
- [El caso práctico](#el-caso-práctico)
- [Qué hay en el repositorio](#qué-hay-en-el-repositorio)
- [Instalación](#instalación)
- [S7 paso a paso: un modelo abierto en tu equipo](#s7-paso-a-paso-un-modelo-abierto-en-tu-equipo)
- [S8 paso a paso: modelos por API](#s8-paso-a-paso-modelos-por-api)
- [Resultados de referencia](#resultados-de-referencia)
- [Ejercicios](#ejercicios)
- [Trabajar con un asistente de IA](#trabajar-con-un-asistente-de-ia)
- [Documentación](#documentación)
- [Comprobaciones, seguridad y licencia](#comprobaciones-seguridad-y-licencia)

## Las dos clases

### S7 · Modelos abiertos

> **¿Qué modelo puedo utilizar para mi tarea, cómo lo pongo en funcionamiento y
> cómo compruebo si sus respuestas son suficientemente buenas?**

Recorrido: **equipo → elección del modelo → Hugging Face → ejecución local →
comparación → prompting → adaptación → evaluación**.

Al terminar deberías poder:

- Distinguir **modelo, pesos, formato, runtime y proveedor**.
- Elegir un candidato por tarea, idioma, hardware, licencia y calidad observada,
  y estimar su memoria (parámetros × bits ÷ 8, más contexto y runtime).
- Leer una model card de Hugging Face y reconocer los archivos necesarios
  (Safetensors, GGUF, MLX) y la revisión que estás usando.
- Descargar un formato compatible, cargarlo y generar una respuesta local
  (Transformers o LM Studio) y entender cómo otra aplicación lo consulta por API.
- Comparar respuestas con criterios concretos, también a ciegas en isbetter.ai.
- Distinguir **prompting, few-shot y fine-tuning**, y reconocer qué contiene un
  adaptador LoRA y cómo se recarga.
- Evaluar con casos separados y documentar resultados y limitaciones.

En el repositorio: `Qwen/Qwen3-0.6B` con revisión fijada, Transformers + PEFT +
TRL para entrenar un adaptador LoRA, una API local para comparar original y LoRA
en isbetter.ai y la ruta de LM Studio. LoRA, QLoRA y sus alternativas se
profundizan más adelante en el programa (S13).

### S8 · Modelos cerrados y APIs

> **¿Cómo conecto mi aplicación a estos modelos y qué pruebas necesito para
> elegir el que mejor resuelve mi tarea?**

Recorrido: **proveedor y modelo → petición API → respuesta y consumo →
comparación por tareas → adaptación gestionada → decisión**.

Al terminar deberías poder:

- Identificar el **proveedor**, el **modelo** concreto y la **ruta** que usas
  (API nativa o interfaz común como OpenRouter). Abierto o cerrado depende de
  los pesos; remoto o local, de dónde se ejecuta: son decisiones distintas.
- Entender una petición: endpoint, autenticación, instrucciones, entrada y
  límite de salida. Leer la respuesta: texto, motivo de parada, tokens, tiempo y
  coste disponible.
- Comparar modelos con el mismo caso y criterios definidos antes, y votar con
  una prueba observable. En clase se comparan tres tareas en isbetter.ai
  (un dossier con citas, código con un contrato y un plan con restricciones).
- Justificar una elección por calidad, integración, latencia, coste y
  condiciones de los datos: abierto, cerrado o híbrido.
- Diferenciar **inferencia, prompting, few-shot y entrenamiento gestionado**, y
  reconocer qué evidencia demuestra que un modelo se ha ajustado.

En el repositorio: una capa de proveedores intercambiables (`curso/remote.py`)
con Anthropic, OpenAI, OpenRouter, cualquier API compatible con Chat
Completions, Gemini API, LM Studio y Vertex AI; la evaluación remota con el mismo
test de S7; y la conversión de datos y el procedimiento de ajuste de Gemini en
Google Cloud.

## El caso práctico

Entrada: una incidencia escrita por un usuario. Salida esperada, solo JSON:

```json
{
  "categoria": "facturacion",
  "prioridad": "media",
  "respuesta": "Indica el concepto y la fecha del pago para revisar la incidencia de facturación."
}
```

Las reglas están en [`prompts/basico.txt`](prompts/basico.txt) y son las mismas
para todos los modelos:

| Campo | Valores |
|---|---|
| `categoria` | `acceso` (login, contraseña, permisos), `facturacion` (cobros, facturas, pagos), `tecnico` (fallos de la plataforma), `consulta` (información general o sin problema concreto) |
| `prioridad` | `alta` si hay impacto colectivo explícito o datos privados expuestos; `media` si una persona no puede completar una acción o hay un problema de cobro; `baja` para consultas, sugerencias o fallos sin bloqueo |
| `respuesta` | Una o dos frases en español, sin inventar acciones, plazos ni reembolsos y sin pedir contraseñas |

«Urgente» por sí solo no sube la prioridad. Si faltan datos: `consulta`, `baja`
y una aclaración breve. Son convenciones del ejercicio, no una taxonomía
universal. Una respuesta solo es válida si es JSON, usa valores admitidos y
acierta las etiquetas: lo comprueba `curso/common.py:validate_answer` sin
reparar nada.

## Qué hay en el repositorio

```text
README.md                 Este documento
AGENTS.md (= CLAUDE.md)   Guía para asistentes de IA: mapa, contrato y cómo responder
SECURITY.md               Límites de seguridad de la API local y de las rutas remotas
.env.example              Plantilla de configuración: proveedor remoto, claves y modelo local

curso/                    Núcleo compartido
  common.py               Datos, mensajes, few-shot, validador, manifiesto congelado
  qwen.py                 Carga Qwen (y su adaptador), chat template y generación
  remote.py               Proveedores remotos y locales: route() + chat() desde .env
  http.py                 Lectura acotada y segura de respuestas HTTP

scripts/                  Comandos de la práctica
  preflight.py            Comprueba datos, dependencias y configuración (sin inferencia)
  descargar_modelo.py     Descarga Qwen3-0.6B en la revisión fijada
  demo.py                 Una consulta: Qwen original/few-shot/LoRA o cualquier proveedor
  entrenar_lora.py        Entrena el adaptador LoRA (SFT con TRL + PEFT)
  evaluar_qwen.py         Evalúa básico, few-shot y LoRA sobre una partición
  evaluar_remotos.py      Evalúa condiciones remotas (Claude, Gemini, cualquier ruta)
  resumir.py              Une resultados en JSON, CSV y una tabla Markdown
  servir_comparacion.py   API local con qwen-original y qwen-lora-soporte para isbetter.ai
  convertir_gemini.py     Convierte los datos al formato de ajuste de Gemini
  gemini_tuning.py        Prepara, sube, crea y consulta el ajuste gestionado de Gemini
  publicar_adaptador.py   Publica el adaptador en tu cuenta de Hugging Face y lo recarga
  preparar_datos.py       Generó el dataset sintético (ejecutarlo lo regenera)

datos/                    Train 240, validación 30, test 30 y desarrollo 6 (JSONL)
  gemini/                 Los mismos datos en formato Gemini e ids.json
prompts/                  basico.txt (instrucciones) y fewshot-ids.json (4 ejemplos)
config/                   experimento-congelado.json: modelo, revisión y hashes
notebooks/                S7-LoRA.ipynb y S8-Gemini-Claude.ipynb, por etapas
tests/                    Contratos de datos, validador, HTTP, proveedores y API local
docs/                     Guía del código, preguntas frecuentes, protocolo, ejercicios,
                          modelos remotos, resultados de referencia y publicación
```

No vienen en Git, y es normal que no existan en una copia nueva: pesos del
modelo, adaptadores (`modelos/`), resultados (`resultados/`), tu `.env` con
claves y las carpetas de trabajo local (`material/`). El modelo se descarga
con un script y el adaptador se obtiene al entrenar.

## Instalación

Clona el repo, abre su carpeta y utiliza Python 3.12. Con [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12 .venv
uv pip sync --python .venv requirements-training.lock
source .venv/bin/activate
cp .env.example .env
```

En Windows PowerShell: `.venv\Scripts\Activate.ps1` y
`Copy-Item .env.example .env`. Sin uv, ejecuta `python -m venv .venv`, activa
el entorno y ejecuta `python -m pip install --require-hashes -r requirements-training.lock`.
Para ejecutar solo los contratos, sin PyTorch, basta con `requirements.lock`.

La práctica local de S7 no necesita claves. Para S8, rellena en `.env` la clave
del proveedor que vayas a usar. Los locks incluyen hashes y dependencias por
plataforma; Linux puede descargar los paquetes CUDA de PyTorch aunque el equipo
use CPU. La ejecución en Mac se ha comprobado; resolver paquetes para
Windows/Linux no verifica su hardware.

## S7 paso a paso: un modelo abierto en tu equipo

### 1. Comprobar y descargar el modelo original

```bash
python scripts/preflight.py
python scripts/descargar_modelo.py
```

La primera descarga necesita Internet y espacio en disco: unos 1,5 GB de pesos
Safetensors en BF16, en una revisión fija y sin credenciales.
`adaptador_local: false` es normal antes del primer entrenamiento.

### 2. Consultar al original y con few-shot

```bash
python scripts/demo.py --proveedor qwen --caso DE-04 \
  --salida resultados/ensayo/base-DE04.json
python scripts/demo.py --proveedor qwen --caso DE-04 --fewshot
```

`DE-04` es *«Tengo un cobro repetido en mi matrícula»*. Few-shot añade cuatro
ejemplos de train al contexto de la consulta: no cambia ningún peso.

### 3. Entrenar y recargar el adaptador LoRA

```bash
python scripts/entrenar_lora.py --salida modelos/lora-soporte
python scripts/demo.py --proveedor qwen --caso DE-04 \
  --adaptador modelos/lora-soporte --salida resultados/ensayo/lora-DE04.json
```

La receta es fija: dos épocas, r=8, alpha=16, dropout=0.05, batch 2 y
acumulación 4 (60 pasos de optimizador). Solo se entrenan las matrices LoRA;
el modelo base queda congelado. Se usa CUDA, MPS o CPU según disponibilidad;
CPU puede resultar lenta. El entrenador rechaza una salida no vacía: para otro
ensayo usa otro `--salida`. El adaptador guardado (unos 20 MB) contiene
`adapter_model.safetensors`, `adapter_config.json`, el tokenizer y
`entrenamiento.json` con la receta y las métricas.

### 4. Comparar original y LoRA en isbetter.ai

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/servir_comparacion.py --adaptador modelos/lora-soporte
```

En isbetter.ai, proveedor **Local**: `http://localhost:1235/v1`. Elige
**qwen-original** y **qwen-lora-soporte**: es el mismo Qwen, y solo cambia
activar el adaptador. En **Advanced instructions** pega `prompts/basico.txt`;
en el prompt normal, una incidencia. Lee el JSON en **Output** y vota con una
razón observable. Las peticiones se atienden en serie: el tiempo visible puede
incluir cola. `/health` identifica la revisión y el hash del adaptador. Ctrl+C
detiene el servidor. Escucha solo en 127.0.0.1 y su CORS solo admite isbetter.ai:
no lo expongas a una red. [Límites de seguridad](SECURITY.md).

### 5. Hugging Face y LM Studio

Para ejecutar el modelo como aplicación de escritorio, en Hugging Face abre
`Qwen/Qwen3-0.6B-GGUF`, revisa autor, model card, licencia y archivos, y
descarga en LM Studio (Discover) `Qwen3-0.6B-Q8_0.gguf`. Carga el modelo,
arranca su servidor local y consúltalo desde el repo:

```bash
python scripts/demo.py --proveedor local --caso DE-01
```

LM Studio escucha por defecto en `http://localhost:1234/v1` (`LOCAL_BASE_URL`).
Descargar, cargar en memoria y consultar son operaciones distintas; chatear no
entrena el modelo.

### 6. Evaluar con el test

Congela instrucciones y configuración antes de mirar test:

```bash
python scripts/evaluar_qwen.py --condicion qwen_base --split test \
  --salida resultados/ensayo/qwen_base.jsonl
python scripts/evaluar_qwen.py --condicion qwen_fewshot --split test \
  --salida resultados/ensayo/qwen_fewshot.jsonl
python scripts/evaluar_qwen.py --condicion qwen_lora --split test \
  --adaptador modelos/lora-soporte --salida resultados/ensayo/qwen_lora.jsonl
python scripts/resumir.py --carpeta resultados/ensayo \
  --destino resultados/ensayo/informe
```

El informe incluye JSON, CSV y una tabla Markdown. Se conservan las respuestas
originales, sin reparar ni reintentar: un JSON inválido cuenta como fallo y
sigue en el denominador.
JSON válido, esquema, categoría, prioridad y ambas etiquetas se miden por
separado. Los runners rechazan archivos existentes. [Protocolo](docs/experimento.md).

## S8 paso a paso: modelos por API

### 1. Elegir proveedor y modelo en `.env`

El código no está atado a ningún proveedor. Tres líneas de `.env` deciden a
quién se pregunta: **por dónde** (`LLM_PROVIDER`), **qué modelo** (`LLM_MODEL`,
con el formato de ID de esa ruta) y **la clave** de esa ruta. Rellena solo la
clave que vayas a usar; `.env` nunca se sube a Git.

```bash
# .env — la ruta del curso
LLM_PROVIDER=anthropic
LLM_MODEL=claude-haiku-5-5
ANTHROPIC_API_KEY=…            # tu clave; nunca en Git, capturas ni chats
```

| Para usar… | `LLM_PROVIDER` | Qué poner en `.env` | Ejemplo de `LLM_MODEL` | Coste |
|---|---|---|---|---|
| Claude por la API de Anthropic (ruta del curso) | `anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-5-5` | Tarifa de `LLM_PRECIO_*` |
| GPT por la API de OpenAI | `openai` | `OPENAI_API_KEY` | El ID que muestre tu cuenta | Tarifa de `LLM_PRECIO_*` |
| **Cualquier modelo de OpenRouter**: Claude, GPT, Gemini, GLM, Kimi, DeepSeek, Qwen, Llama, Grok… | `openrouter` | `OPENROUTER_API_KEY` | `z-ai/glm-5.3` (ID exacto de su catálogo) | Lo informa OpenRouter |
| Gemini con clave de Google AI Studio | `gemini` | `GEMINI_API_KEY` | `gemini-3.1-flash-lite` | Tarifa de `LLM_PRECIO_*` |
| Groq, Mistral, DeepSeek, Together… | `compatible` | `COMPAT_BASE_URL` (https) y `COMPAT_API_KEY` | El ID de ese servidor | Tarifa de `LLM_PRECIO_*` |
| Ollama o vLLM en tu equipo | `compatible` | `COMPAT_BASE_URL=http://localhost:11434/v1` | `qwen3:0.6b` | No medido (`null`) |
| LM Studio | `local` | Nada | Lo descubre solo | No medido (`null`) |
| Gemini base o ajustado en Google Cloud | `google` | `GOOGLE_CLOUD_PROJECT` y login con `gcloud` (no usa clave) | `gemini-3.1-flash-lite` o el recurso ajustado | No medido (`null`); estímalo con la tarifa de Google |

Para cambiar de proveedor:

1. En `.env`, cambia `LLM_PROVIDER` y `LLM_MODEL` y pon la clave de esa ruta.
2. Ajusta la tarifa: `LLM_PRECIO_ENTRADA_USD_MTOK` y `LLM_PRECIO_SALIDA_USD_MTOK`
   (USD por millón de tokens) con el precio de tu modelo, o déjalas vacías y el
   coste quedará `null` (no medido). Si dejas la tarifa de Claude con otro
   modelo, el coste saldrá mal. OpenRouter no la necesita: informa el importe.
3. `python scripts/preflight.py` muestra la ruta, el modelo y si encuentra la
   clave, sin enseñarla y sin llamar a ningún modelo.
4. `python scripts/demo.py --caso DE-04` hace una consulta de prueba (de pago).

Algunos ejemplos de `.env`:

```bash
# GLM, Kimi, GPT o cualquier otro modelo por OpenRouter: cambia solo LLM_MODEL
LLM_PROVIDER=openrouter
LLM_MODEL=z-ai/glm-5.3            # o moonshotai/kimi-k3, openai/gpt-6-luna, anthropic/claude-haiku-5.5…
OPENROUTER_API_KEY=…

# Gemini con una clave de Google AI Studio
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.1-flash-lite
GEMINI_API_KEY=…
```

**OpenRouter** da acceso a cientos de modelos con una sola clave. Copia el ID
exacto de [openrouter.ai/models](https://openrouter.ai/models): tiene la forma
`proveedor/modelo` y no siempre coincide con el de la API directa
(`anthropic/claude-haiku-5.5` lleva punto; en Anthropic es `claude-haiku-5-5`).
Cada modelo admite parámetros distintos, así que antes de consultar el repo lee
el catálogo público de OpenRouter (gratis, sin clave) y envía solo lo que ese
modelo acepta: temperatura 0 si la admite; el razonamiento apagado si se puede
apagar, o en su nivel más bajo si el modelo siempre razona (en Gemini, el
mínimo); y el límite de 180 tokens con el nombre que entienda. Un ID que no
existe se rechaza antes de gastar nada. Lo enviado queda en
`parametros_solicitados`: anótalo en tu entrega. El 12 de octubre de 2026 el
caso `DE-04` funcionó así, sin tocar nada más, con Claude Haiku 5.5, GPT-6
Luna, GPT-6.1 Sol, GLM-5.3, Kimi K3, Gemini 3.1 Flash-Lite, DeepSeek V4 Flash,
Qwen3.8 Flash, Llama 4 Maverick y Grok 4.7. El catálogo cambia: comprueba tu
modelo con `demo.py` antes de evaluar.

**Google** ofrece dos caminos distintos. Con una clave de Google AI Studio usas
`gemini`: sirve para consultar Gemini con `demo.py` y con las condiciones
`claude_prompt` y `remoto_basico`. El ajuste gestionado y las condiciones
`gemini_base` y `gemini_ajustado` necesitan Google Cloud (`google`): un proyecto
del laboratorio y `gcloud auth application-default login`; ahí no se usa clave.

Cada ruta tiene su propio formato de ID y algunos parámetros propios
(temperatura admitida, razonamiento, nombre del límite de salida). El repo ya
los resuelve para el modelo del curso y para OpenRouter; si otro modelo rechaza
un parámetro, ajústalo con `LLM_TEMPERATURE` y `LLM_EXTRA_BODY` sin tocar el
código. [Todos los detalles](docs/modelos-remotos.md).

### 2. Comprobar y hacer la primera consulta

```bash
python scripts/preflight.py                # ruta, modelo y qué claves hay, sin mostrarlas
python scripts/demo.py --caso DE-04            # ruta de .env
python scripts/demo.py --caso DE-04 --fewshot  # con los cuatro ejemplos fijados
python scripts/demo.py --proveedor openrouter --modelo moonshotai/kimi-k3 --caso DE-04
```

`--proveedor` y `--modelo` prueban otra ruta sin tocar `.env` (`LLM_MODEL`
solo se aplica al proveedor de `LLM_PROVIDER`, así que con otro proveedor
indica `--modelo`). Cada consulta guarda la respuesta original, la ruta, el
modelo devuelto, el proveedor que respondió, el identificador de la petición,
el motivo de parada, los tokens, la latencia y el coste. El coste sale de
`usage.cost` cuando la ruta lo informa (OpenRouter) o de la tarifa que declares
en `LLM_PRECIO_*`; sin tarifa queda `null`, que significa no medido, no gratis.

Si algo falla:

| Mensaje | Qué hacer |
|---|---|
| `Falta …_API_KEY` | Pon la clave de esa ruta en `.env`. |
| `… no está en el catálogo de OpenRouter` | Copia el ID exacto de openrouter.ai/models. |
| `… HTTP 401` o `403` | Clave incorrecta o sin acceso a ese modelo. |
| `… HTTP 402` o `429` | Sin saldo o con límite de uso; no se reintenta solo. |
| `… HTTP 400`, o `404` en OpenRouter | El modelo no admite un parámetro enviado: revisa `LLM_TEMPERATURE` y `LLM_EXTRA_BODY`. |
| `finish_reason: length` y JSON cortado | El modelo gastó los 180 tokens (a menudo razonando). Cuenta como fallo; elige otro modelo o declara un experimento distinto. |

Más casos en las [preguntas frecuentes](docs/preguntas-frecuentes.md).

### 3. Evaluar con el mismo test de S7

```bash
python scripts/evaluar_remotos.py --condicion claude_prompt --split test \
  --salida resultados/remotos/claude_prompt.jsonl
python scripts/evaluar_remotos.py --condicion remoto_basico --split test \
  --salida resultados/remotos/remoto_basico.jsonl
python scripts/resumir.py --carpeta resultados/remotos --destino resultados/remotos/informe
```

`claude_prompt` es la condición del ejercicio: Claude con el prompt básico y los
cuatro ejemplos fijados de S7, por la ruta de `.env`. `remoto_basico` prueba
cualquier ruta con el prompt básico. El informe muestra qué modelo y qué ruta se
usaron en cada condición. Evaluar el test hace 30 llamadas de pago a tu cuenta.

### 4. Ajuste gestionado de Gemini

```bash
python scripts/convertir_gemini.py
python scripts/gemini_tuning.py preparar --proyecto ID_PROYECTO --bucket BUCKET
```

La conversión y `preparar` son locales: convierten train y validación a
`systemInstruction` + `contents` y muestran la especificación del trabajo (sin
subir el test). Ejecutar el ajuste necesita el laboratorio de Google Cloud del
curso: proyecto, permisos, bucket y presupuesto. Después, `subir-datos`, `crear`
y `estado` gestionan un trabajo real, y `evaluar_remotos.py --condicion
gemini_base` / `gemini_ajustado` comparan base y ajustado. Una llamada de
inferencia a Gemini por otra ruta no demuestra un ajuste.
[Procedimiento](docs/modelos-remotos.md#ajuste-real-de-gemini).

## Resultados de referencia

Ejecución del **2 de octubre de 2026** sobre los 30 casos de test, con datos
sintéticos y sin reparar respuestas ([JSON](docs/resultados-referencia.json)):

| Condición | Ruta | Categoría y prioridad correctas |
|---|---|---:|
| Qwen3-0.6B, instrucciones básicas | Local, Apple M5 | 3/30 |
| Qwen3-0.6B, few-shot | Local, Apple M5 | 10/30 |
| Qwen3-0.6B + LoRA | Local, Apple M5 | 26/30 |
| Claude Haiku 4.5, prompt + few-shot | OpenRouter | 24/30 |
| Gemini 3.1 Flash-Lite, prompt básico (demo adicional) | OpenRouter | 26/30 |
| Gemini base y ajustado | Google Cloud | Pendiente |
| Claude Haiku 5.5, ruta actual del curso | API de Anthropic | Pendiente |

Son cifras de una ejecución identificada, con las versiones de dependencias de
entonces: no son una garantía ni un ranking universal, y tu entorno debe
evaluarse de nuevo. Treinta casos sintéticos no deciden qué modelo es mejor en
general. LoRA no tiene obligación de ganar.

## Ejercicios

- **S7 · Adaptar un modelo abierto:** ajustar Qwen3-0.6B, comparar básico,
  few-shot y LoRA con datos separados, conservar las respuestas originales,
  publicar el adaptador con su model card y demostrar su recarga desde el Hub.
- **S8 · Modelos cerrados:** ajuste real de Gemini en Google Cloud con los datos
  de S7, Claude Haiku 5.5 como referencia con prompting y una tabla de seis
  condiciones con calidad, latencia, tokens y coste.

[Enunciados, entregables y modalidades según recursos](docs/ejercicio.md).
Se valora una comparación reproducible y honesta, no que gane un modelo.

## Trabajar con un asistente de IA

Puedes abrir el repo con Claude Code, Codex, Cursor u otro asistente y
preguntarle en lenguaje natural. El asistente lee [AGENTS.md](AGENTS.md) (también
enlazado como `CLAUDE.md`): ahí tiene el contexto de las clases, el mapa de
archivos y funciones, el contrato del experimento y cómo explicar las cosas con
sencillez. Las respuestas a las dudas habituales están en
[docs/preguntas-frecuentes.md](docs/preguntas-frecuentes.md).

Ejemplos de preguntas:

- «¿Qué hace `validate_answer` y por qué esta respuesta cuenta como error?»
- «Explícame paso a paso qué pasa con una fila de `train.jsonl` hasta que entrena el LoRA.»
- «¿Cómo cambio el `.env` para usar OpenAI (u Ollama) en lugar de Claude?»
- «¿Qué diferencia hay entre few-shot y LoRA en el código?»
- «Me sale `Falta ANTHROPIC_API_KEY`: ¿qué hago?»
- «¿Qué tengo que entregar en el ejercicio de S8?»

Buenas prácticas: no pegues claves en el chat; pide al asistente que te
explique antes de ejecutar algo que cueste dinero, suba datos o entrene; y
comprueba lo que te diga en el código, que es la fuente de verdad.

## Documentación

| Documento | Para qué sirve |
|---|---|
| [docs/preguntas-frecuentes.md](docs/preguntas-frecuentes.md) | Cómo se hace cada cosa, con comandos y dónde mirar en el código |
| [docs/guia-codigo.md](docs/guia-codigo.md) | Orden de lectura del código, de un ejemplo de datos al informe |
| [docs/experimento.md](docs/experimento.md) | Protocolo: datos, condiciones, métricas y reglas que no se rompen |
| [docs/modelos-remotos.md](docs/modelos-remotos.md) | Proveedores, `.env`, ajustes, coste, evaluación remota y Gemini |
| [docs/ejercicio.md](docs/ejercicio.md) | Ejercicios de S7 y S8 |
| [docs/resultados-referencia.json](docs/resultados-referencia.json) | Cifras de referencia fechadas y sus limitaciones |
| [AGENTS.md](AGENTS.md) | Guía para asistentes de IA |
| [SECURITY.md](SECURITY.md) | Límites de seguridad |
| [docs/publicacion.md](docs/publicacion.md) | Controles de seguridad, dependencias y verificación |

Los notebooks recorren las clases por etapas con la inferencia y el entrenamiento
desactivados por defecto: `notebooks/S7-LoRA.ipynb` (datos, template, few-shot,
original, LoRA, recarga, evaluación, isbetter.ai) y
`notebooks/S8-Gemini-Claude.ipynb` (proveedores, conversión, especificación del
ajuste y comparación).

## Comprobaciones, seguridad y licencia

```bash
python -m unittest discover -s tests -v
python -m compileall -q curso scripts tests
python scripts/preflight.py
```

La CI ejecuta los contratos sin descargar pesos ni llamar a proveedores: un
resultado correcto no certifica una inferencia remota ni un entrenamiento.
Usa solo datos sintéticos, mantén las claves en `.env` y no expongas la API local
a la red. El dataset tiene diversidad limitada y respuestas objetivo genéricas:
la utilidad del texto se revisa aparte de las etiquetas.

Código y datos sintéticos bajo licencia MIT. Qwen, los modelos de cada proveedor
y las dependencias conservan sus propias licencias y condiciones.

# Preguntas frecuentes · cómo se hace cada cosa

Respuestas cortas a las dudas habituales de S7 (modelos abiertos) y S8 (modelos
cerrados y APIs). Cada una indica el comando y dónde mirar en el código. Los
comandos se ejecutan desde la raíz del repo con el entorno activado (o con
`.venv/bin/python` en lugar de `python`).

Si trabajas con un asistente de IA, puedes preguntarle cualquiera de estas
cosas con tus palabras: usa este documento y el código como fuente.

## Empezar

### ¿Qué es este repo y por dónde empiezo?

Es la práctica de S7 y S8: convertir una incidencia ficticia en un JSON con
`categoria`, `prioridad` y `respuesta`, primero con un modelo abierto en tu
equipo (S7) y después con modelos por API (S8). Empieza por el
[README](../README.md), instala el entorno y ejecuta `python scripts/preflight.py`.
Para leer el código en orden, sigue [guia-codigo.md](guia-codigo.md).

### ¿Cómo instalo el entorno?

```bash
uv venv --python 3.12 .venv
uv pip sync --python .venv requirements-training.lock
source .venv/bin/activate
cp .env.example .env
```

Sin uv: `python -m venv .venv`, activa el entorno y
`python -m pip install --require-hashes -r requirements-training.lock`.
`requirements.lock` es la versión ligera, sin PyTorch, para ejecutar solo los tests.

### ¿Cómo compruebo que todo está bien?

```bash
python scripts/preflight.py
python -m unittest discover -s tests -v
```

El preflight muestra las particiones, si el experimento sigue congelado, qué
librerías hay, la ruta remota del `.env` (`ruta_remota`), qué claves están
puestas (`credenciales`, solo verdadero/falso) y si existe un adaptador. No hace
ninguna inferencia. Código: `scripts/preflight.py:main`.

### ¿Qué no viene en el repo y por qué?

Los pesos del modelo, los adaptadores (`modelos/`), los resultados
(`resultados/`), tu `.env` y las carpetas de trabajo local (`material/`). Los
pesos se descargan con un script, el adaptador se obtiene al entrenar y las
claves son personales. Es normal que esas carpetas no existan en una copia nueva.

## Datos y reglas

### ¿Qué contiene un ejemplo del dataset?

Cada línea de `datos/train.jsonl` es un JSON con `id`, `familia`, `incidencia`
y `objetivo`. Solo `incidencia` (lo que escribe el usuario) y `objetivo` (la
respuesta correcta) entran en la conversación; `id` y `familia` sirven para
identificar y separar casos. Código: `curso/common.py:read_jsonl`, `messages`.

### ¿Por qué hay train, validación, test y desarrollo?

Train (240) enseña al LoRA y aporta los ejemplos few-shot. Validación (30) mide
la pérdida durante el entrenamiento sin actualizar pesos. Test (30) se usa al
final para evaluar, nunca para ajustar. Desarrollo (6) son las consultas de las
demos. Una misma familia de frases nunca cruza particiones.
Código: `curso/common.py:validate_splits`. Protocolo: [experimento.md](experimento.md).

### ¿Dónde están las reglas de categoría y prioridad?

En `prompts/basico.txt`. Es el mismo texto para todos los modelos, en
entrenamiento e inferencia. El validador usa las mismas categorías y prioridades:
`curso/common.py:CATEGORIES`, `PRIORITIES`.

### ¿Puedo cambiar los datos o el prompt?

No en este experimento: `config/experimento-congelado.json` guarda sus hashes y
los scripts se detienen con *«Datos o protocolo cambiaron después de congelar el
experimento»*. Si quieres probar otra variante, es otro experimento: crea datos,
manifiesto y resultados propios y no los mezcles con la referencia.
Código: `curso/common.py:assert_frozen`.

## Modelo abierto en tu equipo (S7)

### ¿Cómo descargo Qwen?

```bash
python scripts/descargar_modelo.py
```

Descarga `Qwen/Qwen3-0.6B` en una revisión fija (unos 1,5 GB) a la caché de
Hugging Face. Después puedes trabajar sin conexión con `HF_HUB_OFFLINE=1`.

### ¿Cómo le pregunto algo al modelo original?

```bash
python scripts/demo.py --proveedor qwen --caso DE-04
```

`--caso` elige una incidencia de `datos/desarrollo.jsonl` (DE-01 a DE-06).
La respuesta se valida y se imprime; `--salida archivo.json` la guarda.
Código: `curso/qwen.py:load_qwen`, `render`, `generate`.

### ¿Qué es few-shot y cómo lo pruebo?

Añadir ejemplos resueltos al contexto de cada consulta. Aquí son cuatro registros
fijos de train (`prompts/fewshot-ids.json`). No cambia ningún peso: los ejemplos
viajan en cada petición y la encarecen.

```bash
python scripts/demo.py --proveedor qwen --caso DE-04 --fewshot
```

Código: `curso/common.py:messages(incident, fewshot=True)`.

### ¿Cómo entreno el LoRA? ¿Cuánto tarda?

```bash
python scripts/entrenar_lora.py --salida modelos/lora-soporte
```

Entrena solo dos matrices pequeñas (A y B) en 7 proyecciones de cada capa; el
modelo base queda congelado. Receta fija: 2 épocas, r=8, alpha=16, dropout 0,05,
batch 2 y acumulación 4, es decir, 60 pasos de optimizador. El tiempo depende
del equipo: unos 11 minutos en un Mac con Apple M5, más con CPU. Usa CUDA, MPS
o CPU según disponibilidad. Código: `scripts/entrenar_lora.py:main`
(`dataset`, `SFTConfig`, `LoraConfig`).

### ¿Qué hace `labels = -100`?

Excluye de la pérdida los tokens de las instrucciones y de la incidencia: el
modelo los lee como contexto, pero solo aprende a predecir los tokens de la
respuesta objetivo. Código: `scripts/entrenar_lora.py`, función `dataset` dentro
de `main`.

### ¿Qué se guarda al entrenar y cómo lo recargo?

La carpeta de `--salida` contiene `adapter_model.safetensors` (las matrices
aprendidas, unos 20 MB), `adapter_config.json` (cómo aplicarlas), el tokenizer y
`entrenamiento.json` (receta, métricas y versiones). Para usarlo se carga el
modelo base y encima el adaptador:

```bash
python scripts/demo.py --proveedor qwen --caso DE-04 --adaptador modelos/lora-soporte
```

Código: `curso/qwen.py:load_qwen(adapter)`. Consultar no entrena: no hay optimizador.

### ¿Cómo comparo original y LoRA en isbetter.ai?

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/servir_comparacion.py --adaptador modelos/lora-soporte
```

En isbetter.ai, proveedor Local `http://localhost:1235/v1`; elige
`qwen-original` y `qwen-lora-soporte`. En Advanced instructions pega
`prompts/basico.txt` y lee el JSON en Output. Es un solo Qwen que activa o
desactiva el adaptador; las peticiones van en cola. Código:
`scripts/servir_comparacion.py` (`ComparisonEngine`, `ComparisonHandler`).

### ¿Cómo uso LM Studio con el repo?

En LM Studio descarga `Qwen3-0.6B-Q8_0.gguf` del repositorio
`Qwen/Qwen3-0.6B-GGUF`, cárgalo y arranca el servidor local. Después:

```bash
python scripts/demo.py --proveedor local --caso DE-01
```

Usa `LOCAL_BASE_URL` (por defecto `http://localhost:1234/v1`) y descubre el
modelo si solo hay uno cargado; si no, indica `--modelo` o `LOCAL_MODEL_ID`.
Con `--fewshot`, usa `--local-api openai`. Código: `curso/remote.py:chat_local`.

### ¿Cuánta memoria necesito?

Para los pesos: parámetros × bits ÷ 8. Qwen3-0.6B en BF16 ocupa unos 1,5 GB;
en GGUF Q8_0, unos 0,64 GB. Un modelo de 8B ocupa 16 GB en 16 bits, 8 GB en 8 bits
y 4 GB en 4 bits. Hay que sumar el contexto, el runtime y el sistema.

### ¿Cómo publico mi adaptador en Hugging Face?

```bash
python scripts/publicar_adaptador.py --repo TU_USUARIO/NOMBRE_NUEVO \
  --adaptador modelos/lora-soporte
```

Crea un repo privado nuevo en tu cuenta (no sobrescribe uno existente), sube el
adaptador, `entrenamiento.json` y el `README.md` de la carpeta, y lo recarga desde
ese commit para comprobarlo. Ese `README.md` es la model card: el entrenador no
lo crea, escríbelo antes de publicar (modelo base y revisión, datos, configuración,
resultados, licencia y límites). Necesitas haber iniciado sesión en Hugging Face
fuera del repo.
Código: `scripts/publicar_adaptador.py:main`.

## Modelos por API (S8)

### ¿Cómo uso Claude?

La ruta del curso es Claude Haiku 5.5 por la API de Anthropic. En `.env`:

```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-haiku-5-5
ANTHROPIC_API_KEY=tu_clave
```

```bash
python scripts/demo.py --caso DE-04
```

El repo usa el SDK oficial, sin reintentos, con 180 tokens de salida y sin
razonamiento extendido. No envía `temperature`, porque Haiku 5.5 solo admite la
predeterminada. Código: `curso/remote.py:chat_anthropic`.

### ¿Cómo cambio a OpenAI, OpenRouter, Gemini, Groq u Ollama?

Cambia `LLM_PROVIDER`, `LLM_MODEL` y pon la clave de esa ruta en `.env`:

| Quiero usar | `LLM_PROVIDER` | Además |
|---|---|---|
| OpenAI | `openai` | `OPENAI_API_KEY` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY`; ID `proveedor/modelo` |
| Gemini con clave | `gemini` | `GEMINI_API_KEY` |
| Groq, Mistral, DeepSeek… | `compatible` | `COMPAT_BASE_URL` (https) y `COMPAT_API_KEY` |
| Ollama o vLLM en tu equipo | `compatible` | `COMPAT_BASE_URL=http://localhost:11434/v1` |
| LM Studio | `local` | Nada |

Comprueba con `python scripts/preflight.py`. Ejemplos completos en
[modelos-remotos.md](modelos-remotos.md). Código: `curso/remote.py:route`, `chat`.

### ¿Cómo pruebo otro proveedor sin cambiar el `.env`?

```bash
python scripts/demo.py --proveedor openrouter --modelo google/gemini-3.1-flash-lite --caso DE-04
```

`--proveedor` y `--modelo` tienen prioridad. `LLM_MODEL` solo se aplica al
proveedor de `LLM_PROVIDER`, así que con otro proveedor indica `--modelo`.

### El proveedor devuelve un error 400 por un parámetro. ¿Qué hago?

Cada modelo acepta parámetros distintos. Ajústalos en `.env` para la ruta de
`LLM_PROVIDER`, sin tocar código:

- `LLM_TEMPERATURE=omitir` deja de enviar la temperatura; un número la fija.
- `LLM_EXTRA_BODY` añade o cambia campos de la petición. Por ejemplo,
  `{"reasoning_effort": "minimal"}` en OpenAI o
  `{"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}` en
  Claude Opus o Sonnet 5.5. Un valor `null` retira un campo.

Anota el cambio junto a tus resultados: forma parte del experimento.
Código: `curso/remote.py:route_settings`, `merged`.

### ¿Cómo se calcula el coste?

OpenRouter informa el importe en `usage.cost`. Anthropic, OpenAI y Google
devuelven tokens, no importes: el repo multiplica los tokens por la tarifa que
declares en `LLM_PRECIO_ENTRADA_USD_MTOK` y `LLM_PRECIO_SALIDA_USD_MTOK` (USD por
millón) para el modelo de `LLM_MODEL`. La plantilla trae la de Claude Haiku 5.5
(0,10 y 0,50). Sin tarifa, el coste es `null`, que significa no medido.
Código: `curso/remote.py:tariff_cost`.

### ¿Cómo evalúo un modelo remoto con los 30 casos de test?

```bash
python scripts/evaluar_remotos.py --condicion claude_prompt --split test \
  --salida resultados/remotos/claude_prompt.jsonl
python scripts/resumir.py --carpeta resultados/remotos --destino resultados/remotos/informe
```

`claude_prompt` usa la ruta de `.env` con el prompt y los cuatro ejemplos fijados;
`remoto_basico`, cualquier ruta con el prompt básico. Son 30 llamadas de pago.
Si falta la clave, el script se detiene antes de llamar. Código:
`scripts/evaluar_remotos.py:main`.

### ¿Cómo convierto los datos para Gemini y preparo el ajuste?

```bash
python scripts/convertir_gemini.py
python scripts/gemini_tuning.py preparar --proyecto ID_PROYECTO --bucket BUCKET
```

Convierte train y validación a `systemInstruction` + `contents` (el rol
`assistant` pasa a llamarse `model`) y guarda los IDs en `datos/gemini/ids.json`.
`preparar` no se conecta: muestra la especificación del trabajo. Subir datos y
crear el trabajo necesitan el laboratorio de Google Cloud del curso, con
proyecto, permisos y presupuesto. Código: `curso/remote.py:gemini_record`,
`scripts/gemini_tuning.py:job_spec`, `main`.

### ¿Por qué una llamada a Gemini no es un fine-tuning?

Una llamada es inferencia: usa el modelo tal como está. Un ajuste gestionado se
demuestra con el ID del trabajo, su estado terminado, el recurso del modelo
ajustado y sus predicciones. Cambiar el prompt o añadir few-shot tampoco es
ajustar. Las condiciones `gemini_base` y `gemini_ajustado` van siempre por
Google Cloud.

## Evaluación y resultados

### ¿Cómo evalúo las tres condiciones de Qwen?

```bash
python scripts/evaluar_qwen.py --condicion qwen_base --split test --salida resultados/ensayo/qwen_base.jsonl
python scripts/evaluar_qwen.py --condicion qwen_fewshot --split test --salida resultados/ensayo/qwen_fewshot.jsonl
python scripts/evaluar_qwen.py --condicion qwen_lora --split test --adaptador modelos/lora-soporte --salida resultados/ensayo/qwen_lora.jsonl
python scripts/resumir.py --carpeta resultados/ensayo --destino resultados/ensayo/informe
```

### ¿Cómo leo el informe?

`comparacion.md` tiene una fila por condición con el modelo y la ruta usados,
los casos, JSON válido, categoría, prioridad, ambas correctas, la mediana de
tiempo y el coste. `resultados.json` conserva cada respuesta original;
`resultados.csv` es lo mismo en tabla. Lo más informativo es «ambas»: categoría y
prioridad correctas a la vez.

### ¿Por qué esta respuesta cuenta como error?

`curso/common.py:validate_answer` comprueba, por separado: que el texto entero
sea JSON (sin quitar bloques Markdown), que tenga exactamente las tres claves con
valores admitidos, y si categoría y prioridad coinciden con el objetivo. Una
categoría inventada como `cobros` o una prioridad `urgente` rompen el esquema.

### ¿Por qué no se reparan ni reintentan respuestas?

Porque mediríamos nuestra reparación, no el modelo. Un error se queda en el
denominador y se cuenta. Tampoco se usa JSON mode ni herramientas que fuercen el
formato: queremos ver si el modelo sigue las instrucciones.

### ¿Qué son los resultados de referencia?

`docs/resultados-referencia.json`: una ejecución del 2 de octubre de 2026 (Qwen
básico 3/30, few-shot 10/30 y LoRA 26/30 en ambas etiquetas; Claude Haiku 4.5 por
OpenRouter 24/30; Gemini 3.1 Flash-Lite por OpenRouter 26/30). Son de un entorno
y unas versiones concretos: tu ejecución tendrá sus propias cifras.

## Errores frecuentes

| Mensaje o síntoma | Qué significa y qué hacer |
|---|---|
| `ModuleNotFoundError` | El entorno no está activo o falta el lock. Activa `.venv` e instala `requirements-training.lock`. |
| `Falta ANTHROPIC_API_KEY` (u otra clave) | Esa ruta necesita tu clave en `.env`. Ponla allí, nunca en el código ni en un chat. |
| `Indica el modelo de … con --modelo o LLM_MODEL` | Usas un proveedor distinto del de `.env` sin decir el modelo. Añade `--modelo`. |
| `Proveedor desconocido` | Revisa `LLM_PROVIDER`: anthropic, openai, openrouter, compatible, gemini, local o google. |
| `… HTTP 401/403` | Clave inválida o sin permiso para ese modelo. |
| `… HTTP 402` o `429` | Sin saldo o con límite de uso. No se reintenta automáticamente. |
| `… HTTP 400` | Un parámetro no admitido por ese modelo. Revisa `LLM_TEMPERATURE` y `LLM_EXTRA_BODY`. |
| `Salida no vacía; usa otro --salida…` | El entrenador protege un adaptador anterior. Usa otra carpeta. |
| `Salida existente; conservar resultados` | Los evaluadores no sobrescriben resultados. Usa otro archivo. |
| `Datos o protocolo cambiaron…` | Se tocaron datos o prompts congelados. Revisa `git diff -- datos prompts config`. |
| `adaptador_local: false` | Normal antes de entrenar. |
| isbetter.ai no conecta | Servidor arrancado, URL acabada en `/v1` (1235 para el comparador, 1234 para LM Studio) y permiso del navegador. |
| Entrenamiento lento o sin memoria | Comprueba el dispositivo (CUDA, MPS o CPU) y cierra otras aplicaciones. |

## Ejercicio

### ¿Qué tengo que entregar?

S7: código, datos separados, respuestas originales de básico, few-shot y LoRA,
tabla de resultados, adaptador publicado con model card y evidencia de recarga
desde el Hub. S8: conversor, ajuste real de Gemini, seis condiciones con el mismo
test, respuestas originales, tabla con calidad, latencia, tokens y coste, y una
elección justificada. Detalle en [ejercicio.md](ejercicio.md).

### ¿Y si no tengo GPU o acceso al laboratorio?

Puedes entregar la preparación completa: datos comprobados, código,
configuración, comandos y la explicación de la limitación, con las métricas no
obtenidas marcadas como pendientes. No inventes resultados ni contrates recursos
de pago para completarla.

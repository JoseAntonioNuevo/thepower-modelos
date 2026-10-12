# Modelos remotos · cualquier proveedor desde `.env`

La práctica local no necesita claves externas. Las rutas remotas requieren una
cuenta propia y pueden generar costes; no vienen recursos cloud creados.

En S8 la ruta del curso es **Claude Haiku 5.5 por la API de Anthropic**. El código
no está atado a ella: `curso/remote.py:chat` lee la ruta del `.env` y llama al
adaptador correspondiente. Cambiar de proveedor es cambiar dos variables.

## Elegir la ruta

```bash
# .env
LLM_PROVIDER=anthropic          # anthropic | openai | openrouter | compatible | gemini | local | google
LLM_MODEL=claude-haiku-5-5      # ID del modelo en esa ruta
ANTHROPIC_API_KEY=…             # solo la clave de la ruta que uses
```

`scripts/demo.py` y `scripts/evaluar_remotos.py` usan esa ruta cuando no indicas
`--proveedor`. Con `--proveedor` y `--modelo` puedes probar otra sin tocar `.env`;
`LLM_MODEL` solo se aplica al proveedor de `LLM_PROVIDER`.

| `LLM_PROVIDER` | Servicio | Credencial | Formato del ID |
|---|---|---|---|
| `anthropic` | Messages API de Anthropic, SDK oficial | `ANTHROPIC_API_KEY` | `claude-haiku-5-5` |
| `openai` | Chat Completions de OpenAI | `OPENAI_API_KEY` (y `OPENAI_BASE_URL` opcional) | El que muestre tu cuenta |
| `openrouter` | Interfaz común para cientos de modelos (Claude, GPT, Gemini, GLM, Kimi, DeepSeek, Qwen, Llama, Grok…) | `OPENROUTER_API_KEY` | `proveedor/modelo` exacto de su catálogo |
| `compatible` | Cualquier API compatible con Chat Completions | `COMPAT_BASE_URL` y `COMPAT_API_KEY` si la pide | El del servidor elegido |
| `gemini` | Gemini API de Google con clave | `GEMINI_API_KEY` | `gemini-3.1-flash-lite` |
| `local` | LM Studio en tu equipo | Ninguna | Lo descubre `/models` o `LOCAL_MODEL_ID` |
| `google` | Vertex AI en el proyecto del laboratorio | `GOOGLE_CLOUD_PROJECT` + credenciales ADC | Modelo base o recurso ajustado |

Ejemplos de `.env` para otras rutas:

```bash
# OpenAI
LLM_PROVIDER=openai
LLM_MODEL=ID_DE_TU_CUENTA
OPENAI_API_KEY=…

# Cualquier modelo de OpenRouter: copia el ID exacto de openrouter.ai/models
LLM_PROVIDER=openrouter
LLM_MODEL=z-ai/glm-5.3      # o anthropic/claude-haiku-5.5, moonshotai/kimi-k3, openai/gpt-6-luna…
OPENROUTER_API_KEY=…

# Gemini con una clave de Google AI Studio (no sirve para el ajuste en Google Cloud)
LLM_PROVIDER=gemini
LLM_MODEL=gemini-3.1-flash-lite
GEMINI_API_KEY=…

# Un servidor compatible remoto (Groq, Mistral, DeepSeek…): https obligatorio
LLM_PROVIDER=compatible
LLM_MODEL=ID_DEL_SERVIDOR
COMPAT_BASE_URL=https://api.groq.com/openai/v1
COMPAT_API_KEY=…

# Un modelo abierto en tu equipo con Ollama
LLM_PROVIDER=compatible
LLM_MODEL=qwen3:0.6b
COMPAT_BASE_URL=http://localhost:11434/v1
```

Comprueba qué ruta y credenciales ve el código, sin imprimir claves:

```bash
python scripts/preflight.py
```

## Qué envía cada ruta

Todas comparten el protocolo de S7: las mismas instrucciones (`prompts/basico.txt`),
los mismos cuatro ejemplos few-shot cuando se piden, como máximo **180 tokens de
salida**, JSON pedido como texto, sin reparar la respuesta y **sin reintentos**.
Un fallo queda registrado como error en el denominador.

| Ruta | Instrucciones | Límite de salida | Temperatura por defecto | Razonamiento |
|---|---|---|---|---|
| `anthropic` | `system` | `max_tokens` | No se envía | `thinking: disabled` |
| `openai` | mensaje `system` | `max_completion_tokens` | No se envía | El del modelo |
| `openrouter` | mensaje `system` | `max_tokens` (o `max_completion_tokens` si el modelo solo admite ese) | 0 si el modelo la admite; si no, no se envía | Desactivado si se puede; el nivel más bajo si es obligatorio; mínimo en `google/…` |
| `compatible` | mensaje `system` | `max_tokens` (`COMPAT_TOKEN_FIELD` lo cambia) | 0 | El del modelo |
| `gemini` y `google` | `systemInstruction` | `maxOutputTokens` | 0 | `thinkingLevel: MINIMAL` |
| `local` | `system_prompt` o mensaje `system` | `max_output_tokens` o `max_tokens` | 0 | `reasoning: off` |

Claude Haiku 5.5 no admite una temperatura distinta de la predeterminada, así que
sus respuestas no son deterministas aunque el resto del protocolo sea fijo. Con el
razonamiento activado, los tokens de razonamiento también cuentan para el límite
de 180 y una respuesta puede cortarse antes del JSON.

Cada registro guarda la respuesta original, `ruta`, `model` devuelto, `provider`
que respondió, `request_id`, motivo de parada, tokens, latencia y los parámetros
solicitados, nunca las claves ni los mensajes completos.

### OpenRouter: cualquier modelo de su catálogo

Con `openrouter` sirve cualquier modelo de [openrouter.ai/models](https://openrouter.ai/models)
sin tocar el código. Cada modelo admite parámetros distintos: Claude Haiku 5.5 y
los GPT de razonamiento rechazan la temperatura, algunos GLM o Grok razonan
siempre y modelos como Llama no razonan. Por eso, antes de consultar,
`curso/remote.py:openrouter_model` lee el catálogo público de OpenRouter (gratis,
sin clave ni inferencia; una vez por modelo y ejecución) y
`openrouter_settings` construye la petición solo con lo que ese modelo admite:

- Temperatura 0 si la admite; si no, no se envía.
- Razonamiento desactivado (`enabled: false`) si se puede apagar; si es
  obligatorio, el nivel más bajo de su lista (`supported_efforts`); en
  `google/…`, `effort: minimal`, como `thinkingLevel` MINIMAL en las rutas de Google.
- Límite de 180 tokens como `max_tokens`, o `max_completion_tokens` si el modelo
  solo acepta ese nombre.
- `provider.require_parameters: true`: OpenRouter solo usa proveedores que
  respeten todo lo enviado, en lugar de ignorar un parámetro en silencio.

Un ID que no existe en el catálogo se rechaza antes de gastar una llamada;
`evaluar_remotos.py` lo comprueba antes del lote. Un sufijo de enrutado como
`:nitro` usa la ficha del modelo sin sufijo. `LLM_TEMPERATURE` y
`LLM_EXTRA_BODY` siguen mandando sobre estos valores: si fijas una temperatura
que el modelo no admite, OpenRouter responde 404 porque ningún proveedor la acepta.

Un modelo que siempre razona gasta parte de los 180 tokens pensando. Si
`finish_reason` es `length` y el JSON llega cortado o vacío, el caso cuenta como
fallo: elige otra variante o declara un experimento distinto con otro límite.
El 12 de octubre de 2026 el caso `DE-04` respondió con JSON válido y etiquetas
correctas, con esta lógica y sin ajustes, en Claude Haiku 5.5, GPT-6 Luna,
GPT-6.1 Sol, GLM-5.3, Kimi K3, Gemini 3.1 Flash-Lite, DeepSeek V4 Flash, Qwen3.8
Flash, Llama 4 Maverick y Grok 4.7. Es una comprobación de un caso, no una
evaluación: el catálogo y los modelos cambian.

### Ajustes por proveedor sin tocar el código

Para la ruta de `LLM_PROVIDER`:

- `LLM_TEMPERATURE`: un número la envía; `omitir` no la envía; vacío usa el valor
  del adaptador.
- `LLM_EXTRA_BODY`: un objeto JSON que se mezcla con la petición. `null` retira un
  campo. No puede cambiar `model`, `messages`, `system` ni `contents`.

```bash
# Un modelo de OpenAI con razonamiento: que no consuma los 180 tokens pensando
LLM_EXTRA_BODY={"reasoning_effort": "minimal"}
# Claude Opus 5.5 o Sonnet 5.5, que no admiten thinking desactivado
LLM_EXTRA_BODY={"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}
# Un Gemini Pro, que no admite thinkingLevel MINIMAL
LLM_EXTRA_BODY={"generationConfig": {"thinkingConfig": {"thinkingLevel": "LOW"}}}
```

Cambiar estos ajustes cambia el experimento: anótalo junto a los resultados.

### Coste

OpenRouter informa el importe en `usage.cost`. Anthropic, OpenAI, la Gemini API
y las APIs compatibles devuelven tokens, no importes: el coste se calcula con la
tarifa que declares en `LLM_PRECIO_ENTRADA_USD_MTOK` y `LLM_PRECIO_SALIDA_USD_MTOK`
(USD por millón de tokens) para el modelo de `LLM_MODEL`. Si cambias de modelo,
cambia también la tarifa: la de Claude aplicada a otro modelo da un coste falso.
Sin tarifa, el coste queda `null`, que significa no medido, no gratis. Google
Cloud (Vertex AI) y LM Studio quedan siempre `null`. La plantilla trae la tarifa publicada de
`claude-haiku-5-5` para prompts de hasta 100.000 tokens: verifica el precio
vigente antes de presupuestar.

## Probar y evaluar

```bash
# Un caso de desarrollo con la ruta de .env (Claude Haiku 5.5 por defecto)
python scripts/demo.py --caso DE-04
python scripts/demo.py --caso DE-04 --fewshot

# La misma consulta por otra ruta, sin cambiar .env
python scripts/demo.py --proveedor openrouter --modelo google/gemini-3.1-flash-lite --caso DE-04

# Condición del ejercicio: Claude con el prompt y los ejemplos fijados (30 casos de test)
python scripts/evaluar_remotos.py --condicion claude_prompt --split test \
  --salida resultados/remotos/claude_prompt.jsonl

# Condición adicional: cualquier ruta con el prompt básico
python scripts/evaluar_remotos.py --condicion remoto_basico --split test \
  --salida resultados/remotos/remoto_basico.jsonl

python scripts/resumir.py --carpeta resultados/remotos --destino resultados/remotos/informe
```

La evaluación de test llama 30 veces al proveedor y consume saldo de tu cuenta.
El informe indica, en cada condición, qué modelo y qué ruta se usaron. Si usas
un modelo que no es Claude en `claude_prompt`, el informe lo mostrará así:
nómbralo en tu entrega.

`gemini_base` y `gemini_ajustado` usan siempre la ruta `google` (Vertex AI),
porque el ajuste gestionado y su modelo resultante viven en Google Cloud.
`gemini_openrouter_demo` reproduce la demo histórica de Gemini por OpenRouter y
no sustituye a esa pareja.

## Preparación de Gemini sin crear recursos

```bash
python scripts/convertir_gemini.py
python scripts/gemini_tuning.py preparar \
  --proyecto ID_PROYECTO_CURSO --bucket BUCKET_CURSO
```

Los nombres son marcadores. `preparar` es offline. Train/validación se convierten
a contents/parts; test queda separado. `ids.json` mantiene correspondencia.

## Ajuste real de Gemini

Necesitas proyecto identificado, permisos, facturación/presupuesto, bucket,
cuota y acceso al modelo. Configura .env.example y autentica tu cuenta con
Google. No tomar un proyecto personal por defecto como laboratorio.

La receta usa europe-west4 para tuning y eu para serving. `eu` es una
multirregión: el código la consulta en `aiplatform.eu.rep.googleapis.com`. El
modelo ajustado se consulta en la región que figura en su recurso
(`projects/…/locations/europe-west4/…`). Verifica soporte, regiones y tarifas
actuales. Estima tokens, épocas, almacenamiento e inferencia.
El presupuesto del script es una comprobación previa, no un tope de Google.

Solo con recursos/gasto preparados usa `subir-datos` y `crear`, indicando
`--proyecto`, `--bucket`, `--presupuesto-usd` y `--coste-estimado-usd`.
Conserva el ID. `estado --proyecto ... --job ...` consulta ese job; `crear` de
nuevo inicia otro. La comparación base/ajustado mantiene servicio, datos,
instrucciones y thinking MINIMAL. Evidencia: job terminado, modelo y predicciones.

## Publicar el adaptador Qwen

Revisa archivos y utiliza un nombre nuevo en tu cuenta:

```bash
python scripts/publicar_adaptador.py \
  --repo TU_USUARIO/NOMBRE_NUEVO --adaptador modelos/lora-soporte
```

Verifica el usuario, crea un repo privado nuevo, sube pesos/configuración/model
card/receta y prueba recarga por commit. Configura la credencial fuera del repo.
No compartir claves ni cachés.

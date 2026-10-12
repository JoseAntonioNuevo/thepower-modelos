---
tags: [thePower, S8, API, OpenRouter, Gemini, Claude]
---

# Tres rutas para pedir una respuesta

**Una aplicación puede llegar a un modelo por una interfaz común, como OpenRouter, o por la API propia del proveedor. Todas llevan los mismos elementos, pero con distinto nombre y lugar.**

## Las rutas de esta clase

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    APP["🧑‍💻 <b>Tu aplicación</b><br/>o el terminal"]
    APP -->|"Authorization: Bearer"| OR["🛣️ <b>OpenRouter</b><br/>interfaz común<br/>/api/v1/chat/completions"]
    APP -->|"x-goog-api-key"| GA["🛣️ <b>API nativa de Google</b><br/>models/{id}:generateContent"]
    APP -->|"x-api-key<br/>+ anthropic-version"| MA["🛣️ <b>Messages API de Anthropic</b><br/>/v1/messages"]
    OR --> M1["🔓 <b>Modelo abierto</b><br/>servido por API<br/>ID por fijar para la edición"]
    GA --> M2["🔒 <b>Gemini</b><br/>gemini-3.1-flash-lite"]
    MA --> M3["🔒 <b>Claude Haiku 5.5</b><br/>claude-haiku-5-5"]
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    class APP decision
    class OR,GA,MA herramienta
    class M1 ajuste
    class M2,M3 base
```

> [!warning] OpenRouter no es una familia de modelos
> Une a muchos proveedores bajo una misma interfaz. Reduce el trabajo de integración, pero **no iguala sus funciones**: cada proveedor admite parámetros distintos. Nuestro código añade `provider.require_parameters` para que OpenRouter solo use proveedores que acepten los parámetros enviados.

## Qué enviamos y qué comprobamos

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    A["🧑‍💻 Aplicación"] --> P["📤 <b>Petición</b><br/><br/>Endpoint y API key<br/>ID del modelo<br/>Instrucciones y mensajes<br/>Límite de salida"]
    P --> API["🛣️ API del<br/>proveedor"]
    API --> M["🧠 Modelo"]
    M --> R["📥 <b>Respuesta</b><br/><br/>Bloques de texto<br/>Motivo de parada<br/>Tokens usados<br/>Latencia y coste"]
    R --> A
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    class A decision
    class P datos
    class API herramienta
    class M,R base
```

> [!tip] La primera prueba: «Responde exactamente: API lista.»
> Comprueba que la clave, el endpoint y el ID funcionan. **No mide la calidad**: para eso hay que probar nuestra tarea.

## Misma llamada, tres estructuras

Los mismos elementos cambian de nombre y de lugar según la ruta:

| Elemento | Claude · Messages API | Gemini · API nativa | OpenRouter |
|---|---|---|---|
| **Endpoint** | `/v1/messages` | `models/{id}:generateContent` | `/api/v1/chat/completions` |
| **Clave** | `x-api-key` + `anthropic-version` | `x-goog-api-key` | `Authorization: Bearer` |
| **Modelo** | `model` | en la URL | `model: proveedor/modelo` |
| **Instrucciones** | `system` | `systemInstruction` | mensaje `role: system` |
| **Entrada** | `messages` | `contents → parts` | `messages` |
| **Límite** | `max_tokens` (obligatorio) | `generationConfig.maxOutputTokens` | `max_tokens` |
| **Texto** | `content[].text` | `candidates → content → parts → text` | `choices → message → content` |
| **Consumo** | `usage` | `usageMetadata` | `usage` (incluye `cost`) |
| **Parada** | `stop_reason` | `finishReason` | `finish_reason` y `native_finish_reason` |
| **Errores típicos** | 401 · 429 · 529 | 400 · 403 · 429 | 401 · 402 · 429 |

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart TB
    subgraph CL["🟣 Claude · Messages API"]
        direction LR
        c1["<b>POST</b> api.anthropic.com/v1/messages<br/>x-api-key · anthropic-version"] --> c2["<b>model</b><br/><b>system</b> instrucciones<br/><b>messages</b> [user]<br/><b>max_tokens</b> obligatorio"] --> c3["<b>content[0].text</b><br/><b>stop_reason</b><br/><b>usage</b> input · output"]
    end
    subgraph GE["🔵 Gemini · API nativa"]
        direction LR
        g1["<b>POST</b> …/models/<b>{id}</b>:generateContent<br/>x-goog-api-key"] --> g2["<b>systemInstruction</b> → parts<br/><b>contents</b> → parts<br/><b>generationConfig</b><br/>maxOutputTokens · thinkingLevel"] --> g3["<b>candidates[0] → content → parts → text</b><br/><b>finishReason</b><br/><b>usageMetadata</b>"]
    end
    subgraph OP["🟠 OpenRouter · interfaz común"]
        direction LR
        o1["<b>POST</b> openrouter.ai/api/v1/chat/completions<br/>Authorization: Bearer"] --> o2["<b>model</b> proveedor/modelo<br/><b>messages</b> [system, user]<br/><b>max_tokens</b>"] --> o3["<b>choices[0].message.content</b><br/><b>finish_reason</b><br/><b>usage</b> · cost · <b>provider</b>"]
    end
    CL ~~~ GE ~~~ OP
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef grupo fill:transparent,stroke:#8a8f98,stroke-dasharray:5 4
    class c1,g1,o1 herramienta
    class c2,g2,o2 datos
    class c3,g3,o3 base
    class CL,GE,OP grupo
```

### La prueba mínima en cada ruta

Las claves se leen de variables de entorno: nunca se escriben en el código ni se enseñan en pantalla.

**Claude · Messages API** (la ruta de Claude en clase):

```bash
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{
    "model": "claude-haiku-5-5",
    "max_tokens": 50,
    "thinking": {"type": "disabled"},
    "system": "Eres un asistente de pruebas.",
    "messages": [{"role": "user", "content": "Responde exactamente: API lista."}]
  }'
```

**Gemini · API nativa de Google:**

```bash
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent" \
  -H "x-goog-api-key: $GEMINI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "systemInstruction": {"parts": [{"text": "Eres un asistente de pruebas."}]},
    "contents": [{"role": "user", "parts": [{"text": "Responde exactamente: API lista."}]}],
    "generationConfig": {"maxOutputTokens": 50, "thinkingConfig": {"thinkingLevel": "MINIMAL"}}
  }'
```

**OpenRouter · interfaz común:**

```bash
curl https://openrouter.ai/api/v1/chat/completions \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "PROVEEDOR/MODELO_ELEGIDO",
    "max_tokens": 50,
    "messages": [
      {"role": "system", "content": "Eres un asistente de pruebas."},
      {"role": "user", "content": "Responde exactamente: API lista."}
    ]
  }'
```

Los IDs son los de referencia del curso; antes de clase hay que confirmar los elegidos para esta edición. Con la API de Anthropic, `GET /v1/models` devuelve los disponibles para tu clave.

> [!warning] Claude Haiku 5.5 y sus parámetros
> Haiku 5.5 razona por defecto y ese razonamiento consume salida: con un límite corto lo desactivamos (`"thinking": {"type": "disabled"}`). Tampoco admite una `temperature` distinta de la predeterminada: si la envías, devuelve un 400. Cada modelo tiene su contrato; por eso registramos los parámetros solicitados.

## Dos puertas de Google para Gemini

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    G["🔵 Gemini"] --> D["🔑 <b>Gemini API</b><br/>generativelanguage.googleapis.com<br/>clave en x-goog-api-key<br/><i>la estructura de la diapositiva</i>"]
    G --> V["☁️ <b>Google Cloud · Vertex AI</b><br/>{región}-aiplatform.googleapis.com<br/>proyecto + credenciales de Google Cloud<br/><i>la ruta de curso/remote.py y del ajuste</i>"]
    V --> T["🎯 Trabajo de ajuste,<br/>modelo ajustado<br/>y sus predicciones"]
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    class G base
    class D,V herramienta
    class T ajuste
```

El cuerpo de la petición es casi idéntico (`systemInstruction`, `contents`, `generationConfig`), pero cambian la URL, la autenticación y el proyecto que se factura. El ajuste gestionado de [[05-Fine-tuning-gestionado]] se hace en Google Cloud.

## Cambiar de proveedor desde `.env`

El repositorio del curso no está atado a un proveedor. `curso/remote.py` lee la ruta del `.env` y llama al adaptador que corresponde; los scripts, el validador y el informe no cambian.

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    ENV["⚙️ <b>.env</b><br/>LLM_PROVIDER=anthropic<br/>LLM_MODEL=claude-haiku-5-5<br/>ANTHROPIC_API_KEY=…"] --> R["🧭 <b>route()</b><br/>elige proveedor<br/>y modelo"]
    CLI["⌨️ --proveedor<br/>--modelo<br/><i>opcional</i>"] -.-> R
    R --> C["🔌 <b>chat()</b><br/>la misma llamada<br/>para todos"]
    C --> A1["🟣 <b>chat_anthropic</b><br/>SDK oficial<br/>Messages API"]
    C --> A2["🟠 <b>chat_completions</b><br/>OpenAI · OpenRouter<br/>Groq · Mistral · Ollama…"]
    C --> A3["🔵 <b>chat_gemini</b><br/>Gemini API<br/>con clave"]
    C --> A4["💻 <b>chat_local</b><br/>LM Studio"]
    C --> A5["☁️ <b>chat_google</b><br/>Vertex AI<br/>base y ajustado"]
    A1 & A2 & A3 & A4 & A5 --> REG["📒 <b>Mismo registro</b><br/>respuesta original, ruta, modelo,<br/>tokens, latencia y coste"]
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    class ENV,CLI datos
    class R,C decision
    class A1,A2,A3,A4,A5 herramienta
    class REG ajuste
```

| Para usar… | `LLM_PROVIDER` | `LLM_MODEL` | Clave en `.env` |
|---|---|---|---|
| Claude, ruta del curso | `anthropic` | `claude-haiku-5-5` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai` | el ID de tu cuenta | `OPENAI_API_KEY` |
| Cualquier modelo de OpenRouter | `openrouter` | `proveedor/modelo` | `OPENROUTER_API_KEY` |
| Gemini con clave | `gemini` | `gemini-3.1-flash-lite` | `GEMINI_API_KEY` |
| Groq, Mistral, Ollama… | `compatible` | el ID del servidor | `COMPAT_BASE_URL` y `COMPAT_API_KEY` |

```bash
python scripts/preflight.py          # qué ruta y qué credenciales ve el código, sin mostrar claves
python scripts/demo.py --caso DE-04  # una consulta con la ruta de .env
```

Lo que cambia entre proveedores (temperatura admitida, razonamiento, nombre del límite de salida) se ajusta con `LLM_TEMPERATURE` y `LLM_EXTRA_BODY`, sin tocar el código. Ese cambio forma parte del experimento: se anota junto a los resultados.

## Cuando la llamada falla

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    E{"🚨 Código HTTP"}
    E -->|"400"| e400["Petición mal formada<br/>campo o parámetro no admitido"]
    E -->|"401 · 403"| e401["Clave ausente, inválida<br/>o sin permiso para ese modelo"]
    E -->|"402"| e402["OpenRouter: sin créditos"]
    E -->|"429"| e429["Límite de uso o de gasto:<br/>esperar o revisar cuota"]
    E -->|"529"| e529["Anthropic: servicio<br/>sobrecargado temporalmente"]
    e400 --> X["🧾 Registrar el error y su código.<br/>En la evaluación cuenta como fallo:<br/>no se reintenta ni se cambia de modelo"]
    e401 --> X
    e402 --> X
    e429 --> X
    e529 --> X
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    classDef aviso fill:#ffeaea,stroke:#b94a48,color:#6c2221
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    class E decision
    class e400,e401,e402,e429,e529 aviso
    class X datos
```

Nuestros scripts de evaluación no reintentan ni sustituyen el modelo cuando falla una llamada: así el denominador sigue siendo el mismo para todas las condiciones.

> [!warning] Claves fuera de pantalla
> Las claves viven en `.env` o en variables de entorno, fuera de Git. No se pegan en el código, en el notebook ni en capturas de la clase.

> [!question] Pregunta al grupo
> Si cambias de OpenRouter a la API nativa de Claude, ¿qué tienes que tocar en tu código? ¿Cambia algo de lo que mides en la respuesta?

[← Abierto, cerrado y proveedores](01-Abierto-cerrado-y-proveedores.md) · [Siguiente: respuesta, consumo y coste →](03-Respuesta-consumo-y-coste.md)

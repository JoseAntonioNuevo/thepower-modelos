---
tags: [thePower, S8, proveedores, modelos-cerrados]
---

# Abierto, cerrado y quién responde a tu petición

**Abierto o cerrado depende de los pesos, no de cómo accedemos al modelo. En cada llamada conviene identificar tres cosas: el proveedor, el modelo concreto y la ruta por la que llega la petición.**

## Dos preguntas distintas

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    Q{"🔑 ¿Se publican<br/>los pesos?"}
    Q -->|"Sí: abierto"| D{"💻☁️ ¿Dónde<br/>lo ejecutas?"}
    Q -->|"No: cerrado"| C["☁️ Solo remoto<br/>por app o API"]
    D -->|"En tu equipo"| E1["🔓💻 <b>Abierto y local</b><br/>Qwen3-0.6B en LM Studio<br/>o Transformers · S7"]
    D -->|"En un servidor"| E2["🔓☁️ <b>Abierto y remoto</b><br/>un modelo de pesos abiertos<br/>servido por API · OpenRouter u otro"]
    C --> E3["🔒☁️ <b>Cerrado y remoto</b><br/>Claude, Gemini, GPT<br/>por app o API · S8"]
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    class Q,D decision
    class E1,E2 ajuste
    class C,E3 herramienta
```

> [!important] Remoto no significa cerrado
> Un modelo de pesos abiertos servido por OpenRouter es remoto y se usa por API, pero sigue siendo abierto: podrías descargarlo y ejecutarlo tú. Un modelo cerrado solo existe como servicio.

## Qué implica usar un modelo cerrado

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    C["🔒 Modelo cerrado<br/>por API"] --> P1["💳 <b>Pago por uso</b><br/>tokens de entrada y salida;<br/>el proveedor fija el precio"]
    C --> P2["🔄 <b>Versiones</b><br/>el proveedor publica,<br/>cambia y retira modelos"]
    C --> P3["📜 <b>Datos</b><br/>tus entradas viajan al proveedor<br/>según sus condiciones"]
    C --> P4["🧩 <b>Adaptación</b><br/>solo la que ofrezca el servicio;<br/>los pesos no se descargan"]
    C --> P5["🚀 <b>Sin infraestructura</b><br/>no instalas ni mantienes<br/>hardware para el modelo"]
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    classDef aviso fill:#ffeaea,stroke:#b94a48,color:#6c2221
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    class C herramienta
    class P1,P2,P3 datos
    class P4 aviso
    class P5 ajuste
```

## Proveedor → familia → ID concreto

En la llamada no se escribe «Claude» ni «Gemini»: se escribe el **ID concreto** de un modelo. Panorama de la presentación de S8:

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    AN["🏢 <b>Anthropic</b>"] --> CL["<b>Claude</b><br/>Haiku · rapidez<br/>Sonnet · equilibrio<br/>Opus / Fable · capacidad"]
    GO["🏢 <b>Google</b>"] --> GE["<b>Gemini</b><br/>Flash-Lite · eficiencia<br/>Flash · equilibrio<br/>Pro · tareas complejas"]
    OA["🏢 <b>OpenAI</b>"] --> GP["<b>GPT</b><br/>Luna · volumen y coste<br/>Sol · equilibrio<br/>Astra · tareas complejas"]
    CL --> id1["🔑 ID del curso en la API de Anthropic<br/><code>claude-haiku-5-5</code><br/><i>evidencia del 2 oct. por OpenRouter:</i><br/><code>anthropic/claude-haiku-4.5</code>"]
    GE --> id2["🔑 ID en la API de Google<br/><code>gemini-3.1-flash-lite</code><br/><i>en OpenRouter:</i> <code>google/gemini-3.1-flash-lite</code>"]
    GP --> id3["Referencia del panorama;<br/>no forma parte del trío<br/>de comparación"]
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    class AN,GO,OA herramienta
    class CL,GE,GP base
    class id1,id2 datos
    class id3 decision
```

> [!warning] ChatGPT es una aplicación, no un modelo
> Igual que la app de Claude o la de Gemini: son interfaces que usan modelos por detrás. En una API eliges y registras el ID del modelo.

El mismo modelo tiene un ID distinto según la ruta. Los nombres de familia cambian con frecuencia; lo estable es el método: anotar el ID exacto de la petición y el que devuelve la respuesta. Con la API de Anthropic, `GET /v1/models` lista los IDs disponibles para tu clave.

## Tres distinciones que debemos conservar

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    PR["🏢 <b>Proveedor</b><br/>la organización<br/>que presta el servicio"]
    MO["🧠 <b>Modelo</b><br/>la versión concreta<br/>que procesa la petición"]
    RU["🛣️ <b>Ruta de acceso</b><br/>API nativa del proveedor<br/>o interfaz común"]
    PR ~~~ MO ~~~ RU
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    class PR herramienta
    class MO base
    class RU decision
```

### Un caso real: la ruta decide quién responde

En las pruebas del **2 de octubre de 2026** llamamos a Claude y a Gemini **a través de OpenRouter**. La respuesta indica qué proveedor sirvió cada petición. **En clase llamaremos a Claude Haiku 5.5 directamente por la API de Anthropic**, así que cambian el modelo y el proveedor que procesa los datos:

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    subgraph OCT["📁 Evidencia del 2 de octubre · ruta OpenRouter"]
        direction LR
        OR["🛣️ OpenRouter<br/>/api/v1/chat/completions"] --> R1["📤 <code>anthropic/claude-haiku-4.5</code>"]
        OR --> R2["📤 <code>google/gemini-3.1-flash-lite</code>"]
        R1 --> BR["🏢 Respondió<br/><b>Amazon Bedrock</b>"]
        R2 --> GG["🏢 Respondió<br/><b>Google</b>"]
    end
    subgraph CLA["🎓 En clase · ruta nativa de Claude"]
        direction LR
        MA["🛣️ Messages API<br/>api.anthropic.com/v1/messages"] --> R3["📤 <code>claude-haiku-5-5</code><br/>Claude Haiku 5.5"]
        R3 --> AN["🏢 Responde<br/><b>Anthropic</b>"]
    end
    OCT ~~~ CLA
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef grupo fill:transparent,stroke:#8a8f98,stroke-dasharray:5 4
    class OR,BR,GG,MA,AN herramienta
    class R1,R2,R3 datos
    class OCT,CLA grupo
```

El modelo Claude es de Anthropic, pero la petición del 2 de octubre la sirvió **Amazon Bedrock** a través de OpenRouter. Por eso registramos siempre la ruta, el ID solicitado, el modelo devuelto, el proveedor y el identificador de la petición: las condiciones de datos, los precios y algunos parámetros dependen de quién procesa la llamada.

> [!question] Pregunta al grupo
> Si tu empresa solo permite enviar datos a un proveedor concreto, ¿basta con elegir el modelo? ¿Qué tendrías que fijar también en la ruta?

[← Guía](00-Guia-S8.md) · [Siguiente: rutas y peticiones API →](02-Rutas-y-peticiones-API.md)

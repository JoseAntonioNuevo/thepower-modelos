---
tags: [thePower, S8, respuesta, tokens, coste, latencia]
---

# Leer la respuesta: texto, parada, tokens, tiempo y coste

**Una respuesta de API trae mucho más que texto. Para comparar modelos con rigor registramos por qué terminó, cuántos tokens consumió, cuánto tardó y qué costó, junto con la ruta y el modelo que respondió.**

## Las cinco cosas que leemos

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    R["📥 <b>Respuesta</b>"] --> T["💬 <b>Texto generado</b><br/>lo que evaluaremos,<br/>tal como salió"]
    R --> S["🛑 <b>Motivo de parada</b><br/>¿terminó o se cortó?"]
    R --> K["🔢 <b>Tokens</b><br/>entrada y salida<br/>según el proveedor"]
    R --> L["⏱️ <b>Tiempo</b><br/>lo medimos nosotros<br/>alrededor de la llamada"]
    R --> C["💶 <b>Coste</b><br/>si la ruta lo devuelve;<br/>si no, se calcula aparte"]
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    class R base
    class T,S decision
    class K,L,C datos
```

## ¿Terminó o se cortó?

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    P{"🛑 Motivo<br/>de parada"}
    P -->|"Claude: end_turn<br/>Gemini: STOP<br/>OpenRouter: stop"| OK["✅ El modelo terminó<br/>su respuesta"]
    P -->|"Claude: max_tokens<br/>Gemini: MAX_TOKENS<br/>OpenRouter: length"| CUT["✂️ Se agotó el límite<br/>de salida"]
    CUT --> J["❌ Un JSON cortado<br/>no se puede leer:<br/>cuenta como error"]
    CUT --> TH["⚠️ En modelos que razonan,<br/>el razonamiento también<br/>consume salida"]
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef aviso fill:#ffeaea,stroke:#b94a48,color:#6c2221
    class P decision
    class OK ajuste
    class CUT,J,TH aviso
```

Por eso fijamos un límite de salida suficiente para la tarea y controlamos el razonamiento: en nuestras llamadas Gemini usa nivel de pensamiento mínimo y Claude se llama sin razonamiento extendido.

## De dónde sale el coste

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    a["🟣 <b>Messages API de Anthropic</b><br/>devuelve tokens en usage"] --> X["🧮 <b>Coste calculado</b><br/>tokens × precio publicado<br/>anotar fecha, moneda y método"]
    b["🔵 <b>API nativa de Google</b><br/>devuelve tokens en usageMetadata"] --> X
    c["🟠 <b>OpenRouter</b><br/>devuelve tokens y usage.cost"] --> Y["🧾 <b>Coste informado</b><br/>anotar que viene de usage.cost"]
    d["💻 <b>Modelo local</b><br/>sin tarifa por consulta"] --> Z["❔ <b>No medido</b><br/>hardware y energía:<br/>se registra null, no 0"]
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef aviso fill:#ffeaea,stroke:#b94a48,color:#6c2221
    class a,b,c,d herramienta
    class X,Y datos
    class Z aviso
```

> [!important] `null` no es cero
> Si no hemos medido un coste, lo registramos como ausente. Un modelo local no paga por consulta, pero usa hardware y energía que no hemos medido.

## Qué registramos de cada llamada

Así queda un registro real del proyecto:

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    QUI["🏢 <b>Quién y cómo</b><br/><br/>fecha · 2026-10-02T12:14:57Z<br/>model · anthropic/claude-haiku-4.5<br/>provider · Amazon Bedrock<br/>request_id · gen-1790943296-…<br/>max_tokens 180 · temperature 0<br/>reasoning desactivado"]
    QUE["📥 <b>Qué devolvió</b><br/><br/>respuesta_original · el texto sin tocar<br/>finish_reason · stop<br/>tokens_entrada 545 · tokens_salida 58<br/>latencia_s · 1,38<br/>coste_usd · 0,000835<br/>coste_metodo · usage.cost de OpenRouter"]
    VAL["✅ <b>Cómo se valoró</b><br/><br/>json_valido · true<br/>esquema_valido · true<br/>categoría · correcta<br/>prioridad · correcta"]
    QUI --> QUE --> VAL
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    class QUI herramienta
    class QUE base
    class VAL ajuste
```

Con la API de Anthropic en clase cambian los nombres (`stop_reason`, `usage.input_tokens`, `usage.output_tokens`) y no hay `usage.cost`: el coste se calcula con el precio publicado. Lo que registramos es lo mismo.

> [!note] La tarifa de Claude Haiku 5.5
> Anthropic publica **0,10 USD por millón de tokens de entrada y 0,50 USD por millón de salida** para `claude-haiku-5-5` con prompts de hasta 100.000 tokens (Claude Haiku 4.5: 1 y 5 USD). El repositorio multiplica los tokens por la tarifa que declares en `.env` (`LLM_PRECIO_ENTRADA_USD_MTOK`, `LLM_PRECIO_SALIDA_USD_MTOK`); sin tarifa, el coste queda `null`. Haiku 5.5 usa un tokenizador nuevo: el mismo texto cuenta unos 30 % más tokens que en Haiku 4.5.

## Una consulta real, dos modelos

Caso de soporte `DE-01` (*La lección no carga para mí y no puedo verla*), 2 de octubre de 2026, ambos **por OpenRouter**:

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    IN["💬 <b>DE-01</b><br/>esperado: tecnico · media"]
    IN --> GE["🔵 <b>Gemini 3.1 Flash-Lite</b><br/>servido por Google<br/>prompt básico<br/><br/>tecnico · media ✅<br/>223 tokens de entrada · 63 de salida<br/>1,22 s · 0,000150 USD"]
    IN --> CL["🟣 <b>Claude Haiku 4.5</b><br/>servido por Amazon Bedrock<br/>prompt + 4 ejemplos few-shot<br/><br/>tecnico · media ✅<br/>545 tokens de entrada · 58 de salida<br/>1,38 s · 0,000835 USD"]
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    class IN datos
    class GE base
    class CL herramienta
```

> [!warning] Tokens de proveedores distintos no son comparables uno a uno
> Cada familia usa su propio tokenizador. Además, Claude recibió los cuatro ejemplos few-shot y Gemini solo el prompt básico. Comparamos el coste final y la calidad, no el número de tokens entre familias.

## Treinta casos: tiempo y coste

Los 30 casos de test de soporte, mismo validador, 2 de octubre de 2026:

```mermaid
---
config:
  themeVariables:
    xyChart:
      plotColorPalette: "#7e4fb8"
---
xychart-beta
    title "Mediana de tiempo por respuesta (s)"
    x-axis ["Gemini · OpenRouter", "Claude · OpenRouter", "Qwen LoRA · Mac"]
    y-axis "segundos" 0 --> 3.5
    bar [1.10, 1.42, 3.07]
```

```mermaid
---
config:
  themeVariables:
    xyChart:
      plotColorPalette: "#ae7500"
---
xychart-beta
    title "Coste informado de los 30 casos (USD)"
    x-axis ["Gemini Flash-Lite", "Claude Haiku 4.5"]
    y-axis "USD" 0 --> 0.03
    bar [0.0044, 0.0249]
```

| Condición | Ruta | Ambas etiquetas | Mediana s | Tokens entrada (mediana) | Coste 30 casos |
|---|---|---:|---:|---:|---:|
| Gemini 3.1 Flash-Lite, prompt básico | OpenRouter → Google | 26/30 | 1,10 | 225 | 0,0044 USD |
| Claude Haiku 4.5, prompt + few-shot | OpenRouter → Amazon Bedrock | 24/30 | 1,42 | 549 | 0,0249 USD |
| Qwen3-0.6B + LoRA | Mac M5 local | 26/30 | 3,07 | 267 | no medido |

> [!example] Cálculo orientativo
> A esos precios y con esos prompts, 10.000 consultas costarían unos **1,5 USD** con Gemini y unos **8,3 USD** con Claude (coste de 30 casos ÷ 30 × 10.000). Es aritmética sobre una muestra pequeña: los precios cambian y el few-shot encarece cada consulta de Claude.

> [!note] El tiempo depende de más cosas que el modelo
> Red, cola del proveedor, ruta y hardware local influyen. Las cifras de Claude son de Haiku 4.5 por la ruta OpenRouter; **en clase llamaremos a Claude Haiku 5.5 por la API de Anthropic** y sus tiempos serán otros. Medimos siempre igual: alrededor de la llamada completa.

> [!question] Pregunta al grupo
> Con estas cifras, ¿elegirías el modelo más barato? ¿Qué más necesitarías saber de tu tarea antes de decidir?

[← Rutas y peticiones](02-Rutas-y-peticiones-API.md) · [Siguiente: comparar por tareas →](04-Comparar-por-tareas.md)

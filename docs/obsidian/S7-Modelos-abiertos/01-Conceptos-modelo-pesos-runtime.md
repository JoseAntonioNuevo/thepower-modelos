---
tags: [thePower, S7, conceptos, pesos-abiertos]
---

# Modelo, pesos, formato, runtime y proveedor

**Para usar un modelo abierto necesitamos cinco piezas distintas. Confundirlas es el origen de la mayoría de errores al descargar o ejecutar.**

## Las cinco piezas

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    MO["🧠 <b>Modelo</b><br/>arquitectura que calcula<br/>el siguiente token<br/><i>Qwen/Qwen3-0.6B</i>"]
    PE["🔢 <b>Pesos</b><br/>números aprendidos<br/>al entrenar<br/><i>751,6 millones en BF16</i>"]
    FO["📦 <b>Formato</b><br/>cómo se guardan<br/>en disco<br/><i>safetensors 1,50 GB<br/>GGUF Q8_0 639 MB</i>"]
    RU["⚙️ <b>Runtime</b><br/>programa que los carga<br/>en memoria y calcula<br/><i>Transformers · LM Studio</i>"]
    RE["💬 <b>Respuesta</b><br/>texto generado<br/><i>el JSON de la incidencia</i>"]
    PR["🏢 <b>Proveedor</b><br/>quién lo ejecuta<br/>y te da acceso<br/><i>tu propio ordenador</i>"]
    MO --> PE --> FO --> RU --> RE
    PR -.-> RU
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    class MO,PE base
    class FO datos
    class RU,PR herramienta
    class RE decision
```

| Pieza | Qué es | Ejemplo en S7 |
|---|---|---|
| **Modelo** | La red neuronal y su forma de calcular | Qwen3-0.6B, un modelo de lenguaje |
| **Pesos** | Los parámetros numéricos que aprendió | 751.632.384 valores en BF16 |
| **Formato** | El archivo que guarda esos pesos | Safetensors, GGUF o MLX |
| **Runtime** | El programa que los carga en memoria y hace la inferencia | Transformers, LM Studio, llama.cpp, Ollama |
| **Proveedor** | Quién ejecuta el modelo y cómo accedes | Tu Mac, un servidor propio o una API de terceros |

> [!tip] Una frase para recordarlo
> Los pesos son el conocimiento aprendido, el formato es la caja, el runtime es quien abre la caja y calcula, y el proveedor es dónde ocurre todo eso.

## Qué hace un modelo de lenguaje: token a token

Un modelo de lenguaje no «busca» una respuesta: calcula la probabilidad del siguiente token, elige uno, lo añade al texto y repite.

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    T["📝 Texto de entrada<br/>instrucciones + incidencia"] --> K["✂️ Tokenizer<br/>texto → IDs"]
    K --> M["🧠 Modelo con sus pesos"]
    M --> P["📊 Probabilidades<br/>del siguiente token"]
    P --> G["👆 Elegir uno<br/>greedy: el más probable"]
    G --> A{"¿Fin de la respuesta<br/>o 180 tokens?"}
    A -->|"No: añadirlo<br/>y repetir"| M
    A -->|"Sí"| R["💬 Texto generado"]
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    class T datos
    class K,M,P,G,R base
    class A decision
```

En nuestro experimento la generación es **greedy** (sin azar), con un máximo de **180 tokens** y el modo *thinking* de Qwen desactivado. Así dos ejecuciones con la misma entrada son comparables.

> [!note] Inferencia no es entrenamiento
> Durante esta generación los pesos solo se **leen**. Chatear con un modelo no lo entrena ni le enseña nada de forma permanente.

## Open-weight, open-source o servicio propietario

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart TB
    Q1{"¿Puedes descargar<br/>los pesos?"}
    Q1 -->|"No"| PROP["🔒 <b>Servicio propietario</b><br/>solo API o app<br/>Gemini, GPT, Claude"]
    Q1 -->|"Sí"| Q2{"¿Publican datos y código de entrenamiento,<br/>o información suficiente para reproducirlo?"}
    Q2 -->|"No necesariamente"| OW["🔓 <b>Open-weight</b><br/>pesos descargables<br/>Qwen3-0.6B, Llama"]
    Q2 -->|"Sí"| OS["🌍 <b>Open-source</b><br/>pesos + receta reproducible<br/>OLMo de Ai2"]
    OW --> L["📜 Leer la licencia:<br/>usos, redistribución,<br/>uso comercial"]
    OS --> L
    PROP --> T["📜 Términos del servicio"]
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef aviso fill:#ffeaea,stroke:#b94a48,color:#6c2221
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    class Q1,Q2 decision
    class PROP aviso
    class OW base
    class OS ajuste
    class L,T datos
```

| | Open-weight | Open-source | Servicio propietario |
|---|---|---|---|
| Pesos descargables | Sí | Sí | No: solo API o app |
| Datos y código de entrenamiento | No necesariamente | Sí, o lo suficiente para reproducirlo | No |
| Ejecución local y adaptación | Sí, si el hardware y la licencia lo permiten | Sí | Solo lo que ofrezca el proveedor |
| Licencia | Fija usos y redistribución; puede limitar el uso comercial | Permite usar, estudiar, modificar y compartir | Términos del servicio |

> [!warning] Dos confusiones frecuentes
> - Un modelo con **pesos abiertos también puede usarse por API**: abierto describe qué se publica, no dónde se ejecuta.
> - **Gemma** (pesos descargables de Google) es distinta de **Gemini** (servicio por API que veremos en S8).

## Familias y tareas

Las familias publican modelos de distintos tamaños y especialidades. Los nombres concretos cambian rápido; lo estable es el método para elegir. Ejemplos de la presentación de S7:

```mermaid
---
config:
  themeVariables:
    git0: "#9db4e8"
    gitBranchLabel0: "#17294a"
    cScale1: "#ffd88a"
    cScale2: "#c9a7f0"
    cScale3: "#8fd3dc"
    cScale4: "#94d3ad"
    cScaleLabel1: "#3d2a00"
    cScaleLabel2: "#2b1450"
    cScaleLabel3: "#0d3a40"
    cScaleLabel4: "#143d2c"
---
mindmap
  root((Modelos abiertos<br/>según la tarea))
    💬 Conversación
      Chat, resumen, extracción
      Qwen · Gemma
    🧩 Razonamiento
      Varios pasos y comprobaciones
      DeepSeek · Qwen · GLM
    👩‍💻 Código y agentes
      Generar, revisar, usar herramientas
      Qwen · GLM · MiniMax · Kimi
    🖼️ Multimodales
      Texto con imagen o audio
      Gemma · MiniMax · Qwen con visión
```

> [!question] Pregunta al grupo
> Si una familia tiene un modelo de 0,6B y otro de cientos de miles de millones de parámetros, ¿comparten capacidades? ¿Y requisitos? La familia orienta; el modelo concreto y su archivo deciden.

[← Guía](00-Guia-S7.md) · [Siguiente: elegir un modelo →](02-Elegir-un-modelo.md)

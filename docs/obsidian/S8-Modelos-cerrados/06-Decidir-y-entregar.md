---
tags: [thePower, S8, decision, resultados, entrega]
---

# Decidir: abierto, cerrado o híbrido

**La elección se justifica por tarea: primero calidad comprobada, después integración, latencia, coste y condiciones de los datos. Puede ganar un modelo abierto, uno cerrado o una combinación de ambos.**

## Cómo decidir

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart TB
    T["🎯 Tarea, criterios y casos de prueba"] --> Q0{"¿Qué opciones superan<br/>la prueba de calidad?"}
    Q0 -->|"Ninguna"| N["🔁 Mejorar prompt o datos,<br/>probar otro modelo o ajustar"]
    Q0 -->|"Una o varias"| Q1{"¿Los datos pueden salir<br/>de tu infraestructura?"}
    Q1 -->|"No"| L["🔓💻 <b>Abierto en tu equipo</b><br/>o en tu servidor"]
    Q1 -->|"Sí, con condiciones"| Q2{"¿Qué pesa más<br/>en tu caso?"}
    Q2 -->|"Mínima integración<br/>y sin operar hardware"| C["🔒☁️ <b>Cerrado por API</b>"]
    Q2 -->|"Control de versión<br/>y coste a gran volumen"| O["🔓☁️ <b>Abierto servido</b><br/>o propio ajustado"]
    Q2 -->|"Casos fáciles y difíciles<br/>muy distintos"| H["🔀 <b>Híbrido</b><br/>cada caso al modelo<br/>que lo resuelve mejor"]
    L --> R["📒 Documentar la decisión<br/>con evidencias y límites"]
    C --> R
    O --> R
    H --> R
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    classDef aviso fill:#ffeaea,stroke:#b94a48,color:#6c2221
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    class T datos
    class Q0,Q1,Q2 decision
    class N aviso
    class L,O ajuste
    class C herramienta
    class H base
    class R datos
```

| Criterio | Qué preguntamos | Cómo lo medimos |
|---|---|---|
| **Calidad** | ¿Supera la prueba observable de la tarea? | Validador, tests, restricciones |
| **Integración** | ¿Cuánto código y operación exige? | Ruta, SDK, formato de respuesta |
| **Latencia** | ¿Responde a tiempo para el uso real? | Tiempo medido en las mismas condiciones |
| **Coste** | ¿Cuánto cuesta a nuestro volumen? | Tokens × precio, o coste informado; hardware si es local |
| **Datos** | ¿Dónde se procesan y con qué condiciones? | Proveedor real de la ruta y sus términos |

## Un ejemplo de híbrido

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    IN["📨 Incidencia"] --> L["🔓 Modelo abierto ajustado<br/>clasifica la mayoría<br/>a bajo coste por consulta"]
    L --> Q{"¿Caso dudoso, prioridad alta<br/>o respuesta no válida?"}
    Q -->|"No"| OK["✅ Respuesta"]
    Q -->|"Sí"| C["🔒 Modelo cerrado<br/>o revisión humana"]
    C --> OK
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    class IN datos
    class L,OK ajuste
    class Q decision
    class C herramienta
```

Es un patrón posible, no una recomendación general: también se valida con casos y se mide su coste total.

## El problema de soporte: abiertos y cerrados con el mismo test

Los 30 casos de test de S7, mismo validador, sin reparar respuestas. Ejecuciones del **2 de octubre de 2026**:

```mermaid
---
config:
  themeVariables:
    xyChart:
      plotColorPalette: "#268158"
---
xychart-beta
    title "Categoría y prioridad correctas a la vez (de 30)"
    x-axis ["Qwen básico", "Qwen few-shot", "Qwen LoRA", "Claude few-shot", "Gemini básico"]
    y-axis "casos" 0 --> 30
    bar [3, 10, 26, 24, 26]
```

| Condición | Ruta y proveedor | JSON | Esquema | Categoría | Prioridad | **Ambas** | Mediana s | Coste 30 casos |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B básico | Local, Mac M5 | 30 | 29 | 14 | 7 | **3** | 3,27 | no medido |
| Qwen3-0.6B few-shot | Local, Mac M5 | 30 | 27 | 15 | 19 | **10** | 3,34 | no medido |
| Qwen3-0.6B + LoRA | Local, Mac M5 | 30 | 30 | 26 | 29 | **26** | 3,07 | no medido |
| Claude Haiku 4.5, prompt + few-shot | OpenRouter → Amazon Bedrock | 30 | 30 | 26 | 28 | **24** | 1,42 | 0,0249 USD |
| Gemini 3.1 Flash-Lite, prompt básico | OpenRouter → Google | 30 | 30 | 26 | 30 | **26** | 1,10 | 0,0044 USD |
| Gemini base · Google Cloud | Google Cloud | — | — | — | — | **pendiente** | — | — |
| Gemini ajustado · Google Cloud | Google Cloud | — | — | — | — | **pendiente** | — | — |

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    Q["🔓💻 <b>Qwen3-0.6B + LoRA</b><br/>26/30 · 3,07 s<br/>coste por consulta no medido<br/>datos en tu equipo<br/>requiere entrenar y operar"]
    G["🔵☁️ <b>Gemini 3.1 Flash-Lite</b><br/>26/30 · 1,10 s<br/>0,0044 USD los 30 casos<br/>datos al proveedor<br/>sin entrenar"]
    C["🟣☁️ <b>Claude Haiku 4.5</b><br/>24/30 · 1,42 s<br/>0,0249 USD los 30 casos<br/>datos al proveedor<br/>con 4 ejemplos few-shot"]
    Q ~~~ G ~~~ C
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    class Q ajuste
    class G base
    class C herramienta
```

En esta tarea acotada y esta muestra, un modelo abierto de 0,6B ajustado con LoRA obtuvo el mismo acierto conjunto que Gemini Flash-Lite con el prompt básico. No es un ranking: son 30 casos sintéticos, una tarea concreta y rutas y precios de una fecha.

> [!note] Rutas de estas cifras
> Las condiciones de Claude (Haiku 4.5) y Gemini se midieron **por OpenRouter**. En clase llamaremos a Claude Haiku 5.5 por la API de Anthropic; para la tabla del ejercicio, la pareja Gemini base y Gemini ajustado se mide en Google Cloud.

## Cuando varios modelos fallan el mismo caso

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    subgraph AMB["❗ Fallan Claude y Gemini"]
        direction TB
        t14["<b>TE-14</b> · <i>¿Qué diferencia de precio hay<br/>entre las modalidades de matrícula?</i><br/>esperado facturacion · baja → ambos: <b>consulta</b>"]
        t15["<b>TE-15</b> · <i>Urgente: ¿las tarifas publicadas incluyen<br/>los impuestos? Solo estoy consultando.</i><br/>esperado facturacion · baja → ambos: <b>consulta</b>"]
    end
    subgraph DAT["🔐 Exposición de datos · prioridad alta"]
        direction TB
        t05["<b>TE-05</b> · <i>…puedo ver el teléfono privado de otra alumna</i><br/>esperado acceso · alta → Claude: <b>tecnico</b> · alta"]
        t13["<b>TE-13</b> · <i>…muestra números de cuenta privados de otros clientes</i><br/>esperado facturacion · alta → Claude: <b>tecnico</b> · alta"]
    end
    AMB --> R["🤔 ¿Error del modelo o de la taxonomía?<br/>Una pregunta sobre precios, ¿es facturación o consulta?<br/>Se discute y se documenta, pero <b>no se cambian<br/>las etiquetas del test</b> para mejorar la cifra"]
    DAT --> R
    classDef aviso fill:#ffeaea,stroke:#b94a48,color:#6c2221
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    classDef grupo fill:transparent,stroke:#8a8f98,stroke-dasharray:5 4
    class t14,t15,t05,t13 aviso
    class R decision
    class AMB,DAT grupo
```

Gemini falló 4 casos, todos por elegir `consulta` (TE-06, TE-07, TE-14, TE-15). Claude falló 6: dos por subir la prioridad a `alta` (TE-03, TE-08), dos de exposición de datos clasificados como `tecnico` y los dos de precios. Las reglas de la tarea, y no solo el modelo, deciden qué cuenta como acierto.

## El ejercicio: seis condiciones con el mismo test

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    D[("🗂️ <b>Datos de S7</b><br/>train · validación · test<br/>mismo validador")]
    D --> Q1["🔓 Qwen básico"]
    D --> Q2["🔓 Qwen few-shot"]
    D --> Q3["🔓 Qwen LoRA"]
    D --> G1["🔵 Gemini base<br/>Google Cloud"]
    D --> G2["🔵 Gemini ajustado<br/>trabajo de ajuste real"]
    D --> C1["🟣 Claude Haiku 5.5<br/>API de Anthropic<br/>prompt + ejemplos fijados"]
    Q1 --> T["📊 Tabla de seis condiciones<br/>calidad · latencia · tokens · coste<br/>entrenamiento e inferencia por separado"]
    Q2 --> T
    Q3 --> T
    G1 --> T
    G2 --> T
    C1 --> T
    T --> J["⚖️ Elección justificada<br/>por calidad, coste y control"]
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    class D datos
    class Q1,Q2,Q3 ajuste
    class G1,G2 base
    class C1 herramienta
    class T,J decision
```

```mermaid
---
config:
  themeVariables:
    git0: "#9db4e8"
    gitBranchLabel0: "#17294a"
    cScale1: "#ffd88a"
    cScale2: "#94d3ad"
    cScale3: "#9db4e8"
    cScale4: "#8fd3dc"
    cScaleLabel1: "#3d2a00"
    cScaleLabel2: "#143d2c"
    cScaleLabel3: "#17294a"
    cScaleLabel4: "#0d3a40"
---
mindmap
  root((Entrega S8))
    🗂️ Datos
      Mismos train, validación y test de S7
      Conversor a contents y parts
      Correspondencia de IDs
      Test fuera del ajuste y del few-shot
    🎯 Ajuste de Gemini
      ID del trabajo
      Estado terminado
      Recurso del modelo
      Predicciones
    🧪 Evaluación
      Respuestas originales
      Sin JSON mode ni reparación
      Un único validador
      Mismos 30 casos
    📊 Análisis
      Calidad, latencia y tokens
      Coste de entrenamiento e inferencia
      Hardware y proveedor declarados
      Elección justificada
```

> [!tip] Cualquier proveedor
> El código admite el proveedor que tengas: se cambia en `.env` (`LLM_PROVIDER`, `LLM_MODEL`). El curso usa Claude Haiku 5.5 por la API de Anthropic; si usas otra ruta o modelo, el informe de `resumir.py` lo muestra y debes declararlo.

> [!tip] Lo que se valora
> Ajuste verificable, Claude como referencia, separación del test y análisis. Dedicación orientativa: **4–6 horas activas** con datos y accesos preparados, aparte de las esperas.

> [!warning] Claves y recursos
> IDs de modelo, proyecto y regiones en variables; claves fuera del repositorio y de las capturas. Revisa el presupuesto antes de crear un trabajo y no supongas la misma tarifa para base y ajustado.

> [!question] Pregunta de cierre
> Para tu propia aplicación, ¿qué prueba decidiría entre un modelo abierto ajustado y uno cerrado con buen prompt? ¿Qué dato de coste o de privacidad cambiaría tu respuesta?

[← Fine-tuning gestionado](05-Fine-tuning-gestionado.md) · [Volver a la guía](00-Guia-S8.md)

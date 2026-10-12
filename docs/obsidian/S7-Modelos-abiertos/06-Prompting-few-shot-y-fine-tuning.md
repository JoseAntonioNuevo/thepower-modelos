---
tags: [thePower, S7, prompting, few-shot, fine-tuning]
---

# Prompting, few-shot y fine-tuning

**Las instrucciones y los ejemplos viajan dentro del contexto de cada consulta. El fine-tuning usa respuestas objetivo para actualizar parámetros que se conservan. Con LoRA, esos parámetros viven en un adaptador separado.**

## Qué recibe el modelo en cada condición

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    subgraph B["1 · Instrucciones básicas"]
        direction TB
        b1["SYSTEM<br/>reglas de prompts/basico.txt"] --> b2["USER<br/>incidencia nueva"]
    end
    subgraph F["2 · Few-shot"]
        direction TB
        f1["SYSTEM<br/>las mismas reglas"] --> f2["USER · TR-001 acceso<br/>ASSISTANT · su JSON"]
        f2 --> f3["USER · TR-061 facturación<br/>ASSISTANT · su JSON"]
        f3 --> f4["USER · TR-121 técnico<br/>ASSISTANT · su JSON"]
        f4 --> f5["USER · TR-181 consulta<br/>ASSISTANT · su JSON"]
        f5 --> f6["USER<br/>incidencia nueva"]
    end
    subgraph L["3 · LoRA"]
        direction TB
        l1["SYSTEM<br/>las mismas reglas"] --> l2["USER<br/>incidencia nueva"]
        l3["🧩 Adaptador cargado<br/>junto al modelo<br/>5 M parámetros aprendidos"]
    end
    B ~~~ F ~~~ L
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef grupo fill:transparent,stroke:#8a8f98,stroke-dasharray:5 4
    class b1,f1,l1 datos
    class f2,f3,f4,f5 datos
    class b2,f6,l2 base
    class l3 ajuste
    class B,F,L grupo
```

| Técnica | Qué hacemos | Qué cambia | Dónde vive | Cuándo se paga |
|---|---|---|---|---|
| **Instrucciones** | Explicar las reglas en cada consulta | El contexto | En cada petición | En cada consulta |
| **Few-shot** | Añadir ejemplos resueltos de train | El contexto | En cada petición | En cada consulta: más tokens |
| **LoRA** | Entrenar con train un adaptador | Parámetros del adaptador | En un archivo que se carga con el modelo | Una vez al entrenar |

> [!question] Pregunta al grupo
> ¿Dónde se guarda lo «aprendido» con few-shot cuando termina la consulta? ¿Y lo aprendido con LoRA?

## El coste del contexto

Mediana de tokens de entrada por consulta en los 30 casos de test (2 de octubre de 2026):

```mermaid
---
config:
  themeVariables:
    xyChart:
      plotColorPalette: "#ae7500"
---
xychart-beta
    title "Tokens de entrada por consulta (mediana, 30 casos)"
    x-axis ["Instrucciones básicas", "Few-shot (4 ejemplos)", "LoRA"]
    y-axis "tokens" 0 --> 550
    bar [267, 503, 267]
```

Few-shot casi **duplica** la entrada de cada consulta: los cuatro ejemplos viajan siempre. LoRA usa el mismo prompt que la condición básica; su coste se pagó una vez, en los **662,9 s** del entrenamiento.

## La trampa de «urgente»

Nuestra regla dice: *la palabra urgente sola no sube la prioridad*. El test tiene cuatro incidencias que empiezan por «Urgente:» y que, según las reglas, son **baja**. Esto hizo cada condición:

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    U["⚠️ <b>4 casos de test</b><br/>TE-07 · TE-15 · TE-22 · TE-29<br/><i>Urgente: ¿puedo configurar un método<br/>de acceso adicional? No estoy bloqueado.</i><br/>Prioridad esperada: <b>baja</b>"]
    U --> B["<b>Instrucciones básicas</b><br/>prioridad <b>alta</b> en 4 de 4<br/>❌ obedece a la palabra,<br/>no a la regla"]
    U --> F["<b>Few-shot</b><br/>inventa la prioridad <b>urgente</b><br/>en 3 de 4<br/>❌ un valor que no existe:<br/>rompe el esquema"]
    U --> L["<b>LoRA</b><br/>prioridad <b>baja</b> en 4 de 4<br/>✅ aplica la regla"]
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef aviso fill:#ffeaea,stroke:#b94a48,color:#6c2221
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    class U datos
    class B,F aviso
    class L ajuste
```

> [!question] Pregunta al grupo
> Si alguien escribe «urgente», ¿debe subir siempre la prioridad? ¿Qué ha cambiado al añadir few-shot? Los cuatro ejemplos few-shot no contienen la palabra «urgente», y aun así el modelo la usó como valor.

Una regla escrita en el prompt no garantiza que el modelo la aplique. Por eso medimos con casos que ponen a prueba cada regla.

## ¿Cuándo pasar a la siguiente técnica?

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart TB
    S["🎯 Tarea, reglas y<br/>casos de validación"] --> P["✍️ Instrucciones claras<br/>formato, etiquetas, reglas"]
    P --> Q1{"¿Cumple en<br/>validación?"}
    Q1 -->|"Sí"| OK["🏁 Quedarse aquí:<br/>lo más simple que funciona"]
    Q1 -->|"No"| FS["📚 Few-shot<br/>ejemplos de train en el contexto"]
    FS --> Q2{"¿Cumple en<br/>validación?"}
    Q2 -->|"Sí"| OK
    Q2 -->|"No"| Q3{"¿Qué falta?"}
    Q3 -->|"Datos nuevos o cambiantes:<br/>catálogo, normativa, estado"| RAG["🔎 Recuperar información<br/>y añadirla al contexto (RAG)"]
    Q3 -->|"Comportamiento estable:<br/>formato, taxonomía, tono"| Q4{"¿Tienes cientos de<br/>pares entrada → respuesta<br/>revisados?"}
    Q4 -->|"Sí"| LORA["🧩 LoRA<br/>entrenar un adaptador"]
    Q4 -->|"No"| DATA["🗂️ Reunir y anotar datos<br/>o probar otro modelo"]
    LORA --> EV["🧪 Comparar las condiciones<br/>con los mismos casos de test"]
    RAG --> EV
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    classDef aviso fill:#ffeaea,stroke:#b94a48,color:#6c2221
    class S,P,FS datos
    class Q1,Q2,Q3,Q4,EV decision
    class OK,LORA ajuste
    class RAG herramienta
    class DATA aviso
```

> [!important] LoRA no enseña datos que cambian
> LoRA sirve para **fijar un comportamiento**: formato, tono, vocabulario o taxonomía de un dominio. Si el problema es conocer información nueva o cambiante, es mejor recuperarla y dársela al modelo en el contexto.

## Una comparación justa

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    IG["🔒 <b>Igual en las tres condiciones</b><br/><br/>Qwen/Qwen3-0.6B · revisión c1899de…<br/>Tokenizer y chat template · thinking off<br/>prompts/basico.txt<br/>Greedy · máximo 180 tokens<br/>Los mismos 30 casos de test<br/>El mismo validador, sin reparar"]
    IG -->|"+ nada"| C1["📝 <b>Básico</b>"]
    IG -->|"+ 4 ejemplos de train<br/>en el contexto"| C2["📚 <b>Few-shot</b>"]
    IG -->|"+ adaptador activo"| C3["🧩 <b>LoRA</b>"]
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    class IG base
    class C1,C2 datos
    class C3 ajuste
```

En la comparación original/LoRA usamos el **mismo prompt básico sin few-shot**. Si añadimos ejemplos solo a una condición, estaríamos cambiando dos variables a la vez. Los ejemplos few-shot salen siempre de train: nunca de validación ni de test.

[← isbetter.ai con tres modelos](05a-isbetter-tres-modelos.md) · [Siguiente: cómo se entrena un LoRA →](07-Como-se-entrena-un-LoRA.md)

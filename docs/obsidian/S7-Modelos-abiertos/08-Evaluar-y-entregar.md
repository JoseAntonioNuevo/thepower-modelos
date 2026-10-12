---
tags: [thePower, S7, evaluacion, resultados, entrega]
---

# Evaluar, explicar los errores y entregar

**Una mejora solo existe si se mide con casos separados, el mismo protocolo y el mismo validador. Después se documenta qué se ha demostrado y qué no.**

## El protocolo de evaluación

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    T[("🔒 <b>test.jsonl</b><br/>30 casos que no se usaron<br/>para entrenar ni ajustar")] --> C1["📝 Básico"]
    T --> C2["📚 Few-shot"]
    T --> C3["🧩 LoRA"]
    C1 --> R["📄 Respuestas originales<br/>guardadas sin reparar<br/>ni reintentar"]
    C2 --> R
    C3 --> R
    R --> V["✅ Un único validador<br/>validate_answer"]
    V --> M["📊 Métricas por separado<br/>JSON · esquema · categoría<br/>prioridad · ambas"]
    M --> I["📒 Informe con fecha,<br/>entorno y límites"]
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    class T datos
    class C1,C2,R base
    class C3 ajuste
    class V,M,I decision
```

> [!important] Reglas que no se rompen
> - No se cambia el prompt, los datos ni el entrenamiento **después** de mirar el test.
> - Una respuesta que no es JSON cuenta como error en el denominador: no se borra.
> - No se extraen bloques de Markdown, no se repara el JSON y no se reintenta.
> - Tiempo y coste se registran con su entorno: `null` significa «no medido», no cero.

## Resultados de referencia

Experimento del **2 de octubre de 2026**, 30 casos sintéticos de test por condición, Qwen3-0.6B en un Apple M5 con 24 GB:

| Condición Qwen | JSON válido | Esquema válido | Categoría | Prioridad | **Ambas** |
|---|---:|---:|---:|---:|---:|
| Original + instrucciones | 30/30 | 29/30 | 14/30 | 7/30 | **3/30** |
| Original + few-shot | 30/30 | 27/30 | 15/30 | 19/30 | **10/30** |
| Original + LoRA | 30/30 | 30/30 | 26/30 | 29/30 | **26/30** |

```mermaid
---
config:
  themeVariables:
    xyChart:
      plotColorPalette: "#268158"
---
xychart-beta
    title "Categoría y prioridad correctas a la vez (de 30)"
    x-axis ["Básico", "Few-shot", "LoRA"]
    y-axis "casos" 0 --> 30
    bar [3, 10, 26]
```

```mermaid
---
config:
  radar:
    curveTension: 0
  themeVariables:
    cScale0: "#4569b2"
    cScale1: "#d08a00"
    cScale2: "#268158"
---
radar-beta
    title Las cinco medidas, de 0 a 30
    axis json["JSON válido"], esq["Esquema válido"], cat["Categoría"], pri["Prioridad"], amb["Ambas"]
    curve b["Básico"]{30, 29, 14, 7, 3}
    curve f["Few-shot"]{30, 27, 15, 19, 10}
    curve l["LoRA"]{30, 30, 26, 29, 26}
    max 30
    min 0
```

Las tres condiciones producen JSON válido en 30 de 30. La diferencia está en **aplicar las reglas**: elegir una categoría admitida y la prioridad correcta.

```mermaid
quadrantChart
    title Aciertos de categoría frente a prioridad (de 30)
    x-axis "Categoría 0/30" --> "Categoría 30/30"
    y-axis "Prioridad 0/30" --> "Prioridad 30/30"
    "Básico · 14 y 7": [0.47, 0.23] radius: 9, color: #4569b2
    "Few-shot · 15 y 19": [0.50, 0.63] radius: 9, color: #d08a00
    "LoRA · 26 y 29": [0.87, 0.97] radius: 9, color: #268158
```

Cada punto se sitúa en aciertos ÷ 30. Arriba a la derecha está la condición que aplica las dos reglas a la vez.

## Dónde se equivoca cada uno

### Categoría por categoría: original frente a LoRA

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    A["🔑 <b>acceso</b><br/>8 casos"] --> AB["🔵 Original · <b>0 de 8</b> ✓<br/>tecnico 5 · consulta 3"]
    A --> AL["🟢 LoRA · <b>7 de 8</b> ✓<br/>consulta 1"]
    F["💳 <b>facturacion</b><br/>7 casos"] --> FB["🔵 Original · <b>2 de 7</b> ✓<br/>consulta 3 · tecnico 1 · acceso 1"]
    F --> FL["🟢 LoRA · <b>7 de 7</b> ✓"]
    T["🛠️ <b>tecnico</b><br/>7 casos"] --> TB["🔵 Original · <b>4 de 7</b> ✓<br/>consulta 2 · <b>consulto</b> 1 (no existe)"]
    T --> TL["🟢 LoRA · <b>6 de 7</b> ✓<br/>acceso 1"]
    C["❓ <b>consulta</b><br/>8 casos"] --> CB["🔵 Original · <b>8 de 8</b> ✓<br/>pero la usa como cajón de sastre:<br/>16 de sus 30 respuestas"]
    C --> CL["🟢 LoRA · <b>6 de 8</b> ✓<br/>acceso 1 · tecnico 1"]
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef aviso fill:#ffeaea,stroke:#b94a48,color:#6c2221
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    class A,F,T,C datos
    class AB,FB,TB aviso
    class CB base
    class AL,FL,TL,CL ajuste
```

| | Categoría correcta | Prioridad correcta |
|---|---:|---:|
| Qwen original | 14/30 | 7/30 |
| Qwen con LoRA | 26/30 | 29/30 |

El original **no acertó ninguno de los 8 casos de acceso**, usó `consulta` como cajón de sastre e inventó la categoría `consulto`. En prioridad tendió a subir: dio `alta` a 5 casos que eran `baja` y a 8 que eran `media`.

### Los cuatro fallos del LoRA

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart TB
    L["🧩 LoRA · 4 fallos de 30<br/>en el acierto conjunto"]
    L --> E1["<b>TE-06</b> · <i>Sigo entrando correctamente.<br/>¿Cómo cambio el correo asociado a mi usuario?</i><br/>esperado acceso · baja → generó <b>consulta</b> · baja"]
    L --> E2["<b>TE-21</b> · <i>El color del icono del tema es incorrecto,<br/>aunque puedo utilizar la plataforma.</i><br/>esperado tecnico · baja → generó <b>acceso</b> · baja"]
    L --> E3["<b>TE-23</b> · <i>¿Puedo cursar este módulo<br/>antes que el anterior?</i><br/>esperado consulta · baja → generó <b>acceso</b> · baja"]
    L --> E4["<b>TE-25</b> · <i>¿Los directos se quedan grabados<br/>durante todo el programa?</i><br/>esperado consulta · baja → generó <b>tecnico · media</b>"]
    E1 --> D1["🤔 Anotación discutible:<br/>cambiar el correo, ¿acceso o consulta?<br/>No se cambia la etiqueta para mejorar la cifra"]
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef aviso fill:#ffeaea,stroke:#b94a48,color:#6c2221
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    class L ajuste
    class E1,E2,E3,E4 aviso
    class D1 decision
```

## Por qué una loss baja no lo resuelve todo

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    SI["<b>✅ Qué demuestra este experimento</b><br/><br/>Un ajuste real, guardado y recargado en otro proceso<br/>Una comparación controlada del mismo modelo<br/>Mejor seguimiento de nuestras reglas en esta muestra sintética"]
    NO["<b>❌ Qué no demuestra</b><br/><br/>Que resuelva incidencias reales: la respuesta es una frase tipo por categoría<br/>Una mejora general del modelo ni un ranking de modelos<br/>Coste cero ni ausencia de errores: quedan 4 fallos en 30 casos"]
    SI ~~~ NO
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef aviso fill:#ffeaea,stroke:#b94a48,color:#6c2221
    class SI ajuste
    class NO aviso
```

Si usáramos estos errores del test para reajustar el modelo, el test dejaría de ser una medida independiente. Un siguiente experimento necesitaría decisiones previas y un test nuevo.

> [!question] Pregunta de cierre
> ¿Qué evidencia necesitarías antes de usar este clasificador con incidencias reales?

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    A["🧪 Clasificador<br/>que funciona en<br/>30 casos sintéticos"] --> B["📥 Incidencias reales<br/>anonimizadas y<br/>anotadas por personas"]
    B --> C["📊 Una muestra mayor<br/>con la distribución real<br/>de categorías y prioridades"]
    C --> D["⚖️ Coste de cada error<br/>¿qué pasa si una alta<br/>se clasifica como baja?"]
    D --> E["👀 Revisión humana<br/>en los casos dudosos<br/>o de prioridad alta"]
    E --> F["📈 Seguimiento continuo<br/>y un test nuevo<br/>para cada cambio"]
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    classDef aviso fill:#ffeaea,stroke:#b94a48,color:#6c2221
    class A ajuste
    class B,C datos
    class D aviso
    class E,F decision
```

## Publicar el adaptador y recargarlo desde el Hub

El ejercicio pide publicar el adaptador en **tu** cuenta de Hugging Face, con model card, y demostrar que se puede recargar desde allí. Es una comprobación distinta de la recarga local. `scripts/publicar_adaptador.py` sigue estos pasos:

```mermaid
sequenceDiagram
    autonumber
    actor A as 👩‍🎓 Tú
    participant P as 🐍 publicar_adaptador.py
    participant H as 🤗 Hugging Face Hub
    participant Q as 🧠 Qwen3-0.6B local
    A->>P: --repo tu-usuario/qwen3-soporte-lora
    P->>H: whoami: ¿el repo es de tu usuario?
    P->>H: create_repo privado, sin sobrescribir uno existente
    P->>H: upload_folder: adapter_config.json,<br/>adapter_model.safetensors, README.md, entrenamiento.json
    H-->>P: commit publicado = revisión exacta
    rect rgba(38,129,88,0.15)
    P->>H: snapshot_download de esa revisión
    H-->>P: archivos del adaptador
    P->>Q: cargar base + adaptador descargado
    Q-->>P: respuesta a «Tengo un cargo duplicado en mi matrícula.»
    end
    P-->>A: registro de la recarga<br/>repo · revisión · respuesta
```

> [!warning] Credenciales
> El inicio de sesión en Hugging Face es tuyo y local. Nunca incluyas tokens, `.env` ni capturas con claves en la entrega.

> [!note] Estado del adaptador de referencia
> El adaptador del profesor está recargado y comprobado **localmente**. Su publicación y recarga desde el Hub es una evidencia distinta y se presenta solo cuando exista.

## Qué entregar

```mermaid
---
config:
  themeVariables:
    git0: "#94d3ad"
    gitBranchLabel0: "#143d2c"
    cScale1: "#ffd88a"
    cScale2: "#9db4e8"
    cScale3: "#94d3ad"
    cScale4: "#8fd3dc"
    cScale5: "#c9a7f0"
    cScaleLabel1: "#3d2a00"
    cScaleLabel2: "#17294a"
    cScaleLabel3: "#143d2c"
    cScaleLabel4: "#0d3a40"
    cScaleLabel5: "#2b1450"
---
mindmap
  root((Entrega S7))
    🗂️ Datos
      Sintéticos, sin datos personales
      train · validación · test separados
      Sin duplicados entre particiones
    🧪 Comparación
      Básico · few-shot · LoRA
      Mismo template y thinking off
      Mismos parámetros de generación
      Un único validador
    🧩 Adaptador
      Configuración LoRA fija
      Publicado en el Hub
      Model card con límites
      Recarga desde el Hub
    📊 Evidencia
      Respuestas originales
      Tabla de resultados
      Versiones y hardware
    ⭐ Opcional
      Analizar tres errores
      sin reajustar con test
```

> [!tip] Se valora la comparación reproducible
> Separación de datos, comparación justa, entrenamiento reproducible y recarga. **LoRA no tiene obligación de ganar**: un resultado peor, bien medido y documentado, es una entrega válida.

## Puente a S8

En la próxima sesión aplicaremos los mismos criterios a modelos servidos por API, como Gemini y Claude: misma tarea, mismos casos, mismo validador. Cambia dónde se ejecuta el modelo y qué podemos controlar, no la forma de comprobar si responde bien.

[← Cómo se entrena un LoRA](07-Como-se-entrena-un-LoRA.md) · [Volver a la guía](00-Guia-S7.md) · [Ir a S8 →](../S8-Modelos-cerrados/00-Guia-S8.md)

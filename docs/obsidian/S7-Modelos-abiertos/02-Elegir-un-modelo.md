---
tags: [thePower, S7, eleccion, hardware, cuantizacion]
---

# Elegir un modelo para tu tarea y tu equipo

**La elección empieza por la tarea, no por el ranking. Después comprobamos si el candidato cabe, si es compatible y si cumple en nuestros casos.**

## El recorrido de una elección justificada

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    subgraph P1["1 · Definir"]
        direction TB
        T1["🎯 ¿Qué debe hacer?<br/><i>clasificar incidencias<br/>y devolver JSON</i>"]
        T2["✅ ¿Qué es hacerlo bien?<br/><i>etiquetas admitidas,<br/>reglas de prioridad</i>"]
        T3["🚧 ¿Qué restricciones hay?<br/><i>español, datos privados,<br/>portátil, coste</i>"]
        T1 ~~~ T2 ~~~ T3
    end
    subgraph P2["2 · Filtrar candidatos"]
        direction TB
        F1{"📜 ¿La licencia<br/>permite mi uso?"}
        F2{"💾 ¿Cabe en<br/>mi memoria?"}
        F3{"📦 ¿Hay formato para<br/>mi runtime?"}
        F4{"🗣️ ¿Funciona en<br/>mi idioma?"}
        F1 --> F2 --> F3 --> F4
    end
    subgraph P3["3 · Probar con tus casos"]
        direction TB
        E1["🧪 Casos de desarrollo<br/>y validación"] --> E2{"¿Cumple los<br/>criterios?"}
    end
    P1 --> F1
    F1 -->|"No"| X["❌ Descartar<br/>o buscar otro"]
    F2 -->|"No"| Q["🗜️ Cuantizar o<br/>elegir uno menor"]
    Q --> F2
    F4 --> E1
    E2 -->|"Sí"| OK["🏁 Elegido<br/>con evidencia"]
    E2 -->|"No"| MJ["✍️ Mejorar prompt,<br/>few-shot o LoRA"]
    MJ --> E1
    E2 -->|"Tampoco"| X
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    classDef aviso fill:#ffeaea,stroke:#b94a48,color:#6c2221
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    classDef grupo fill:transparent,stroke:#8a8f98,stroke-dasharray:5 4
    class T1,T2,T3 datos
    class F1,F2,F3,F4,E2 decision
    class E1 herramienta
    class X aviso
    class Q,MJ datos
    class OK ajuste
    class P1,P2,P3 grupo
```

| Criterio | Pregunta concreta | Dónde se comprueba |
|---|---|---|
| **Tarea** | ¿Chat, extracción, código, visión, razonamiento? | Model card y pruebas propias |
| **Idioma** | ¿Responde bien en español con nuestro vocabulario? | Casos de desarrollo |
| **Hardware** | ¿Cabe en RAM, VRAM o memoria unificada con el contexto que necesito? | Cálculo + carga real |
| **Licencia** | ¿Permite uso comercial, redistribución, adaptación? | Model card y archivo de licencia |
| **Calidad observada** | ¿Acierta en **nuestros** casos con **nuestro** validador? | Comparación y evaluación |

> [!important] Por qué usamos Qwen3-0.6B en clase
> Es pequeño, tiene licencia Apache-2.0, se descarga en minutos y su LoRA se entrenó en unos 11 minutos en un Mac. Eso lo hace **bueno para aprender el proceso**. No demuestra que sea la mejor opción para un producto: sus errores tampoco representan a toda la familia de modelos abiertos.

## ¿Qué puede ejecutar mi equipo?

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart TB
    subgraph MEM["Tipos de memoria"]
        direction LR
        R["🧮 <b>RAM</b><br/>memoria del sistema,<br/>la usa la CPU"]
        V["🎮 <b>VRAM</b><br/>memoria de la<br/>tarjeta gráfica"]
        U["🍏 <b>Memoria unificada</b><br/>CPU y GPU comparten<br/>la misma, como en Apple Silicon"]
        R ~~~ V ~~~ U
    end
    subgraph USO["Qué la ocupa al ejecutar"]
        direction LR
        W["📦 Pesos<br/>parámetros × bits ÷ 8"]
        C["📜 Contexto<br/>más tokens, más memoria"]
        RT["⚙️ Runtime<br/>y sus estructuras"]
        SO["🖥️ Sistema y<br/>otras aplicaciones"]
        W ~~~ C ~~~ RT ~~~ SO
    end
    USO -->|"la suma debe caber en"| MEM
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    classDef grupo fill:transparent,stroke:#8a8f98,stroke-dasharray:5 4
    class R,V,U herramienta
    class W base
    class C datos
    class RT,SO herramienta
    class MEM,USO grupo
```

### El cálculo teórico de los pesos

> [!example] Memoria de pesos ≈ parámetros × bits ÷ 8
> Un modelo de **8B** (8.000 millones de parámetros):
> - 16 bits → 8.000 M × 16 ÷ 8 = **16 GB**
> - 8 bits → **8 GB**
> - 4 bits → **4 GB**
>
> Es solo la parte de los pesos. El contexto, el runtime y el sistema suman memoria encima.

```mermaid
---
config:
  themeVariables:
    xyChart:
      plotColorPalette: "#4569b2"
---
xychart-beta
    title "Pesos de un modelo de 8B según la cuantización (GB, cálculo teórico)"
    x-axis ["16 bits (FP16/BF16)", "8 bits (Q8)", "4 bits (Q4)"]
    y-axis "GB" 0 --> 18
    bar [16, 8, 4]
```

**Cuantizar** es guardar cada peso con menos bits: ocupa menos memoria y puede perder algo de calidad. Por eso un mismo modelo aparece en varias versiones (Q4, Q8…).

### Nuestros archivos reales

Medidos en disco con el modelo de la clase:

```mermaid
---
config:
  themeVariables:
    xyChart:
      plotColorPalette: "#268158"
---
xychart-beta horizontal
    title "Qwen3-0.6B en disco (MB)"
    x-axis ["Original BF16 · 1503 MB", "GGUF Q8_0 · 639 MB", "Adaptador LoRA · 20 MB"]
    y-axis "MB" 0 --> 1600
    bar [1503, 639, 20]
```

- `model.safetensors`: 751.632.384 parámetros × 16 bits ÷ 8 ≈ **1,50 GB**. Coincide con el cálculo.
- `Qwen3-0.6B-Q8_0.gguf`: **639 MB**, menos de la mitad, para LM Studio.
- Nuestro adaptador LoRA: **20 MB**. No sustituye al modelo: se carga **junto** a él (lo veremos en [[07-Como-se-entrena-un-LoRA]]).

> [!question] Pregunta al grupo
> ¿Qué memoria tiene tu equipo? Con la fórmula, ¿qué tamaño de modelo en 4 bits podrías cargar dejando sitio al contexto y al sistema?

## ¿Modelo abierto en local o modelo por API?

No hay una respuesta universal: depende de la tarea y de las restricciones. En S8 aplicaremos los mismos criterios a Gemini y Claude.

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart TB
    S["🎯 Tarea definida<br/>y casos de prueba"] --> Q1{"¿Los datos no pueden<br/>salir de tu equipo<br/>o necesitas funcionar offline?"}
    Q1 -->|"Sí"| L["🔓💻 <b>Abierto en local</b>"]
    Q1 -->|"No"| Q2{"¿Necesitas fijar una versión<br/>exacta o adaptar los pesos<br/>con tus datos?"}
    Q2 -->|"Sí"| L2["🔓 <b>Abierto</b><br/>local o en tu servidor"]
    Q2 -->|"No"| Q3{"¿Tu tarea exige más calidad<br/>de la que logra un modelo<br/>que cabe en tu hardware?"}
    Q3 -->|"Sí"| A["☁️ <b>API</b><br/>propietario o abierto servido"]
    Q3 -->|"No"| Q4{"¿Qué te sale más caro:<br/>pagar por uso o mantener<br/>hardware y operación?"}
    Q4 -->|"Más caro: pago por uso"| L2
    Q4 -->|"Más caro: mantener"| A
    L --> EV["🧪 En todos los casos:<br/>comparar con tus casos<br/>y el mismo validador"]
    L2 --> EV
    A --> EV
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    class S datos
    class Q1,Q2,Q3,Q4,EV decision
    class L,L2 base
    class A herramienta
```

| | Abierto en tu equipo | Por API |
|---|---|---|
| **Datos** | No salen de tu máquina | Viajan al proveedor según sus términos |
| **Coste** | Sin tarifa por consulta; hardware y energía no son gratis | Pago por tokens; sin hardware propio |
| **Control** | Fijas revisión, archivo y parámetros | El proveedor puede cambiar o retirar versiones |
| **Adaptación** | LoRA con tus datos, guardas el adaptador | Solo la que ofrezca el proveedor |
| **Capacidad** | Limitada por tu memoria | Modelos mayores sin instalar nada |
| **Operación** | Tú instalas, actualizas y sirves | Lo gestiona el proveedor |

> [!warning] Mismos criterios para todos
> La marca, el tamaño o un tono seguro no deciden. Decide la calidad observada en tus casos, con los recursos y el control que necesitas.

[← Conceptos](01-Conceptos-modelo-pesos-runtime.md) · [Siguiente: Hugging Face y formatos →](03-Hugging-Face-y-formatos.md)

---
tags: [thePower, S8, fine-tuning, Gemini, Google-Cloud]
---

# Fine-tuning gestionado: ajustar un modelo que no puedes descargar

**En S7 entrenábamos un adaptador en nuestro equipo y lo guardábamos como archivo. En un ajuste gestionado entregamos los datos al servicio, este ejecuta un trabajo de entrenamiento y obtenemos un modelo ajustado al que llamamos por API. Los pesos no se descargan.**

## Tres formas de cambiar el comportamiento

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    P["✍️ <b>Prompt</b><br/>definimos lo que debe hacer<br/>en cada consulta"] --> F["📚 <b>Few-shot</b><br/>añadimos ejemplos al contexto,<br/>sin entrenar"]
    F --> A["🎯 <b>Ajuste</b><br/>el modelo aprende de datos<br/>mediante un trabajo de ajuste"]
    P -.-> c1["Cambia el contexto<br/>de cada petición"]
    F -.-> c1
    A -.-> c2["Cambia un modelo<br/>que se conserva"]
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    class P,F datos
    class A ajuste
    class c1,c2 decision
```

| Condición del ejercicio | Cómo se adapta | Ruta |
|---|---|---|
| **Gemini base** | Prompt básico | Google Cloud |
| **Gemini ajustado** | Ajuste gestionado con train y validación de S7 | Google Cloud |
| **Claude Haiku 5.5** | Prompt + los mismos 4 ejemplos few-shot de S7, sin ajustar | API de Anthropic (`claude-haiku-5-5`) |

## S7 frente a S8: dónde ocurre el entrenamiento

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart TB
    subgraph L["💻 S7 · LoRA local"]
        direction LR
        l1["🗂️ train.jsonl"] --> l2["🔁 Tu equipo entrena<br/>Transformers + PEFT"] --> l3["🧩 Adaptador<br/>archivo de 20 MB<br/>en tu disco"] --> l4["♻️ Lo cargas tú<br/>con el modelo base"]
    end
    subgraph G["☁️ S8 · Ajuste gestionado en Google Cloud"]
        direction LR
        g1["🗂️ train + validación<br/>en formato Gemini"] --> g2["🪣 Bucket de<br/>Cloud Storage"] --> g3["🔁 Trabajo de ajuste<br/>lo ejecuta Google"] --> g4["🎯 Modelo ajustado<br/>recurso en el proyecto,<br/>se usa por API"]
    end
    L ~~~ G
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef grupo fill:transparent,stroke:#8a8f98,stroke-dasharray:5 4
    class l1,g1 datos
    class l2,g3 herramienta
    class g2 herramienta
    class l3,g4 ajuste
    class l4 base
    class L,G grupo
```

> [!important] La frase para recordar
> En S7 guardábamos un adaptador local. En un ajuste gestionado entregamos los datos al servicio y obtenemos un recurso de modelo ajustado. Para demostrarlo necesitamos identificar el trabajo, comprobar que terminó y usar su resultado.

## 1 · Convertir los datos de S7

Reutilizamos **exactamente** las particiones de S7. `scripts/convertir_gemini.py` transforma cada fila al formato que espera Google:

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    R["📄 <b>Fila de S7</b><br/>TR-001<br/>incidencia + objetivo"]
    R --> S["<b>systemInstruction</b> → parts<br/>texto de prompts/basico.txt"]
    R --> U["<b>contents[0]</b> role: user<br/>la incidencia"]
    R --> M["<b>contents[1]</b> role: model<br/>el JSON objetivo como texto"]
    R --> I["<b>ids.json</b><br/>el ID va aparte,<br/>no entra en la conversación"]
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    class R,S datos
    class U base
    class M ajuste
    class I decision
```

Primera línea real de `datos/gemini/train.jsonl` (instrucciones recortadas):

```json
{
  "systemInstruction": {"parts": [{"text": "Eres un clasificador de incidencias de una plataforma educativa ficticia.…"}]},
  "contents": [
    {"role": "user",  "parts": [{"text": "No puedo iniciar sesión con mi cuenta del curso."}]},
    {"role": "model", "parts": [{"text": "{\"categoria\": \"acceso\", \"prioridad\": \"media\", \"respuesta\": \"Indica el error de acceso y el paso en que aparece, sin compartir tu contraseña.\"}"}]}
  ]
}
```

En S7 el rol del modelo se llamaba `assistant`; en Gemini se llama `model`. Es el mismo ejemplo con otra estructura.

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    T[("<b>train</b><br/>240 ejemplos")] -->|"se sube"| B["🪣 Bucket<br/>del curso"]
    V[("<b>validación</b><br/>30 ejemplos")] -->|"se sube"| B
    X[("<b>test</b><br/>30 casos")] -->|"nunca se sube<br/>para entrenar"| E["🧪 Se queda fuera<br/>para evaluar base<br/>y ajustado"]
    B --> J["🔁 Trabajo de ajuste"]
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    class T,V,X datos
    class B,J herramienta
    class E decision
```

## 2 · El ciclo del trabajo de ajuste

`scripts/gemini_tuning.py` separa cada paso en una acción distinta:

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart TB
    subgraph T1["⌨️ Acciones de scripts/gemini_tuning.py"]
        direction LR
        P["📝 <b>preparar</b><br/>sin red: imprime<br/>especificación y hashes"] --> S["🪣 <b>subir-datos</b><br/>train y validación<br/>sin sobrescribir"]
        S --> C["🚀 <b>crear</b><br/>lanza el trabajo<br/>y guarda su ID"]
        C --> E["🔎 <b>estado</b><br/>consulta ese mismo<br/>trabajo con su ID"]
    end
    subgraph T2["📦 Resultado y evaluación"]
        direction LR
        M["🎯 <b>Modelo ajustado</b><br/>recurso que devuelve<br/>el servicio"] --> PR["💬 <b>Predicciones</b><br/>30 casos de test"]
        PR --> CMP["⚖️ <b>Comparar</b> con Gemini base<br/>mismas instrucciones,<br/>thinking MINIMAL"]
    end
    T1 -->|"cuando el trabajo termina"| T2
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    classDef grupo fill:transparent,stroke:#8a8f98,stroke-dasharray:5 4
    class P,S datos
    class C,E herramienta
    class M ajuste
    class PR,CMP decision
    class T1,T2 grupo
```

> [!warning] `crear` dos veces lanza dos entrenamientos
> Para saber cómo va un trabajo se usa `estado` con su ID. Volver a ejecutar `crear` inicia otro trabajo y otro coste.

### Los estados de un trabajo

```mermaid
stateDiagram-v2
    direction LR
    [*] --> Pendiente: crear
    Pendiente --> Ejecutando
    Ejecutando --> Terminado
    Ejecutando --> Fallido
    Pendiente --> Cancelado
    Ejecutando --> Cancelado
    Terminado --> [*]: modelo ajustado disponible
    Pendiente: ⏳ Pendiente · en cola
    Ejecutando: 🔁 Ejecutando · entrenando
    Terminado: ✅ Terminado · SUCCEEDED
    Fallido: ❌ Fallido · FAILED
    Cancelado: 🚫 Cancelado · CANCELLED
    classDef wait fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef run fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef ok fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef bad fill:#ffeaea,stroke:#b94a48,color:#6c2221
    class Pendiente wait
    class Ejecutando run
    class Terminado ok
    class Fallido,Cancelado bad
```

El trabajo se consulta con intervalos razonables y se espera fuera del directo: no forma parte de los 90 minutos.

## 3 · La especificación preparada

Salida real de `preparar`, con nombres de ejemplo **no creados**:

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    SP["📝 <b>Especificación</b><br/><br/>modelo base · gemini-3.1-flash-lite<br/>región de ajuste · europe-west4<br/>región de uso · eu<br/>épocas · 2<br/>tamaño del adaptador · 8<br/>train · gs://…/thepower-s7-s8/train.jsonl<br/>validación · gs://…/thepower-s7-s8/validacion.jsonl"]
    SP --> ST["📌 Estado<br/>PREPARADO;<br/>no se ha llamado a Google"]
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef aviso fill:#ffeaea,stroke:#b94a48,color:#6c2221
    class SP datos
    class ST aviso
```

Antes de ejecutar hacen falta: un **proyecto identificado del curso**, permisos, facturación y presupuesto revisados, un bucket, cuota y acceso al modelo en la región elegida.

> [!warning] El presupuesto anotado no es un tope
> El script comprueba que la estimación no supera el presupuesto declarado, pero **Google no deja de cobrar** por ese número. Las alertas de presupuesto avisan; no apagan el consumo.

## 4 · Qué demuestra que un modelo se ha ajustado

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    SI["✅ <b>Sí lo demuestra</b><br/><br/>ID del trabajo de ajuste<br/>Estado final: terminado<br/>Recurso del modelo ajustado<br/>Fecha, datos y configuración<br/>Predicciones reales del modelo ajustado<br/>30 resultados base y 30 ajustados<br/>Coste de entrenamiento e inferencia por separado"]
    NO["❌ <b>No lo demuestra</b><br/><br/>Una llamada a Gemini por OpenRouter<br/>Cambiar solo el prompt<br/>Añadir ejemplos few-shot<br/>Un ID de modelo inventado<br/>Una captura sin trabajo identificable"]
    SI ~~~ NO
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef aviso fill:#ffeaea,stroke:#b94a48,color:#6c2221
    class SI ajuste
    class NO aviso
```

> [!note] Estado del laboratorio
> El laboratorio de Google Cloud y el ajuste real de Gemini están **pendientes**: aún no hay proyecto del curso configurado, ni trabajo, ni modelo ajustado que mostrar. En clase explicamos la conversión y la especificación preparada; no presentamos resultados de un ajuste que no existe.

> [!question] Pregunta al grupo
> Si el proveedor retira el modelo base dentro de un año, ¿qué pasa con tu modelo ajustado? ¿Y con el adaptador LoRA de S7?

[← Comparar por tareas](04-Comparar-por-tareas.md) · [Siguiente: decidir y entregar →](06-Decidir-y-entregar.md)

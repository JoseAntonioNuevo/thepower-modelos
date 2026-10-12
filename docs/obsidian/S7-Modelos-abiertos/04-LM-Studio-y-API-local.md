---
tags: [thePower, S7, lm-studio, api, inferencia]
---

# LM Studio y la API local

**LM Studio no es un modelo: es una aplicación que descarga y ejecuta los modelos compatibles en tu equipo. Descargar, cargar y consultar son tres operaciones diferentes.**

## Descargar ≠ cargar ≠ consultar

```mermaid
stateDiagram-v2
    direction LR
    [*] --> EnElHub
    EnElHub --> EnDisco: 1 · Descargar
    EnDisco --> EnMemoria: 2 · Cargar
    EnMemoria --> Generando: 3 · Consultar
    Generando --> EnMemoria: respuesta lista
    EnMemoria --> EnDisco: Expulsar
    EnElHub: 🤗 En Hugging Face
    EnDisco: 💾 En tu disco
    EnMemoria: ⚡ En memoria
    Generando: 💬 Inferencia
    classDef hub fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    classDef disk fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef mem fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef gen fill:#e8f7ee,stroke:#268158,color:#143d2c
    class EnElHub hub
    class EnDisco disk
    class EnMemoria mem
    class Generando gen
```

| Operación | Qué ocurre | Qué consume | Cómo lo ves en LM Studio |
|---|---|---|---|
| **Descargar** | El archivo GGUF/MLX pasa a tu disco | Red y espacio en disco | Discover y la lista de descargas |
| **Cargar** | El runtime lee los pesos y los coloca en memoria | RAM, VRAM o memoria unificada | Selector de modelo cargado |
| **Consultar** | Inferencia: calcula tokens a partir de tu prompt | Cálculo de CPU/GPU | Chat o servidor local |

> [!question] Pregunta al grupo
> Si apago el ordenador, ¿qué se conserva: lo descargado, lo cargado o lo que hablé en el chat? ¿Alguna de esas cosas cambia los pesos del modelo?

## Los cinco pasos de la demo

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    s1["🔎 <b>1 · Buscar</b><br/>nombre o URL de<br/>Hugging Face"] --> s2["⬇️ <b>2 · Descargar</b><br/>Qwen3-0.6B-Q8_0.gguf<br/>639 MB"]
    s2 --> s3["⚡ <b>3 · Cargar</b><br/>modelo en memoria<br/>contexto moderado"]
    s3 --> s4["⚙️ <b>4 · Configurar</b><br/>prompt de sistema,<br/>temperatura 0, thinking off"]
    s4 --> s5["💬 <b>5 · Consultar</b><br/>Chat de la app<br/>o API local"]
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    class s1 herramienta
    class s2 datos
    class s3 base
    class s4 datos
    class s5 ajuste
```

En Mac, Discover se abre con **Cmd+2** (Ctrl+2 en Windows/Linux). Anota el ID del modelo, el formato y la cuantización: son parte de la evidencia.

## El chat es solo una interfaz

Un modelo cargado puede atender a varias aplicaciones a través de la **API local**. LM Studio ofrece rutas compatibles con OpenAI (`/v1/…`), que usa isbetter.ai, y una API nativa (`/api/v1/…`), que usa por defecto `scripts/demo.py` para poder desactivar el razonamiento.

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    subgraph CL["Clientes"]
        direction TB
        c1["💬 Chat de LM Studio"]
        c2["🖥️ curl en la terminal"]
        c3["🐍 scripts/demo.py"]
        c4["⚖️ isbetter.ai<br/>en el navegador"]
    end
    subgraph PC["💻 Tu ordenador"]
        direction TB
        API["🔌 <b>Servidor local</b> · localhost:1234<br/>/v1/… compatible con OpenAI<br/>/api/v1/… API nativa de LM Studio"]
        M["🧠 <b>qwen-curso</b><br/>Qwen3-0.6B GGUF Q8_0<br/>cargado en memoria"]
        API <--> M
    end
    c1 --> M
    c2 -->|"/v1/models"| API
    c3 -->|"/api/v1/chat<br/>por defecto"| API
    c4 -->|"/v1/chat/completions<br/>con CORS y permiso<br/>de red local"| API
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    classDef grupo fill:transparent,stroke:#8a8f98,stroke-dasharray:5 4
    class c1,c2,c3,c4 decision
    class API herramienta
    class M base
    class CL,PC grupo
```

```bash
# ¿Qué modelos ofrece el servidor?
curl http://localhost:1234/v1/models

# Una consulta de desarrollo con el script del curso
python scripts/demo.py --proveedor local --caso DE-01
```

## Una consulta real, paso a paso

Evidencia del 2 de octubre de 2026, caso de desarrollo `DE-01`:

```mermaid
sequenceDiagram
    autonumber
    participant S as 🐍 demo.py
    box Tu ordenador
    participant L as 🔌 LM Studio
    participant Q as 🧠 Qwen3-0.6B GGUF Q8_0
    end
    S->>L: POST /api/v1/chat · «La lección no carga para mí y no puedo verla.»<br/>+ prompt básico · temperatura 0 · reasoning off · máx. 180 tokens
    L->>Q: 263 tokens de entrada
    rect rgba(69,105,178,0.12)
    Note over Q: primer token a los 0,25 s
    Q-->>L: 45 tokens de salida · 80,9 tokens/s
    end
    L-->>S: texto con el JSON · 0,83 s en total
    Note over S,Q: Ningún peso ha cambiado: solo inferencia
```

La respuesta fue:

```json
{
  "categoria": "consulta",
  "prioridad": "media",
  "respuesta": "No se puede cargar la lección por favor intenta revisar los datos o contacte a un técnico"
}
```

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    R["📄 Respuesta de<br/>Qwen3-0.6B en LM Studio"] --> J["✅ JSON válido"]
    J --> E["✅ Esquema válido<br/>claves y valores admitidos"]
    E --> C["❌ Categoría<br/>consulta ≠ tecnico"]
    E --> P["✅ Prioridad<br/>media = media"]
    C --> D["⚠️ Funcionar no<br/>significa acertar:<br/>hay que evaluar con casos"]
    P --> D
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef aviso fill:#ffeaea,stroke:#b94a48,color:#6c2221
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    class R base
    class J,E,P ajuste
    class C aviso
    class D decision
```

> [!important] Funcionar no es acertar
> El modelo cargó, respondió rápido y produjo JSON válido. Aun así eligió una categoría distinta de la que marcan nuestras reglas. Eso es lo que medimos en las notas siguientes.

> [!note] Coste «cero»
> Sin tarifa por consulta no significa gratis: el hardware y la energía no se han medido. Por eso registramos el coste como no disponible, no como cero.

## Si algo falla

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart TB
    P0["🛠️ Síntoma"] --> P1{"¿No aparecen<br/>modelos?"}
    P0 --> P2{"¿curl funciona<br/>pero la web no?"}
    P0 --> P3{"¿Error de formato<br/>al cargar?"}
    P0 --> P4{"¿Respuesta larga<br/>o con thinking?"}
    P0 --> P5{"¿No cabe<br/>en memoria?"}
    P1 --> S1["Servidor arrancado,<br/>URL acabada en /v1<br/>y modelo cargado"]
    P2 --> S2["Activar CORS<br/>en LM Studio y aceptar<br/>el permiso del navegador"]
    P3 --> S3["Revisar GGUF/MLX<br/>y que el runtime<br/>los admite"]
    P4 --> S4["Revisar reasoning off,<br/>límite de salida<br/>y prompt"]
    P5 --> S5["Modelo menor o<br/>más cuantizado;<br/>declarar el cambio"]
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    classDef aviso fill:#ffeaea,stroke:#b94a48,color:#6c2221
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    class P0 aviso
    class P1,P2,P3,P4,P5 decision
    class S1,S2,S3,S4,S5 ajuste
```

Si tu equipo no puede cargar un modelo grande, no concluyas que «los modelos abiertos no funcionan»: elige uno compatible con tu memoria y declara el cambio.

[← Hugging Face y formatos](03-Hugging-Face-y-formatos.md) · [Siguiente: comparar respuestas →](05-Comparar-respuestas.md)

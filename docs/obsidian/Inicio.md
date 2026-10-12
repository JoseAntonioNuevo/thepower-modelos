---
tags: [thePower, AI-Engineer, S7, S8]
---

# Modelos abiertos y cerrados · S7 y S8

Material visual de las dos sesiones. Un mismo caso práctico las une: convertir una incidencia ficticia de una plataforma educativa en un JSON con `categoria`, `prioridad` y `respuesta`.

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    C["💬 <b>Caso práctico</b><br/>incidencia → JSON con<br/>categoria, prioridad y respuesta"]
    C --> S7["🔓💻 <b>S7 · Modelos abiertos</b><br/>13 y 14 de octubre<br/><br/>elegir modelo · Hugging Face<br/>LM Studio · few-shot<br/>LoRA · evaluación"]
    S7 --> S8["🔒☁️ <b>S8 · Modelos cerrados y APIs</b><br/>15 y 19 de octubre<br/><br/>proveedores · peticiones API<br/>consumo y coste · comparación<br/>ajuste gestionado · decisión"]
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    class C datos
    class S7 ajuste
    class S8 herramienta
```

| Clase | Pregunta | Empieza aquí |
|---|---|---|
| **S7 · Modelos abiertos** | ¿Qué modelo puedo utilizar para mi tarea, cómo lo pongo en funcionamiento y cómo compruebo si sus respuestas son suficientemente buenas? | [[00-Guia-S7]] |
| **S8 · Modelos cerrados y APIs** | ¿Cómo conecto mi aplicación a estos modelos y qué pruebas necesito para elegir el que mejor resuelve mi tarea? | [[00-Guia-S8]] |

Cada carpeta tiene sus notas numeradas en orden de clase. Los diagramas están dentro de las notas, en bloques `mermaid`.

> [!tip] Si ves el código en lugar del diagrama
> Obsidian pide permiso antes de dibujar diagramas Mermaid en un vault nuevo. Pulsa el botón de permitir sobre el primer diagrama: se aplica a todo el vault.

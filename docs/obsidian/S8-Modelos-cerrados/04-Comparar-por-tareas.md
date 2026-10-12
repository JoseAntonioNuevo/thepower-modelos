---
tags: [thePower, S8, comparacion, isbetter, evaluacion]
---

# Comparar por tareas: el voto necesita una prueba

**Comparamos tres opciones con el mismo caso y criterios definidos antes de leer. Una respuesta bien redactada puede incumplir la tarea: por eso cada voto se apoya en algo que se puede comprobar.**

## Las tres opciones del directo

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    P["📋 <b>Mismo prompt completo</b><br/>misma tarea, mismos datos,<br/>conversación nueva"]
    P --> A["🔓 <b>Modelo abierto</b><br/>servido por API<br/>vía OpenRouter"]
    P --> B["🔵 <b>Gemini</b><br/>API de Google"]
    P --> C["🟣 <b>Claude Haiku 5.5</b><br/>API de Anthropic"]
    A --> I["⚖️ <b>isbetter.ai</b><br/>respuestas anónimas"]
    B --> I
    C --> I
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    class P datos
    class A ajuste
    class B base
    class C herramienta
    class I herramienta
```

## La misma dinámica en los tres casos

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    s1["1 · 📌 <b>Presentar</b><br/>el caso y los<br/>criterios de éxito"] --> s2["2 · 📤 <b>Enviar</b><br/>el mismo prompt<br/>a las tres opciones"]
    s2 --> s3["3 · 🙈 <b>Leer a ciegas</b><br/>sin saber qué<br/>modelo es cada una"]
    s3 --> s4["4 · 🔍 <b>Puntuar</b><br/>con pruebas<br/>observables"]
    s4 --> s5["5 · 🔓 <b>Revelar</b><br/>los modelos<br/>y discutir"]
    s5 --> O{"Resultado"}
    O --> o1["🏆 Ganador"]
    O --> o2["🤝 Empate"]
    O --> o3["🚫 Ninguna válida"]
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef aviso fill:#ffeaea,stroke:#b94a48,color:#6c2221
    class s1,s2 datos
    class s3,s4 decision
    class s5 herramienta
    class O decision
    class o1,o2 ajuste
    class o3 aviso
```

> [!warning] No es un benchmark
> Tres casos en directo sirven para aprender a evaluar. No dicen qué modelo es mejor en general: diferentes modelos pueden destacar en diferentes tareas.

## Forma frente a cumplimiento

```mermaid
---
config:
  themeVariables:
    quadrant1Fill: "#e8f7ee"
    quadrant2Fill: "#ffeaea"
    quadrant3Fill: "#eeeeee"
    quadrant4Fill: "#fff1d6"
    quadrant1TextFill: "#143d2c"
    quadrant2TextFill: "#6c2221"
    quadrant3TextFill: "#333333"
    quadrant4TextFill: "#503700"
---
quadrantChart
    title ¿Qué respuesta aceptamos?
    x-axis "Incumple la tarea" --> "Cumple la tarea"
    y-axis "Redacción floja" --> "Redacción convincente"
    quadrant-1 "Ideal"
    quadrant-2 "Peligrosa: convence y falla"
    quadrant-3 "Descartar"
    quadrant-4 "Válida, mejorable"
```

El cuadrante de arriba a la izquierda es el que más engaña: suena seguro, está bien escrito y aun así inventa un dato, no pasa una prueba o se salta una restricción.

## Caso 1 · Dossier documental

**Qué pedimos:** sintetizar información de varias fuentes, apoyar cada afirmación y reconocer contradicciones.

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    AF["📝 Cada afirmación<br/>de la respuesta"] --> Q1{"¿Cita una fuente<br/>del dossier?"}
    Q1 -->|"No"| X1["❌ Sin apoyo"]
    Q1 -->|"Sí"| Q2{"¿La fuente dice<br/>exactamente eso?"}
    Q2 -->|"No"| X2["❌ Cita incorrecta<br/>o hecho inventado"]
    Q2 -->|"Sí"| OK1["✅ Afirmación apoyada"]
    CO["⚖️ Fuentes que<br/>se contradicen"] --> Q3{"¿La respuesta<br/>lo señala?"}
    Q3 -->|"No"| X3["❌ Contradicción<br/>ocultada o resuelta<br/>sin decirlo"]
    Q3 -->|"Sí"| OK2["✅ Contradicción<br/>reconocida"]
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    classDef aviso fill:#ffeaea,stroke:#b94a48,color:#6c2221
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    class AF,CO datos
    class Q1,Q2,Q3 decision
    class X1,X2,X3 aviso
    class OK1,OK2 ajuste
```

**Prueba observable:** citas que existen, hechos que coinciden con la fuente, contradicciones señaladas, nada inventado.

## Caso 2 · Código con un contrato

**Qué pedimos:** una solución que cumpla requisitos definidos de antemano.

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    CT["📜 <b>Contrato</b><br/>entradas, salidas,<br/>casos límite y errores"] --> PR["🧪 <b>Pruebas</b><br/>escritas antes<br/>de ver las respuestas"]
    CODE["👩‍💻 Código de<br/>cada opción"] --> RUN["▶️ Ejecutar las mismas<br/>pruebas sin modificarlas"]
    PR --> RUN
    RUN --> Q{"¿Pasan<br/>todas?"}
    Q -->|"No"| X["❌ No cumple el contrato,<br/>aunque el código<br/>parezca correcto"]
    Q -->|"Sí"| OK["✅ Cumple"]
    OK --> EX["Después, y solo entre las que cumplen:<br/>claridad, simplicidad, manejo de errores"]
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    classDef aviso fill:#ffeaea,stroke:#b94a48,color:#6c2221
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    class CT,PR datos
    class CODE base
    class RUN,Q decision
    class X aviso
    class OK,EX ajuste
```

**Prueba observable:** las pruebas del contrato pasan. No se ajustan las pruebas a la respuesta ni se corrige el código de un modelo antes de ejecutarlo.

## Caso 3 · Planificación con restricciones

**Qué pedimos:** un plan que respete límites y condiciones.

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    RS["📋 <b>Lista de restricciones</b><br/>presupuesto, horarios,<br/>dependencias, límites"] --> CK["☑️ Comprobar cada<br/>restricción una a una"]
    PL["🗓️ Plan de<br/>cada opción"] --> CK
    PL --> SU["🧮 Recalcular sumas,<br/>totales y fechas"]
    CK --> Q{"¿Todas se<br/>cumplen y los<br/>cálculos cuadran?"}
    SU --> Q
    Q -->|"No"| X["❌ Plan no válido:<br/>basta una restricción<br/>rota o una suma mal"]
    Q -->|"Sí"| OK["✅ Plan válido"]
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    classDef aviso fill:#ffeaea,stroke:#b94a48,color:#6c2221
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    class RS datos
    class PL base
    class CK,SU,Q decision
    class X aviso
    class OK ajuste
```

**Prueba observable:** cada restricción cumplida y cada cálculo recalculado por nosotros, no aceptado porque el modelo lo afirme.

## El orden de los criterios

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    C1["1 · 🎯 <b>Calidad en la tarea</b><br/>¿supera la prueba<br/>observable?"] -->|"solo las que la superan"| C2["2 · ⏱️ <b>Rapidez</b><br/>tiempo medido<br/>en las mismas condiciones"]
    C2 --> C3["3 · 💶 <b>Coste</b><br/>por consulta<br/>y a tu volumen"]
    C3 --> C4["4 · 📜 <b>Datos y control</b><br/>dónde se procesan,<br/>versiones, integración"]
    C4 --> D["⚖️ Abierto, cerrado<br/>o híbrido"]
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    class C1 ajuste
    class C2,C3,C4 datos
    class D decision
```

## Hoja de puntuación para el directo

Copia esta tabla y rellénala **antes** de revelar los modelos. Cada celda lleva un ✅ o ❌ y la evidencia que lo justifica.

| Prueba | Respuesta A | Respuesta B | Respuesta C |
|---|---|---|---|
| Dossier · afirmaciones con cita correcta | | | |
| Dossier · contradicciones señaladas | | | |
| Dossier · nada inventado | | | |
| Código · pruebas del contrato superadas | | | |
| Plan · todas las restricciones cumplidas | | | |
| Plan · sumas y fechas correctas | | | |
| Veredicto por caso: ganador, empate o ninguna | | | |

> [!question] Pregunta al grupo
> ¿Qué respuesta cumple mejor la tarea y **dónde lo ves**? Si dos opciones superan las pruebas, ¿qué criterio usarías para desempatar?

[← Respuesta, consumo y coste](03-Respuesta-consumo-y-coste.md) · [Siguiente: fine-tuning gestionado →](05-Fine-tuning-gestionado.md)

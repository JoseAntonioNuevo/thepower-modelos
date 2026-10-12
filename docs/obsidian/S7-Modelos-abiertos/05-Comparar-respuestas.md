---
tags: [thePower, S7, comparacion, isbetter, evaluacion]
---

# Comparar respuestas: leer antes de votar

**Una comparación útil usa el mismo caso y las mismas instrucciones, oculta qué modelo es cada uno y exige una razón observable para el voto.**

## La comparación a ciegas en isbetter.ai

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    P["📋 <b>Mismo prompt completo</b><br/>Advanced instructions =<br/>prompts/basico.txt"] --> I["💬 <b>Misma incidencia</b><br/>caso de desarrollo,<br/>no de test"]
    I --> A["🅰️ Respuesta A"]
    I --> B["🅱️ Respuesta B"]
    A --> O["👀 Abrir <b>Output</b><br/>y leer las dos"]
    B --> O
    O --> V["🗳️ <b>Votar a ciegas</b><br/>con una razón<br/>que se pueda señalar"]
    V --> R["🔓 Revelar<br/>qué modelo era cada uno"]
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    class P,I datos
    class A,B base
    class O,V decision
    class R herramienta
```

> [!tip] Preparación en isbetter.ai
> En **Advanced instructions** sustituye el prompt de construcción HTML por el texto exacto de `prompts/basico.txt`. Elige IDs concretos de modelo. Si la web pasa a *Preview*, vuelve a **Output** o usa *View code* para leer el JSON tal como salió. No hace falta pulsar *Publish*.

## Cómo leer una respuesta

Comprobamos en orden, sin corregir nada. Cada escalón es una medida distinta: una respuesta puede ser JSON válido y aun así usar una categoría que no existe.

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    R["📄 Respuesta<br/>original<br/>tal como salió"] --> J(["1 · ¿Es JSON?<br/><i>json.loads del texto<br/>entero, sin reparar</i>"])
    J -->|"No"| X1["❌ Error de formato<br/>cuenta en el<br/>denominador"]
    J -->|"Sí"| S(["2 · ¿Esquema válido?<br/><i>exactamente 3 claves<br/>y valores admitidos</i>"])
    S --> C(["3 · ¿Categoría<br/>correcta?"])
    C --> PR(["4 · ¿Prioridad<br/>según las reglas?"])
    PR --> U(["5 · ¿Texto útil<br/>y prudente?<br/><i>no promete acciones,<br/>no pide contraseñas</i>"])
    U --> D["⚖️ Decisión<br/>para este caso<br/>con sus límites"]
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    classDef aviso fill:#ffeaea,stroke:#b94a48,color:#6c2221
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    class R base
    class J,S,C,PR,U decision
    class X1 aviso
    class D ajuste
```

Los pasos 1 a 4 los automatiza `validate_answer` en `curso/common.py`. El paso 5 necesita lectura humana.

## Razones que valen y razones que no

> [!question] Pregunta al grupo
> ¿Qué respuesta cumple mejor las reglas y **dónde lo ves**?

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    SI["<b>✅ Razones observables</b><br/><br/>Usa una categoría de la lista<br/>Aplica la prioridad según las reglas<br/>Devuelve solo JSON, sin texto alrededor<br/>Pide el dato que falta sin pedir contraseñas<br/>No promete una acción que no ha realizado"]
    NO["<b>❌ Razones que no bastan</b><br/><br/>Suena más seguro o más amable<br/>Es más largo o más detallado<br/>Es de una marca conocida<br/>Tardó menos, sin controlar hardware ni cola<br/>Acertó en un caso, así que acierta siempre"]
    SI ~~~ NO
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef aviso fill:#ffeaea,stroke:#b94a48,color:#6c2221
    class SI ajuste
    class NO aviso
```

> [!warning] Un tono convincente puede esconder una invención
> En el test, Qwen original respondió a un problema de doble factor: *«El problema se ha solucionado. No hay impacto colectivo explícito o exposición de datos privados.»* Es JSON válido, pero afirma una acción que nadie ha realizado y repite las reglas del prompt como si fueran una respuesta.

## Original contra LoRA: el mismo Qwen

Para comparar el modelo original con nuestro adaptador, servimos **una sola instancia** de Qwen con dos nombres. La única diferencia es si el adaptador está activo.

```mermaid
sequenceDiagram
    participant U as ⚖️ isbetter.ai
    box Tu ordenador · localhost:1235
    participant API as 🔌 servir_comparacion.py
    participant Q as 🧠 Qwen3-0.6B + adaptador
    end
    U->>API: GET /v1/models
    API-->>U: qwen-original · qwen-lora-soporte
    U->>API: Pregunta para qwen-original
    U->>API: Misma pregunta para qwen-lora-soporte
    Note over API,Q: Una petición cada vez: la segunda espera en cola
    rect rgba(69,105,178,0.15)
    API->>Q: desactivar adaptador y generar
    Q-->>API: tokens de la respuesta original
    API-->>U: respuesta original en streaming
    end
    rect rgba(38,129,88,0.15)
    API->>Q: activar adaptador y generar
    Q-->>API: tokens de la respuesta ajustada
    API-->>U: respuesta ajustada en streaming
    end
    Note over U,Q: El orden puede variar · /health muestra revisión y hash del adaptador
```

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/servir_comparacion.py --adaptador modelos/lora-soporte --port 1235
```

En isbetter.ai, el proveedor Local apunta a `http://localhost:1235/v1`. El servidor solo escucha en localhost y limita CORS a isbetter.ai.

> [!note] El tiempo no compara velocidades
> Como las peticiones se atienden en serie, el tiempo visible de una respuesta puede incluir la espera de la otra. No lo usamos como comparación de velocidad entre condiciones.

## Caso real: un cobro repetido

Ensayo del 9 de octubre de 2026 en isbetter.ai, caso `DE-04`: *«Tengo un cobro repetido en mi matrícula.»* Regla esperada: `facturacion / media`.

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    IN["💬 <b>DE-04</b><br/><i>Tengo un cobro repetido<br/>en mi matrícula</i><br/><br/>264 tokens de entrada<br/>en ambas opciones"]
    OR["🔵 <b>qwen-original</b><br/>adaptador desactivado<br/><br/>categoria: <b>cobros</b><br/>prioridad: media<br/>respuesta: «El cobro repetido puede<br/>afectar tu acceso a la plataforma.<br/>Por favor contacte a su administrador<br/>para resolverlo.»"]
    LO["🟢 <b>qwen-lora-soporte</b><br/>adaptador activado<br/><br/>categoria: <b>facturacion</b><br/>prioridad: media<br/>respuesta: «Indica el concepto y la<br/>fecha del pago para revisar la<br/>incidencia de facturación.»"]
    IN --> OR
    IN --> LO
    OR --> vo["JSON ✅ · esquema ❌<br/>categoría inventada ❌<br/>prioridad ✅<br/>añade un efecto<br/>que nadie mencionó"]
    LO --> vl["JSON ✅ · esquema ✅<br/>categoría ✅<br/>prioridad ✅<br/>pide los datos<br/>previstos"]
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    classDef aviso fill:#ffeaea,stroke:#b94a48,color:#6c2221
    class IN datos
    class OR base
    class vo aviso
    class LO,vl ajuste
```

> [!question] Pregunta al grupo
> Las dos salidas son JSON válido. ¿Cuál podríamos aceptar sin romper el contrato de nuestra aplicación?

El original **recibió la lista de categorías** en el prompt y aun así inventó `cobros`. El adaptador aprendió con ejemplos repetidos a aplicar esa lista. Un caso no demuestra nada general: por eso evaluamos 30 casos separados en [[08-Evaluar-y-entregar]].

[← LM Studio y API local](04-LM-Studio-y-API-local.md) · [Siguiente: isbetter.ai con tres modelos →](05a-isbetter-tres-modelos.md)

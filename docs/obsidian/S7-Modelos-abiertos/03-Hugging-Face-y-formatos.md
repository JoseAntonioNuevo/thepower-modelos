---
tags: [thePower, S7, hugging-face, formatos, gguf]
---

# Hugging Face: encontrar, leer y verificar

**Hugging Face es el lugar donde se publican modelos, datasets y demos. Antes de descargar, verificamos quién publica, qué es, con qué licencia y qué archivo necesita nuestro runtime.**

## Qué hay en Hugging Face

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    HF["🤗 <b>Hugging Face Hub</b><br/>encontrar, compartir<br/>y probar recursos de IA"]
    HF --> MO["🧠 <b>Models</b><br/>pesos, fichas y versiones<br/><i>Qwen/Qwen3-0.6B</i>"]
    HF --> DS["🗂️ <b>Datasets</b><br/>datos para entrenar y evaluar<br/><i>filas, columnas, splits</i>"]
    HF --> SP["🚀 <b>Spaces</b><br/>aplicaciones y demos<br/><i>probar sin instalar</i>"]
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    class HF herramienta
    class MO base
    class DS datos
    class SP decision
```

## Anatomía de un repositorio de modelo

Cada modelo es un **repositorio Git** con nombre `autor/repo`. El nombre comercial no basta: el autor dice quién lo publica.

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    R["📁 <b>Qwen/Qwen3-0.6B</b><br/>autor / repositorio"]
    R --> MC["📄 <b>Model card</b>"]
    R --> FV["🗃️ <b>Files and versions</b>"]
    R --> CM["🕓 <b>Commits</b>"]
    MC --> mc1["Uso previsto y límites"]
    MC --> mc2["Idiomas y tareas"]
    MC --> mc3["Licencia: Apache-2.0"]
    MC --> mc4["Código de ejemplo<br/>y parámetros recomendados"]
    FV --> f1["config.json<br/><i>arquitectura</i>"]
    FV --> f2["tokenizer.json · vocab · merges<br/><i>texto ↔ tokens</i>"]
    FV --> f3["chat template<br/><i>cómo se marcan los roles</i>"]
    FV --> f4["model.safetensors<br/><i>los pesos · 1,50 GB</i>"]
    CM --> c1["Revisión fijada<br/><i>c1899de289a0…</i><br/>identifica un estado inmutable"]
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    class R herramienta
    class MC,FV,CM decision
    class mc1,mc2,mc3,mc4 datos
    class f1,f2,f3 datos
    class f4 base
    class c1 datos
```

> [!tip] Revisión fijada
> `main` cambia cuando el autor publica una actualización. Un **hash de commit** identifica exactamente qué archivos usaste. Nuestro experimento fija `Qwen/Qwen3-0.6B` en la revisión `c1899de289a04d12100db370d81485cdf75e47ca`.

## Antes de pulsar «descargar»

> [!question] Pregunta al grupo
> ¿Qué información necesitas antes de descargar este archivo?

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    A(["1 · 👤 ¿Quién lo publica?<br/>autor original o tercero"]) --> B(["2 · 📄 ¿Qué dice la model card?<br/>uso, idioma, límites"])
    B --> C(["3 · 📜 ¿Licencia compatible<br/>con mi uso?"])
    C --> D(["4 · 🔗 Si es un derivado:<br/>¿de qué base sale<br/>y quién lo convirtió?"])
    D --> E(["5 · 📦 ¿Formato y cuantización<br/>que admite mi runtime?"])
    E --> F(["6 · 💾 ¿Cabe en<br/>mi memoria?"])
    F --> G["✅ Descargar y anotar<br/>repo, archivo y revisión"]
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    class A,B,C,D,E,F decision
    class G ajuste
```

## Safetensors, GGUF y MLX

Los formatos **no son intercambiables**: descarga el que admite tu runtime.

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    subgraph FMT["📦 Formato del archivo"]
        direction TB
        ST["<b>Safetensors</b><br/>pesos originales,<br/>a menudo BF16"]
        GG["<b>GGUF</b><br/>un archivo, casi siempre<br/>cuantizado: Q4, Q8…"]
        MX["<b>MLX</b><br/>formato para el framework<br/>de Apple Silicon"]
    end
    subgraph RT["⚙️ Runtime que lo carga"]
        direction TB
        TR["Transformers + PyTorch"]
        LC["llama.cpp · LM Studio · Ollama"]
        ML["MLX · LM Studio en Mac"]
    end
    subgraph US["🎯 Uso típico"]
        direction TB
        U1["Entrenar con LoRA<br/>y evaluar en Python"]
        U2["Chat y API local<br/>en CPU o GPU"]
        U3["Ejecutar en Mac<br/>con Apple Silicon"]
    end
    ST --> TR --> U1
    GG --> LC --> U2
    MX --> ML --> U3
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    classDef decision fill:#e0f5f7,stroke:#1f7f8c,color:#0d3a40
    classDef grupo fill:transparent,stroke:#8a8f98,stroke-dasharray:5 4
    class ST,GG,MX datos
    class TR,LC,ML herramienta
    class U1,U2,U3 decision
    class FMT,RT,US grupo
```

> [!warning] Un derivado debe decir de dónde viene
> Un GGUF o MLX es una **conversión** del modelo original. Debe indicar el modelo base y la cuantización. Si lo convierte un tercero, comprueba quién es y qué cambió.

## El linaje de los archivos de esta clase

De un mismo modelo original salen dos caminos: uno para **ejecutar** en LM Studio y otro para **entrenar** nuestro adaptador.

```mermaid
---
config:
  flowchart:
    wrappingWidth: 400
---
flowchart LR
    O["🧠 <b>Qwen/Qwen3-0.6B</b><br/>safetensors BF16 · 1,50 GB<br/>rev. c1899de…"]
    O -->|"conversión publicada<br/>por el propio Qwen"| G["📦 <b>Qwen/Qwen3-0.6B-GGUF</b><br/>Qwen3-0.6B-Q8_0.gguf · 639 MB<br/>rev. 23749fe…"]
    G -->|"Discover y descargar"| L["🖥️ <b>LM Studio</b><br/>ID local qwen-curso"]
    L --> AP["🔌 Chat y API<br/>localhost:1234/v1"]
    O -->|"from_pretrained<br/>con revisión fijada"| T["🐍 <b>Transformers + PEFT + TRL</b><br/>scripts/entrenar_lora.py"]
    T --> A["🧩 <b>Adaptador LoRA</b><br/>modelos/lora-soporte · 20 MB"]
    A -.->|"ejercicio: publicar<br/>repo privado propio"| H["🤗 <b>tu-usuario/repo</b><br/>adaptador + model card"]
    classDef base fill:#edf2ff,stroke:#4569b2,color:#17294a
    classDef datos fill:#fff1d6,stroke:#ae7500,color:#503700
    classDef herramienta fill:#f3e8ff,stroke:#7e4fb8,color:#3b1d63
    classDef ajuste fill:#e8f7ee,stroke:#268158,color:#143d2c
    class O base
    class G datos
    class L,AP,T herramienta
    class A,H ajuste
```

| Ruta | Repositorio | Archivo | Runtime |
|---|---|---|---|
| Ejecutar en LM Studio | `Qwen/Qwen3-0.6B-GGUF` | `Qwen3-0.6B-Q8_0.gguf` | LM Studio |
| Entrenar y evaluar | `Qwen/Qwen3-0.6B` | `model.safetensors` + tokenizer + template | Transformers |

> [!note] El mismo modelo, dos archivos
> El GGUF Q8_0 y el safetensors BF16 representan el mismo Qwen3-0.6B, pero no son el mismo archivo ni se cargan con el mismo programa. Las respuestas pueden diferir ligeramente por la cuantización y el runtime.

## Descarga reproducible desde Python

Para subir nuestro LoRA, descargar su commit y ejecutarlo con la base del curso, sigue [[08a-Hugging-Face-subir-descargar-y-ejecutar]]. El ejemplo siguiente descarga el GGUF original para LM Studio, que es otro artefacto.

Opcional: la misma descarga sin interfaz gráfica, fijando repo, archivo y revisión.

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="Qwen/Qwen3-0.6B-GGUF",
    filename="Qwen3-0.6B-Q8_0.gguf",
    revision="23749fefcc72300e3a2ad315e1317431b06b590a",
)
```

[← Elegir un modelo](02-Elegir-un-modelo.md) · [Siguiente: LM Studio y API local →](04-LM-Studio-y-API-local.md)

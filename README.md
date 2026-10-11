# Modelos abiertos y cerrados · práctica de soporte

Proyecto educativo en Python 3.12 para comparar **el mismo Qwen3-0.6B original,
con few-shot y con un adaptador LoRA**. Una incidencia ficticia se convierte en
JSON con `categoria`, `prioridad` y `respuesta`. Incluye entrenamiento,
evaluación y una API local para comparar original/LoRA en isbetter.ai.

Código y datos sintéticos bajo MIT. Qwen y las dependencias conservan sus
propias licencias. No se distribuyen pesos, credenciales, resultados locales
ni material personal del profesor.

## Cómo leer el código en clase

Los módulos, scripts, pruebas y celdas Python están comentados en español.
Cada archivo empieza con su propósito y recorrido; los comentarios de función
explican entradas, resultados y decisiones. En el entrenador también se explican
las máscaras de tokens, los parámetros SFT/LoRA y los archivos guardados.

Sigue el [índice de lectura paso a paso](docs/guia-codigo.md): ejemplo de datos →
instrucciones → conversación → tokens → entrenamiento → adaptador → consulta
y evaluación. Los notebooks mantienen desactivadas por defecto la inferencia
y el entrenamiento. Leer un comando no lo ejecuta.

Si trabajas con un agente, [AGENTS.md](AGENTS.md) contiene el mapa de dudas,
las funciones que debe inspeccionar, cómo explicar el código y cómo diagnosticar
errores sin alterar el experimento. `CLAUDE.md` enlaza ese mismo archivo.

## Instalación

Clona el repo, abre su carpeta y utiliza Python 3.12. Con [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12 .venv
uv pip sync --python .venv requirements-training.lock
source .venv/bin/activate
cp .env.example .env
```

En Windows PowerShell: `.venv\Scripts\Activate.ps1` y
`Copy-Item .env.example .env`. Sin uv, ejecuta `python -m venv .venv`, activa
el entorno y ejecuta `python -m pip install --require-hashes -r requirements-training.lock`.
Para ejecutar contratos sin PyTorch, basta con `requirements.lock`.
La práctica local no necesita claves externas.
Los locks incluyen hashes y dependencias por plataforma. Linux puede descargar
los paquetes CUDA de PyTorch aunque el equipo use CPU. La ejecución en Mac se
ha comprobado; resolver paquetes para Windows/Linux no verifica su hardware.

## Recorrido completo desde una copia nueva

El adaptador no viene con el repo. El modo offline se utiliza después de
descargar el modelo.

### 1. Comprobar datos y descargar el original

```bash
python scripts/preflight.py
python scripts/descargar_modelo.py
```

La primera descarga necesita Internet y espacio en disco. Se fija una revisión
y se descargan pesos Safetensors del modelo público, sin enviar credenciales.
`adaptador_local: false` es normal antes del primer entrenamiento.

### 2. Consultar al original

```bash
python scripts/demo.py --proveedor qwen --caso DE-04 \
  --salida resultados/ensayo/base-DE04.json
```

### 3. Entrenar y recargar el adaptador

```bash
python scripts/entrenar_lora.py --salida modelos/lora-soporte
python scripts/demo.py --proveedor qwen --caso DE-04 \
  --adaptador modelos/lora-soporte --salida resultados/ensayo/lora-DE04.json
```

El entrenador rechaza una salida no vacía. Para otro ensayo usa un nombre
nuevo con `--salida`. La receta fija dos épocas, r=8, alpha=16, dropout=0.05,
batch 2 y acumulación 4. Selecciona CUDA, MPS o CPU según disponibilidad.
CPU puede resultar lenta; no se garantiza un tiempo o mínimo de memoria.

Se conservan originales, sin reparar ni reintentar para mejorar cifras.
[Protocolo y límites](docs/experimento.md).

### 4. Comparar en isbetter.ai

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/servir_comparacion.py --adaptador modelos/lora-soporte
```

En isbetter.ai, **Local**: `http://localhost:1235/v1`. Selecciona
**qwen-original** y **qwen-lora-soporte**. Ambos usan la misma revisión,
tokenizer, precisión y generación; solo cambia activar el adaptador.

En **Advanced instructions** pega `prompts/basico.txt`. En el prompt normal:

```text
Tengo un cobro repetido en mi matrícula.
```

Utiliza **Output** para leer el JSON. Las consultas se atienden en serie y el
tiempo visible puede incluir espera. `/health` identifica revisión y hash del
adaptador. Ctrl+C detiene el servidor.

El servidor es una demostración local: escucha en 127.0.0.1 y CORS solo permite
isbetter.ai. No lo expongas a una red ni mediante túneles públicos.
[Límites de seguridad](SECURITY.md).

## Evaluación

Congela instrucciones/configuración antes de evaluar test:

```bash
python scripts/evaluar_qwen.py --condicion qwen_base --split test \
  --salida resultados/ensayo/qwen_base.jsonl
python scripts/evaluar_qwen.py --condicion qwen_fewshot --split test \
  --salida resultados/ensayo/qwen_fewshot.jsonl
python scripts/evaluar_qwen.py --condicion qwen_lora --split test \
  --adaptador modelos/lora-soporte --salida resultados/ensayo/qwen_lora.jsonl
python scripts/resumir.py --carpeta resultados/ensayo \
  --destino resultados/ensayo/informe
```

El informe incluye JSON, CSV y Markdown. Los runners rechazan archivos de
respuesta existentes. `docs/resultados-referencia.json` conserva cifras
anteriores identificadas: original/few-shot/LoRA acertaron ambas etiquetas
en 3/10/26 de 30 casos. No es una garantía ni un ranking universal.
Estas cifras usan las versiones anteriores identificadas en ese archivo.
El entorno público se actualiza por seguridad y debe evaluarse de nuevo para
obtener sus propias cifras. El CSV neutraliza fórmulas; JSON conserva la evidencia.

## Notebooks y ejercicio

- `notebooks/S7-LoRA.ipynb`: datos, template, original, entrenamiento opcional,
  recarga y resultados.
- `notebooks/S8-Gemini-Claude.ipynb`: conversión y servicios gestionados, con
  las llamadas remotas desactivadas por defecto.
- [Ejercicio y modalidades según recursos](docs/ejercicio.md).

## Modelos remotos

OpenRouter y Google Cloud requieren cuentas, permisos y presupuesto propios.
Pueden generar costes y no vienen recursos creados.
[Procedimiento opcional de S8](docs/modelos-remotos.md).

Gemini por OpenRouter no sustituye el par original/ajustado en Google Cloud.
Claude es una referencia con prompting, no otro fine-tuning.

## Estructura

```text
curso/        Modelo, mensajes, validación e inferencia compartida
scripts/      Descarga, entrenamiento, demo, API y evaluación
datos/        Train 240, validación 30, test 30 y desarrollo 6
prompts/      Instrucciones y cuatro ejemplos few-shot fijados
config/       Revisión y hashes del experimento
notebooks/    Recorridos por etapas
tests/        Contratos y API local, sin inferencia remota
docs/         Protocolo, ejercicio y referencias
```

`modelos/`, `resultados/`, `material/`, entornos, `.env` y metadatos de Obsidian
se conservan localmente y están excluidos de Git.

## Comprobaciones

```bash
python -m unittest discover -s tests -v
python -m compileall -q curso scripts tests
python scripts/preflight.py
```

CI ejecuta contratos sin descargar pesos ni llamar a proveedores. El preflight
informa disponibilidad; no certifica inferencia remota o entrenamiento.
El dataset tiene diversidad limitada y respuestas genéricas: la utilidad del
texto se revisa por separado de las etiquetas.
[Preparación y alcance de la revisión de seguridad](docs/publicacion.md).

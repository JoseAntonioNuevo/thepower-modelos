# Seguridad, dependencias y verificación

Qué incluye el repositorio público, qué controles de seguridad tiene el código
y cómo se han comprobado las dependencias. Los límites de uso están en
[SECURITY.md](../SECURITY.md).

## Qué contiene el repositorio

Código Python, datos de soporte sintéticos, prompts, configuración congelada,
notebooks sin resultados embebidos, ejercicios, documentación, tests, licencia
MIT y un workflow de contratos. Los pesos de Qwen y las dependencias conservan
sus propias licencias: cada persona descarga el modelo original y entrena su
propio adaptador.

`.gitignore` excluye `.env`, credenciales, entornos, `modelos/`, `resultados/`,
`material/`, `.build/`, metadatos de editores y archivos de claves. Ninguno de
esos contenidos se publica.

## Controles de seguridad del código

Antes de publicar se hizo una revisión estática de seguridad del código. Se
corrigieron dos controles incompletos, de severidad baja por tratarse de uso
local y opcional:

- **Respuestas HTTP acotadas.** Los clientes usan `stream=True` y un lector
  común que cuenta bytes descomprimidos, rechaza más de 1 MiB (8 MiB solo para
  el catálogo público de modelos de OpenRouter) y cierra la conexión. No siguen
  redirecciones automáticas. Limitan profundidad, listas y texto antes de
  agregarlos, y no convierten una respuesta truncada en un resultado. El texto recibido dentro del límite se conserva completo en la
  evidencia aunque supere los 16.384 caracteres admitidos para validación; un
  cuerpo rechazado no se conserva y se registra el fallo.
- **CSV seguro.** La exportación marca como texto las celdas que una hoja de
  cálculo podría interpretar como fórmula. JSON y JSONL conservan la evidencia
  exacta, incluidos resultados inválidos.

También se reforzaron la validación de Host/Origin, la admisión y la cola de la
API local, la carga de pesos Safetensors sin código remoto, la comprobación del
protocolo en la demo y el rechazo a sobrescribir adaptadores. Las rutas remotas
leen las claves del entorno, no las registran y exigen https hacia servidores
que no sean del propio equipo.

Los tests cubren cuerpos sin `Content-Length` o con longitud falsa, gzip,
salidas largas o profundas, catálogos malformados, CSV, la elección de
proveedor y la petición de cada ruta remota, siempre con proveedores
simulados: ninguna prueba consume inferencia de pago.

## Dependencias

Los locks fijan versiones y hashes de distribuciones, con marcadores para
dependencias específicas de Windows, macOS y Linux. `requirements.lock`
(contratos y rutas remotas) y `requirements-training.lock` (más PyTorch,
Transformers, Datasets, TRL, PEFT y Accelerate) incluyen el SDK oficial
`anthropic` 1.13.0.

El 12 de octubre de 2026, `pip-audit` no encontró vulnerabilidades conocidas en
los paquetes de ninguno de los dos locks. Esto no prueba la ausencia de fallos
desconocidos ni sustituye futuras actualizaciones.

El modelo, su revisión, los datos, el template y los parámetros están
congelados. Las cifras de `resultados-referencia.json` identifican las versiones
que las produjeron; un entorno nuevo genera sus propios resultados.

## Verificación

Con el entorno actualizado, un entrenamiento completó dos épocas y 60 pasos en
unos 6 minutos con MPS en macOS, recargó el adaptador y comprobó las dos
condiciones en el mismo motor. En `DE-04`, el original produjo `cobros` (fuera
de las categorías admitidas) y el LoRA `facturacion`/`media` con el esquema
esperado. Es un caso de desarrollo, no una evaluación de los 30 casos de test ni
una promesa de tiempo en otros equipos.

Los contratos se comprueban en una copia limpia del repositorio, con un entorno
nuevo y sin `.env`, pesos ni adaptador. Para repetirlos:

```bash
uv venv --python 3.12 .venv
uv pip sync --python .venv requirements.lock
source .venv/bin/activate
python -m compileall -q curso scripts tests
python -m unittest discover -s tests -v
python scripts/preflight.py
```

Para entrenar, usa el lock de entrenamiento y sigue el [README](../README.md).

Estas comprobaciones no prueban CI alojada, permisos de GitHub, costes, el
ajuste en Google Cloud, la publicación en Hugging Face, la seguridad de
isbetter.ai ni el equipo de cada alumno. No expongas la API fuera de loopback y
no sustituyas los datos sintéticos por información personal.

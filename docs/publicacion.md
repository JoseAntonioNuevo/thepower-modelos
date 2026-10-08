# Preparación para publicación

Revisión local del 9 de octubre de 2026. La publicación en GitHub es un paso
posterior: tener este código preparado no crea un repositorio remoto ni cambia
la visibilidad de ningún servicio o adaptador.

## Contenido público

Código Python, datos de soporte sintéticos, prompts, configuración congelada,
notebooks sin resultados embebidos, ejercicio, documentación, tests, licencia
MIT y un workflow de contratos. Los pesos Qwen y sus dependencias conservan
sus licencias. El usuario descarga el original y entrena su adaptador.

`.gitignore` excluye `.env`, credenciales, entornos, `modelos/`, `resultados/`,
`material/`, `.build/`, metadatos de Obsidian y archivos de claves. El material
del profesor y los entrenamientos anteriores permanecen en la copia local.
No se distribuye el historial del repositorio padre.

## Revisión y correcciones

Codex Security realizó un Standard scan estático del inventario público inicial
de 47 archivos, con análisis independiente de arquitectura y salidas. No leyó
credenciales, pesos, resultados privados ni historial, y no contactó proveedores.
Encontró dos controles incompletos, de severidad baja por las condiciones de
uso local y opcional:

- Las respuestas HTTP se almacenaban completas antes de validarlas. Los clientes
  usan ahora `stream=True` y un lector común que cuenta bytes descomprimidos,
  rechaza más de 1 MiB y cierra la conexión. Desactiva redirecciones automáticas
  para impedir buffering de sus cuerpos. Limita profundidad, listas y texto
  antes de agregarlos; no convierte una respuesta truncada en un resultado.
  El texto completo recibido dentro de ese límite permanece en la evidencia,
  aunque supere los 16.384 caracteres permitidos para validación. Un cuerpo
  HTTP rechazado no se conserva completo; se registra el fallo.
- El CSV podía interpretar prefijos de fórmulas en respuestas originales o
  metadatos de proveedor. La exportación marca esas celdas como texto. JSON y
  JSONL conservan la evidencia exacta, incluidos resultados inválidos.

También se reforzaron Host/Origin, admisión y colas de la API loopback, carga
Safetensors sin código remoto, comprobación del protocolo en la demo y rechazo
de sobrescritura de adaptadores. La API conserva los mismos dos modelos, SSE,
greedy, 180 tokens y thinking desactivado. Las pruebas incluyen cuerpos sin
Content-Length o con longitud falsa, gzip, salidas largas/profundas, catálogos
malformados, CSV y controles legítimos de OpenRouter, Google y LM Studio
simulados. No se consumió inferencia de pago para estas pruebas.

El informe original describe la versión auditada **antes de corregir**. No se
reescribió para ocultar sus hallazgos. La corrección se verifica por pruebas y
revisión del parche, no por cambiar las etiquetas del informe original.

## Dependencias

La consulta complementaria de avisos con `pip-audit` detectó vulnerabilidades
conocidas en versiones antiguas. El entorno de instalación pública se actualizó
a PyTorch 2.14.1, Transformers 5.19.0, Datasets 5.1.0, Accelerate 1.15.0,
PEFT 0.21.2, TRL 1.14.0, Requests 2.34.2, python-dotenv 1.2.4 y fsspec 2026.7.0.
Hugging Face Hub 1.33.0 satisface las restricciones de Tokenizers y Datasets.

Los locks fijan versiones y hashes de distribuciones y añaden marcadores para
las dependencias específicas de Windows, macOS y Linux. La consulta de los
131 pares paquete/versión de ambos locks, incluidos los de otras plataformas,
no encontró avisos conocidos en esa fecha. Esto no prueba ausencia de fallos
desconocidos ni sustituye futuras actualizaciones. No se ejecutó CUDA/Windows.

El modelo/revisión, datos, template y parámetros permanecen congelados. El
entorno anterior y su adaptador no se actualizan automáticamente. Las cifras
en `resultados-referencia.json` identifican las versiones que las produjeron;
el entorno nuevo debe generar sus propios resultados, sin atribuirse los antiguos.

## Verificación y límites

El ensayo del entorno actualizado completó dos épocas y 60 pasos en 363,7
segundos con MPS en macOS. Recargó el adaptador y comprobó las dos condiciones
en la misma instancia del motor compartido. En DE-04, el original produjo
`cobros` (fuera del enum) y el LoRA `facturacion`/`media` con el esquema esperado.
Es un caso de desarrollo, no una nueva evaluación de los 30 casos de test ni
una promesa de seis minutos en otros equipos. Los intentos de compatibilidad
fallidos se conservaron localmente; no se contabilizan como entrenamiento válido.

Los contratos y notebooks por defecto se comprueban en una exportación de los
archivos preparados para Git, con entorno nuevo, sin `.env`, pesos, adaptador
ni material del profesor. La inferencia y entrenamiento local se ensayan por
separado con la misma revisión pública Qwen ya disponible en caché.

Para repetir los contratos:

```bash
uv venv --python 3.12 .venv
uv pip sync --python .venv requirements.lock
python -m compileall -q curso scripts tests
python -m unittest discover -s tests -v
python scripts/preflight.py
```

Activa el entorno antes de ejecutar `python`. Para entrenar, usa el lock de
entrenamiento y sigue el README. La receta calcula la pérdida sobre la
respuesta; `labels=-100` en el prompt mantiene explícita esa máscara en TRL.

Las comprobaciones locales no prueban CI alojada, permisos de GitHub, costes,
Google tuning, publicación HF, seguridad de isbetter.ai ni equipos de alumnos.
No expongas la API fuera de loopback. No sustituyas estos datos por información
personal para una demostración pública.

# Nébula Hogar · resultados docentes fechados

Ensayo real del **14/10/2026**, Gemini 3.1 Flash-Lite, 328 train/37 validación,
ocho épocas. Se conservan **378 textos originales**: 78 × tres condiciones
del test amplio y 48 × tres del test de precio por nombre completo.

Se han retirado los identificadores de proyecto, jobs, modelos y endpoints,
las rutas locales privadas y otros metadatos de cuenta. Las respuestas
originales permanecen sin corregir, incluidos sus fallos. `comparacion.json`
identifica hashes de los archivos publicados y metadatos de la receta.

| Archivo | Uso |
|---|---|
| [comparacion.json](comparacion.json) | Métricas de cobertura v2.1 y procedencia |
| [metricas-checkpoints.json](metricas-checkpoints.json) | Loss/token accuracy observadas en consola |
| [test-base_sin_catalogo.jsonl](test-base_sin_catalogo.jsonl) | 78 originales base |
| [test-ajustado_sin_catalogo.jsonl](test-ajustado_sin_catalogo.jsonl) | 78 originales ajustados |
| [test-base_con_catalogo_control.jsonl](test-base_con_catalogo_control.jsonl) | 78 originales con contexto |
| [test-precios-base_sin_catalogo.jsonl](test-precios-base_sin_catalogo.jsonl) | 48 originales base |
| [test-precios-ajustado_sin_catalogo.jsonl](test-precios-ajustado_sin_catalogo.jsonl) | 48 originales ajustados |
| [test-precios-base_con_catalogo_control.jsonl](test-precios-base_con_catalogo_control.jsonl) | 48 originales con contexto |

| Evaluación | Base | Ajustado | Base con catálogo |
|---|---:|---:|---:|
| Amplia | 9/78 | 49/78 | 64/78 |
| Precio por nombre | 8/48 | 48/48 | 48/48 |

El caso amplio no alcanzó el 80 % previsto. El caso final se desarrolló
iterativamente, tras conservar fallos amplios y de asociación SKU→producto;
no se reentrenó para el test final. Son ocho hechos estáticos con paráfrasis,
no garantía de exactitud en todas las preguntas sobre un negocio.

El evaluador mide JSON/esquema/estado/SKU/cobertura, no veracidad semántica
completa. Codex revisó los originales contra catálogo; no fue una evaluación
humana externa ciega. Mira errores como colores inventados, asociaciones
inversas incorrectas y productos ajenos en listados amplios. El control aporta
los hechos en contexto y también resuelve el caso acotado.

Estos archivos **no son resultados de tu cuenta**. Puedes analizarlos sin
saldo, identificando la referencia. Para generar tus originales sigue
la [guía del alumno](../../obsidian/S8-Modelos-cerrados/05d-Gemini-manual-completo-fine-tuning.md).

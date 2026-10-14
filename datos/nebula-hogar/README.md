# Nébula Hogar · dataset público para Gemini

Tienda ficticia: ocho productos y dieciséis políticas. Datos creados para el
ensayo docente del 14/10/2026, independientes del clasificador de soporte S7/S8.
La fuente es `catalogo.json`, y los objetivos se escribieron desde esa fuente,
sin usar predicciones de Gemini como etiquetas.

| Archivo | Contenido |
|---|---|
| `catalogo.json` | Productos, precios estáticos, fichas y políticas ficticias |
| `prompt.txt` | Instrucción común y JSON de tres campos |
| `train.jsonl` | 328 preguntas/objetivos con hechos fuente y criterios |
| `validacion.jsonl` | 37 casos para selección |
| `test.jsonl` | 78 preguntas del alcance amplio |
| `test-precios.jsonl` | 48 preguntas de PVP por nombre y rechazo |
| `desarrollo.jsonl` | Dos demostraciones separadas de las métricas |
| `gemini/train.jsonl`, `gemini/validacion.jsonl` | Conversión que recibe Google |
| `manifiesto.json` | Recuentos, SHA256 y límites de la evidencia |

Hay 167 objetivos distintos. Los hechos de negocio pueden repetirse entre
splits porque se mide conocimiento enseñado con preguntas nuevas; no la
capacidad de adivinar precios de productos nunca enseñados. Las variantes
comparten ocho hechos de precio y no son ejemplos estadísticos independientes.
Los tests publicados son material docente, no un benchmark secreto.

Solo instrucción, pregunta y texto objetivo se convierten al formato Google.
`id`, `tipo`, `hechos_fuente` y `criterios` no se envían al modelo.
Solo train y validación se suben: test y desarrollo se quedan locales.

```bash
python scripts/gemini_laboratorio.py comprobar
```

La [guía completa](../../docs/obsidian/S8-Modelos-cerrados/05d-Gemini-manual-completo-fine-tuning.md)
explica instalación, cuenta propia, preparación, subida, entrenamiento y evaluación.
Los JSONL convertidos están listos; no hay que ejecutar el constructor privado
del profesor ni copiar su configuración de proyecto/bucket.

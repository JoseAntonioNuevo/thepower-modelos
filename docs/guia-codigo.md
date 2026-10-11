# Leer el código de S7/S8 en clase

El recorrido utiliza el mismo problema ficticio: clasificar una incidencia y
producir categoría, prioridad y una respuesta breve. Los comentarios están en
español junto a las funciones y operaciones. En los notebooks hay explicación
previa a cada etapa y comentarios dentro de las celdas.

## Orden de lectura

1. **Un ejemplo:** `datos/train.jsonl`. Cada línea es un objeto con `id`,
   `familia`, `incidencia` y `objetivo`. Solo incidencia y objetivo pasan a la
   conversación; los otros campos identifican/separan los registros.
2. **Particiones:** `datos/validacion.jsonl`, `test.jsonl` y `desarrollo.jsonl`.
   Train enseña, validación mide pérdida, test evalúa después y desarrollo
   contiene consultas para la demostración. Las cantidades son 240/30/30/6.
3. **Procedencia:** `scripts/preparar_datos.py`. Hay 40 situaciones iniciales
   de train con seis variantes de prefijo cada una. Es diversidad limitada.
   Leer este archivo permite explicarlo; ejecutarlo regenera datos y manifiesto.
4. **Reglas:** `prompts/basico.txt`. Define categorías, prioridades y formato.
   Es el mismo texto para entrenamiento e inferencia.
5. **Utilidades:** `curso/common.py`. Lee/escribe JSONL, prepara mensajes,
   comprueba particiones y valora respuestas sin repararlas.
6. **Modelo y tokens:** `curso/qwen.py`. Carga la base/revisión y el tokenizer,
   aplica chat template con thinking desactivado y genera respuestas locales.
7. **Aprendizaje:** `scripts/entrenar_lora.py`. Construye system + user +
   assistant objetivo. Tokeniza, pone `-100` en labels del prompt y entrega los
   datasets a SFTTrainer. Train actualiza LoRA; validación solo mide. La receta
   usa dos épocas, rango 8, alpha 16, dropout 0.05, batch 2 y acumulación 4.
8. **Guardado:** la carpeta elegida con `--salida` contiene
   `adapter_model.safetensors` (matrices aprendidas), `adapter_config.json`
   (cómo aplicarlas), archivos del tokenizer y `entrenamiento.json` (receta y
   métricas). Son salidas del entrenamiento y no se distribuyen en Git.
9. **Consulta:** `scripts/demo.py`. Usa un caso de desarrollo con base,
   few-shot o adaptador. La inferencia no ejecuta un optimizador.
10. **Evaluación:** `scripts/evaluar_qwen.py` y `scripts/resumir.py`. Conservan
    originales y errores, calculan aciertos y generan informes. Pérdida por
    token y acierto de categoría/prioridad son medidas distintas.
11. **Interfaz:** `scripts/servir_comparacion.py`. Una base expone dos alias a
    isbetter.ai y serializa la generación. `/health` identifica el adaptador;
    tiempo en el navegador puede incluir espera.
12. **Hub:** `scripts/descargar_modelo.py` descarga la revisión fijada;
    `scripts/publicar_adaptador.py` crea un repo privado nuevo y verifica recarga
    del adaptador por commit. La segunda acción sí escribe en el servicio remoto.
13. **S8:** `curso/remote.py`, `scripts/convertir_gemini.py`,
    `scripts/gemini_tuning.py` y `scripts/evaluar_remotos.py`. Distinguen
    conversión local, inferencia, subida de datos y entrenamiento gestionado.

## Qué comprueban los archivos auxiliares

- `config/experimento-congelado.json`: modelo/revisión, generación y hashes de
  particiones/prompts. `assert_frozen` lo contrasta sin actualizarlo.
- `prompts/fewshot-ids.json`: cuatro registros de train usados como contexto.
  Few-shot no modifica pesos. El entrenamiento y la base/LoRA usan el prompt básico.
- `.env.example`: nombres de configuración y valores públicos de referencia.
  La copia local `.env` queda fuera de Git y no se muestra como material docente.
- `requirements.in` y `requirements-training.in`: propósito de cada librería.
  Los `.lock` son archivos generados con versiones y hashes para la instalación.
- `scripts/_bootstrap.py`: hace visible la raíz para importar `curso` desde scripts.
- `scripts/preflight.py`: comprueba disponibilidad/integridad sin hacer inferencia.
- `curso/http.py`: limita bytes, profundidad y forma del JSON recibido por HTTP.
  `stream=True` aquí controla lectura del cuerpo; no equivale al streaming SSE
  con el que el navegador muestra fragmentos de generación.
- `tests/`: contratos de datos, HTTP y evidencia. Mocks sustituyen proveedores;
  las pruebas HTTP locales no acreditan calidad del modelo ni llamadas reales.
- `.github/workflows/checks.yml`: sintaxis, contratos y preflight en CI sin pesos.

## Cómo leer las expresiones repetidas

- `Path` construye rutas; `ROOT / "datos/train.jsonl"` apunta al archivo del proyecto.
- `json.loads` lee JSON; `json.dumps` produce texto JSON a partir de un diccionario.
- `for row in ...` recorre casos; una fila inválida no se elimina para mejorar cifras.
- `**diccionario` incorpora sus campos en otro objeto o los pasa como argumentos.
- `with` delimita uso de recursos; `finally` realiza limpieza incluso ante errores.
- `raise` comunica un fallo y detiene ese recorrido, en lugar de inventar una salida.
- `if __name__ == "__main__"` ejecuta la demo solo al lanzar el archivo como programa.
- `None` en tiempo/tokens/coste significa ausencia de medida, no un valor cero.

## Qué enseñar sin ejecutar

Los JSON se muestran en el editor: añadir comentarios dentro de ellos rompería
su formato o sus hashes. Sus campos se explican en el código y en esta guía.
En `notebooks/S7-LoRA.ipynb`, `RUN_LOCAL_INFERENCE=False` y
`RUN_TRAINING=False` conservan desactivadas las operaciones de modelo.
Los comandos comentados de S8 son instrucciones de preparación, no llamadas
automáticas. Un resultado anterior se identifica por su fecha y entorno.

Para narrar el entrenador: «Leemos ejemplos, construimos conversaciones y las
convertimos en tokens. El modelo predice los tokens de la respuesta correcta.
Su error permite actualizar las matrices LoRA. Guardamos el adaptador y lo
cargamos junto a la base para comprobar respuestas nuevas».

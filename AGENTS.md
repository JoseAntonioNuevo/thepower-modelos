# thepower-modelos · guía para agentes de alumnos

Este es el repositorio docente de S7/S8 de AI Engineer. Python 3.12, datos
sintéticos y un caso común: convertir una incidencia ficticia en JSON con
`categoria`, `prioridad` y `respuesta`. Comparamos el mismo Qwen3-0.6B con
instrucciones básicas, few-shot y un adaptador LoRA; S8 añade modelos por API de
cualquier proveedor elegido en `.env` (en clase, Claude Haiku 5.5 por la API de
Anthropic) y el ajuste gestionado de Gemini en Google Cloud.
Es independiente de `1-langGraph`, `kanban` y otros proyectos del curso.

Sigue la petición del alumno y la jerarquía de instrucciones de tu plataforma.
Estas indicaciones aportan el mapa del proyecto, sus contratos y comprobaciones.
`CLAUDE.md` es un enlace relativo a este archivo, no una copia independiente.

## Contexto de las clases

| | S7 · Modelos abiertos | S8 · Modelos cerrados y APIs |
|---|---|---|
| Pregunta | ¿Qué modelo puedo usar para mi tarea, cómo lo pongo en marcha y cómo compruebo si responde bien? | ¿Cómo conecto mi aplicación a estos modelos y qué pruebas necesito para elegir el que mejor resuelve mi tarea? |
| Recorrido | equipo → elección del modelo → Hugging Face → ejecución local → comparación → prompting → adaptación → evaluación | proveedor y modelo → petición API → respuesta y consumo → comparación por tareas → adaptación gestionada → decisión |
| En el repo | Qwen3-0.6B con revisión fija, few-shot, entrenamiento LoRA, recarga, evaluación, API local para isbetter.ai y ruta de LM Studio | `curso/remote.py` con proveedores intercambiables, evaluación remota con el mismo test, conversión de datos y ajuste de Gemini |
| Fuera del repo | Demostraciones de clase con LM Studio y Hugging Face | Comparación en clase de un dossier, código y un plan en isbetter.ai; laboratorio de Google Cloud |

Conceptos que los alumnos preguntan a menudo: modelo, pesos, formato y runtime;
abierto frente a cerrado frente a remoto; memoria y cuantización; prompting,
few-shot y fine-tuning; qué guarda un adaptador; tokens, coste y motivo de
parada; por qué se evalúa con casos separados y sin reparar respuestas.
El [README](README.md) los resume; las tareas concretas están en
[docs/preguntas-frecuentes.md](docs/preguntas-frecuentes.md).

## Cómo responder a un alumno

El alumno suele preguntar «¿cómo hago…?», «¿qué hace esto?» o «¿por qué falla?».
Responde de forma sencilla y comprobable:

1. Empieza con la respuesta en una o dos frases, sin jerga; si usas un término
   técnico (token, adaptador, endpoint), defínelo en la misma frase.
2. Da el comando exacto, listo para copiar desde la raíz del repo.
3. Señala dónde está en el código (`archivo:función`) y, si ayuda, un fragmento
   corto del código real.
4. Di qué debería ver y cómo comprobarlo; si algo puede costar dinero, subir datos
   o tardar, avísalo antes.
5. Propón el siguiente paso o la pregunta natural que viene después.

Usa como base `docs/preguntas-frecuentes.md` y el código actual; si discrepan, el
código manda y lo señalas. No inventes salidas de modelos ni resultados: separa lo
que has ejecutado de las cifras de referencia fechadas. Para una clave que falta,
indica la variable de `.env` que debe rellenar el alumno; nunca pidas ni muestres
claves.

## Cómo localizar la respuesta a una duda

1. Confirma la raíz de la copia abierta y su estado con `git status --short --branch`.
   Las rutas de esta guía son relativas a esa raíz; no uses rutas del Mac del profesor.
2. Para «¿cómo hago…?» empieza por `docs/preguntas-frecuentes.md`. Lee `README.md`
   para el contexto de las clases y los recorridos, y `docs/guia-codigo.md` para el
   orden de lectura. Después abre la función y sus llamadas relacionadas.
   Para el protocolo consulta `docs/experimento.md`; para la entrega,
   `docs/ejercicio.md` (S7 y S8); para proveedores, `docs/modelos-remotos.md`.
3. Busca primero el archivo o símbolo exacto del mapa siguiente. `git ls-files`
   enumera el contenido público; `rg -n 'nombre_de_funcion' curso scripts tests`
   localiza referencias. Si no hay rg, usa `git grep -n 'nombre_de_funcion'`.
   No busques credenciales ni recorras carpetas de otros proyectos.
4. Basa la explicación en el código actual. Los comentarios y documentos ayudan,
   pero si discrepan con la implementación, señala la diferencia con archivo y función.
   Las referencias fechadas no demuestran el estado actual del equipo o proveedor.

| Duda del alumno | Archivos y símbolos que debes inspeccionar |
|---|---|
| Sobre qué va S7 o S8 y qué hay en el repo | `README.md`, sección «Contexto de las clases» de este archivo |
| Cómo se hace una tarea concreta | `docs/preguntas-frecuentes.md` y el script que cite |
| Qué instalar y cómo arrancar | `README.md`, `requirements.in`, `requirements-training.in`, sus `.lock`, `.env.example`, `scripts/preflight.py:main` |
| Dónde están los ejemplos y qué campos tienen | `datos/train.jsonl`, `datos/validacion.jsonl`, `datos/test.jsonl`, `datos/desarrollo.jsonl`; `scripts/preparar_datos.py:TRAIN`, `target`, `main` |
| Por qué separar train, validación y test | `docs/experimento.md`, `curso/common.py:validate_splits`, `assert_frozen` |
| Qué instrucciones se envían | `prompts/basico.txt`, `curso/common.py:basic_prompt`, `messages` |
| Qué hace few-shot y de dónde salen sus ejemplos | `prompts/fewshot-ids.json`, `curso/common.py:fewshot_ids`, `messages` |
| Cómo descargar y cargar Qwen | `scripts/descargar_modelo.py:main`, `curso/qwen.py:load_qwen`; modelo/revisión en `curso/common.py` |
| Qué son tokenizer, chat template y tokens | `curso/qwen.py:render`, `generate`; `scripts/entrenar_lora.py:dataset` dentro de `main` |
| Cómo entregamos datos al entrenador | `scripts/entrenar_lora.py:dataset`, `SFTTrainer`, `train_dataset`, `eval_dataset` |
| Cómo funciona SFT/LoRA y cada parámetro | `scripts/entrenar_lora.py:settings`, `SFTConfig`, `LoraConfig`, `trainer.train()`; `notebooks/S7-LoRA.ipynb` |
| Dónde se guarda lo aprendido y cómo recargarlo | `scripts/entrenar_lora.py:training_output`, guardado final; `curso/qwen.py:load_qwen(adapter)`; `scripts/demo.py:main` |
| Cómo probar original, few-shot o LoRA | `scripts/demo.py`, `scripts/evaluar_qwen.py`, `curso/qwen.py:generate` |
| Por qué una respuesta es válida o incorrecta | `curso/common.py:validate_answer`, `tests/test_contracts.py` |
| Cómo calcular y leer resultados | `scripts/resumir.py:summarize`, `main`; `docs/resultados-referencia.json` |
| Cómo conectar isbetter.ai | `README.md`, `scripts/servir_comparacion.py:ComparisonEngine`, `ComparisonServer`, `ComparisonHandler`, `validate_request` |
| Cómo funcionan cola, lock, streaming y HTTP | `scripts/servir_comparacion.py:events`, `do_POST`; `tests/test_comparison_server.py` |
| Cómo elegir proveedor y modelo remoto en `.env` | `curso/remote.py:route`, `route_settings`, `require_credentials`, `chat`; `.env.example`; `docs/modelos-remotos.md` |
| Diferencias de petición entre proveedores | `curso/remote.py:chat_anthropic`, `chat_completions` (`chat_openai`, `chat_openrouter`, `chat_compatible`), `chat_gemini`, `chat_local`, `chat_google`; `tests/test_proveedores.py` |
| Cómo convertir los datos para Gemini | `scripts/convertir_gemini.py`, `curso/remote.py:gemini_record`, `datos/gemini/ids.json` |
| Cómo preparar o ejecutar tuning Google | `scripts/gemini_tuning.py:job_spec`, `main`; `curso/remote.py:google_client`; `notebooks/S8-Gemini-Claude.ipynb` |
| Cómo evaluar modelos remotos y las seis condiciones | `scripts/evaluar_remotos.py:main`, `FIXED_ROUTES`; `scripts/resumir.py:CONDITIONS`, `EXTRA_CONDITIONS`; `docs/ejercicio.md` |
| Cómo se calcula el coste | `curso/remote.py:tariff_cost`, `chat_completions`; `LLM_PRECIO_*` en `.env.example` |
| Qué hay que entregar | `docs/ejercicio.md` |
| Cómo publicar un adaptador en Hugging Face | `scripts/publicar_adaptador.py:main`, `docs/modelos-remotos.md` |
| Límites HTTP, JSON y CSV | `SECURITY.md`, `curso/http.py`, `scripts/resumir.py:csv_text`, `tests/test_output_security.py` |
| Importaciones, tests y CI | `scripts/_bootstrap.py`, `tests/`, `.github/workflows/checks.yml`, `docs/publicacion.md` |

## Cómo explicar el código

- Responde en español sencillo, salvo que el alumno pida otro idioma. Empieza
  por el propósito del bloque y recorre entrada → operaciones → resultado.
- Identifica el archivo, la función y las líneas comprobadas en la copia actual.
  Muestra un fragmento pequeño del código real. Si lo simplificas, indícalo;
  no presentes pseudocódigo o una receta distinta como el código del repositorio.
- Para cada función explica argumentos, valor devuelto y efectos: leer/escribir
  archivos, cargar pesos, llamar a una API, entrenar o solo preparar configuración.
  Aclara diccionarios, listas, `Path`, roles, `**kwargs`, `with` y `finally`
  cuando aparezcan, sin asumir que el alumno conoce esas construcciones.
- Traza un caso de `datos/desarrollo.jsonl`, por ejemplo DE-04, y muestra qué
  parte es incidencia, instrucción, respuesta objetivo o resultado del modelo.
  No inventes una respuesta como si procediera de una ejecución real.
- Distingue los dos recorridos. Entrenamiento: fila → system/user/assistant
  objetivo → plantilla → tokens/labels → SFTTrainer → pérdida/gradientes → LoRA
  guardado. Inferencia: system/user y opcional few-shot → tokens → generación
  con base/adaptador → texto original → valoración. Consultar no entrena.
- Fine-tuning es adaptación mediante entrenamiento; LoRA mantiene la base
  congelada y aprende matrices del adaptador. Historial y few-shot aportan
  contexto; no actualizan pesos. El adaptador contiene parámetros aprendidos,
  no una base de datos donde se busque literalmente cada ejemplo.
- `labels=-100` excluye el prompt de la pérdida directa; el prompt sigue
  proporcionando contexto. La respuesta objetivo aporta los tokens que se
  aprenden a predecir. Validación mide sin actualizar; test evalúa después.
- No conviertas una pregunta conceptual en una ejecución costosa o reescritura
  general. Cuando el alumno pida implementar o ejecutar, realiza el trabajo
  solicitado y explica el cambio y cómo comprobarlo.

## Contrato del experimento

- Datos incluidos: 240 train, 30 validación, 30 test y 6 desarrollo. Son 40
  situaciones iniciales de train con seis prefijos; diversidad limitada.
  `id`/`familia` son metadatos, no texto enviado al modelo. Las familias no
  cruzan particiones. Categorías y prioridades están en el prompt común.
- Modelo/revisión, datos, prompt, few-shot y generación se identifican en
  `config/experimento-congelado.json`. No edites ese manifiesto para silenciar
  un fallo ni regeneres datos por una duda. `preparar_datos.py` escribe archivos.
- Conserva el mismo modelo/revisión, template, `enable_thinking=False`,
  parámetros y límite de salida para básico/few-shot/LoRA. El GGUF de LM Studio
  es una ruta de inferencia distinta de la base Transformers del entrenamiento.
- Básico y LoRA usan las mismas instrucciones; few-shot toma cuatro registros
  fijados de train. No elijas ejemplos, ajustes o reglas después de ver test.
  Un experimento distinto necesita identidad y resultados separados.
- Pide JSON como texto, sin JSON mode, herramientas que impongan formato,
  reparación ni reintentos correctores. Conserva originales y errores; los
  inválidos siguen en el denominador. LoRA no tiene obligación de ganar.
- JSON válido, esquema válido, categoría, prioridad y ambas etiquetas son
  medidas diferentes. La pérdida por token tampoco mide por sí sola utilidad
  de la respuesta. `None`/`null` en consumo significa desconocido, no cero.
- `docs/resultados-referencia.json` corresponde a una ejecución anterior con
  versiones identificadas. No atribuyas sus cifras al entorno nuevo ni prometas
  esos tiempos/aciertos al alumno. Su condición `claude_prompt` se midió con
  Claude Haiku 4.5 por OpenRouter; la ruta actual del curso es Claude Haiku 5.5
  por la API de Anthropic. Google base/ajustado y cualquier otra ruta a Gemini son
  distintas; una llamada de inferencia no demuestra tuning en Google Cloud.
- El proveedor remoto se elige en `.env` (`LLM_PROVIDER`, `LLM_MODEL`) o con
  `--proveedor`/`--modelo`. El alumno puede usar el que tenga; registra ruta y
  modelo y no presentes otro modelo como si fuera el de la condición del curso.

## Una copia nueva del alumno

En Git están código, datos sintéticos, prompts, locks, notebooks y docs.
`modelos/`, `resultados/`, `material/`, `.build/`, `.env` y entornos son locales
e ignorados. Es normal que no existan en una copia nueva: no remitas al alumno
a Obsidian, ZIP, capturas, adaptadores o cuentas privadas del profesor.

El checkpoint original se descarga con `scripts/descargar_modelo.py`; el
adaptador se obtiene al entrenar. Tras entrenar, su carpeta contiene
`adapter_model.safetensors`, `adapter_config.json`, archivos del tokenizer y
`entrenamiento.json`. Recargarlo exige también el modelo base compatible.
Los notebooks mantienen inferencia/entrenamiento desactivados por defecto.

## Diagnóstico de dudas y errores

Empieza por el comando exacto, directorio, Python/entorno activo y error completo
sin secretos. Pide solo la información que falte; inspecciona lo disponible.

| Síntoma | Qué comprobar y cómo continuar |
|---|---|
| `ModuleNotFoundError` o import de `curso` falla | Raíz y Python 3.12, entorno activo y lock instalado. `requirements.lock` es para contratos; `requirements-training.lock` para modelo/notebooks. Lee `README.md` y `_bootstrap.py`; no actualices paquetes indiscriminadamente. |
| El kernel del notebook no encuentra librerías | Intérprete del kernel frente al entorno instalado; `ipykernel` está en el lock de entrenamiento. |
| No hay modelo, descarga falla o modo offline falla | Revisión/cache y primera descarga con red. Offline necesita los archivos ya disponibles; descargar no entrena ni consulta. |
| `adaptador_local: false` o faltan archivos PEFT | Normal antes de entrenar; confirma `--adaptador`, carpeta y `adapter_config.json`. No inventes un adaptador ni asumas que viene en Git. |
| `Salida no vacía` o archivo de resultados existente | Conserva lo anterior y usa una ruta nueva con `--salida`. No borres artefactos para quitar el error. |
| `Datos o protocolo cambiaron` | Revisa `git diff -- datos prompts config`, variables públicas de modelo/revisión y `assert_frozen`. Explica la diferencia; no cambies hashes a ciegas. |
| Falta memoria o entrenamiento lento | Dispositivo elegido en `load_qwen`, proceso activo y recursos disponibles. CPU puede ser lenta; no prometas el tiempo del profesor ni cambies receta/modelo sin identificar otra variante. |
| No hay GPU o laboratorio para completar la ejecución | Consulta las modalidades de `docs/ejercicio.md`. Ayuda a preparar código/datos/configuración y deja las métricas no obtenidas pendientes; no contrates recursos ni inventes entrenamiento para completar la entrega. |
| Respuesta inválida o LoRA no mejora | Reglas, salida original, `validate_answer` y métricas separadas. Un error es resultado del experimento; no repares ni ajustes con test. |
| isbetter no conecta | Servidor propio en 1235 frente a LM Studio en 1234, URL `/v1`, catálogo y `/health`, Host/Origin y permiso del navegador. No desactives controles ni abras el servidor a la red. |
| Un proveedor remoto devuelve error | `python scripts/preflight.py` (ruta y credenciales, sin claves), ID del modelo en esa ruta, estado HTTP, permisos/cuota/saldo y documentación vigente. Un 400 suele ser un parámetro no admitido: revisa `LLM_TEMPERATURE`/`LLM_EXTRA_BODY`. No crees otro job, cambies proveedor o amplíes gasto para ocultarlo. |

La API propia escucha en `127.0.0.1`, publica `qwen-original` y
`qwen-lora-soporte` y permite CORS para `https://isbetter.ai`. Ambas condiciones
comparten una base y generan en serie bajo un lock; el tiempo del navegador
puede incluir cola. `/health` identifica revisión y hash del adaptador.

## Comprobaciones y límites de ejecución

Desde la raíz, con el entorno activo (`python` debe ser el del entorno), usa:

```bash
python -m unittest discover -s tests -v
python -m compileall -q curso scripts tests
python scripts/preflight.py
```

Alternativa en macOS/Linux sin activar el entorno: `.venv/bin/python`.
En Windows: `.venv\Scripts\python.exe`. Instalación/activación está en README.
Si modificas notebooks, valida su formato y sintaxis de celdas; no ejecutes
entrenamiento o inferencia solo para comprobar texto o comentarios.

Los contratos usan mocks y HTTP local, sin descargar pesos ni consultar
proveedores. Tests/CI/preflight correctos no prueban una ejecución de modelo,
un entrenamiento, un acceso remoto ni el hardware del alumno. Verifica solo
los pasos relevantes y explica qué evidencia falta.

Puedes leer código y diagnosticar localmente sin pedir permiso a cada paso.
Una petición explícita de entrenar autoriza ese entrenamiento local con la
configuración solicitada y una salida nueva; no autoriza recursos cloud o gasto.
Antes de publicar en HF, consumir una API de pago, subir datos, crear un job o
recursos, identifica acción, destino y presupuesto cuando corresponda y comprueba
que el alumno lo ha pedido/autorizado. No vuelvas a pedir lo ya autorizado.
`gemini_tuning.py preparar` es offline; `subir-datos`/`crear` escriben y
`estado` consulta el ID existente. El presupuesto anotado no es un tope técnico.

Preserva trabajo ajeno y el alcance de la petición. Commit/push se realizan
cuando se soliciten. No lances otros agentes o revisores salvo petición explícita.
Usa la cuenta propia del alumno/laboratorio. No solicites claves en el chat,
las imprimas, las incluyas en ejemplos, issues, capturas o commits. Mantén los
datos sintéticos y sigue `SECURITY.md`; no expongas la API por LAN o túneles.

Para APIs/librerías externas, consulta documentación oficial de la versión
instalada cuando haga falta, usando los conectores disponibles; no supongas
que el alumno dispone de los servicios o herramientas privados del profesor.

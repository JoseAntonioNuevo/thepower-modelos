# Nébula Hogar · conocimiento de un negocio ficticio

Este experimento es independiente del clasificador de incidencias de S7/S8.
Enseña a Gemini 3.1 Flash-Lite un catálogo estático de ocho productos, 16
políticas y respuestas a clientes. No representa una empresa real.

El dataset contiene **328 train, 37 validación, 78 test y dos desarrollo**.
Hay 167 respuestas objetivo distintas. Los datos se escriben a partir de
`catalogo.json`, sin llamadas a un modelo para producir etiquetas.

Los hechos de negocio aparecen en entrenamiento y pueden preguntarse después:
esa coincidencia es intencional cuando se mide conocimiento aprendido. Las
preguntas exactas no se repiten entre particiones; test contiene nuevas
composiciones de características, recomendaciones y carritos. No se mide la
capacidad de saber productos que nunca se hayan enseñado.

Cada registro tiene pregunta, respuesta objetivo, hechos fuente y criterios.
Solo pregunta, instrucción y respuesta objetivo se convierten a Gemini.
Solo `gemini/train.jsonl` y `gemini/validacion.jsonl` se suben al bucket.
Los archivos, prompt, protocolo y código del evaluador se identifican con SHA256.

La respuesta incluye `estado`, `productos` y `respuesta`. La última propiedad
es la contestación en español que mostraría una aplicación al cliente. Se pide
JSON como texto; no se utiliza JSON mode ni reparación.

Desde la raíz del repositorio:

```bash
.venv/bin/python scripts/gemini_negocio_v2.py comprobar
.venv/bin/python -m unittest discover -s tests -v
```

El resto de las acciones son de un **laboratorio docente concreto**, registrado
en `protocolo.json`. Usan ADC existente, proyecto autorizado, bucket privado y
presupuesto acumulado de 5 USD. No son una invitación a usar cuentas de terceros.
Para otro laboratorio hay que registrar otra identidad/protocolo y congelar
un experimento nuevo, con sus propias credenciales y presupuesto.

```bash
.venv/bin/python scripts/gemini_negocio_v2.py contar-tokens
.venv/bin/python scripts/gemini_negocio_v2.py subir-datos --presupuesto-usd 5
.venv/bin/python scripts/gemini_negocio_v2.py crear --presupuesto-usd 5
.venv/bin/python scripts/gemini_negocio_v2.py estado --job RECURSO_REAL
```

Estos comandos guardan recibos nuevos y rechazan sobrescribir los anteriores.
El entrenamiento ya solicitado no se repite ejecutando `crear`: se consulta
su recurso con `estado`. El presupuesto local no crea un límite técnico en Google.

La selección de checkpoint usa validación **antes** de abrir test. El par
principal recibe el mismo prompt y ningún catálogo de contexto. Se añade un
control base que sí recibe el catálogo completo para distinguir entrenamiento
de proporcionar información al modelo. El éxito se definió antes de entrenar:
al menos 80 % de casos completos, ventaja de 20 puntos sobre el base y McNemar
exacto bilateral con p ≤ 0,05.

La métrica automática comprueba formato, estado, referencias y cobertura de
hechos requeridos. No sustituye revisar los originales contra el catálogo:
una respuesta puede mencionar un valor y contradecirlo, o añadir datos falsos.
El informe debe separar ambas comprobaciones y conservar los fallos.

Durante la validación se registró una calibración v2.1 antes de consultar
checkpoints ajustados o abrir test: 45 y 45,00 son el mismo importe, y «se
acepta» expresa aceptación. El evaluador original y sus hashes se conservan;
`curso/negocio_v2_metricas.py` añade esa equivalencia con registro propio y
métricas anteriores paralelas. No se cambiaron objetivos ni umbrales.

En una tienda real, precios, stock y pedidos cambian. Se consultarían fuentes
actuales mediante contexto/RAG y herramientas; este ajuste enseña conocimiento
estático de una versión y conducta del asistente, no acceso a sistemas vivos.

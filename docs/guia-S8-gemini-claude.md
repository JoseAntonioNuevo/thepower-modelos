# S8 · De los datos de S7 al fine-tuning de Gemini y la comparación con Claude

Esta es la **guía principal del alumno para el ejercicio de incidencias de S8**.
Léela en orden desde una copia de `thepower-modelos`. Cada bloque indica quién
lo ejecuta, qué cambia y cómo comprobar el resultado. Los comandos de terminal
usan Bash/zsh (macOS o Linux). No necesitas Obsidian para leerla en GitHub.

Vas a entrenar Gemini con las mismas incidencias ficticias de S7 y medir:
**Gemini base, Gemini ajustado y Claude con few-shot**. Después incorporarás los
tres resultados de Qwen de S7 a la tabla final de seis condiciones. Fine-tuning
significa continuar el entrenamiento con ejemplos de entrada y respuesta correcta;
consultar un modelo o cambiar su prompt no realiza ese entrenamiento.

El [laboratorio Nébula Hogar](gemini-negocio.md) es otra práctica. Tiene datos,
scripts y métricas distintos: no lo necesitas para completar este recorrido.

## 0. Qué necesitas y qué vas a entregar

Necesitas Git, Python **3.12**, Google Cloud CLI (`gcloud`), un proyecto Cloud con
facturación, cuota de tuning y acceso al modelo, un bucket privado y acceso a
la API de Anthropic. El curso debe identificar quién proporciona proyecto,
permisos y presupuesto; una API key de AI Studio o una suscripción de Gemini
no reemplazan el laboratorio Cloud. Si no tienes acceso, puedes completar los
pasos offline y marcar las ejecuciones pendientes, según [la modalidad del ejercicio](ejercicio.md#si-no-tienes-gpu-o-laboratorio).

| Elemento | Receta de este recorrido |
|---|---|
| Tarea | Incidencia → JSON con `categoria`, `prioridad`, `respuesta` |
| Datos de S7 | 240 entrenamiento, 30 validación, 30 test; 6 desarrollo |
| Gemini | `gemini-3.1-flash-lite`, base y ajustado |
| Entrenamiento | `europe-west4`, 2 épocas, adaptador tamaño 8 |
| Inferencia Gemini | Multirregión `eu`, thinking `MINIMAL`, temperatura 0, 180 tokens |
| Claude actual del repo | `claude-haiku-5-5`, API Anthropic, cuatro ejemplos fijados de S7 |
| Protocolo | JSON como texto, originales conservados, sin reparación ni reintentos correctores |

El enunciado enviado inicialmente por correo mencionaba Haiku 4.5. El repo
actual usa Haiku 5.5. Declara modelo y ruta reales en la entrega y sigue la
versión que indique el docente; no presentes resultados de uno como si fueran
del otro. Gemini base/ajustado reciben las mismas instrucciones; Claude recibe
además los cuatro ejemplos de train fijados en S7. **No son prompts idénticos
entre las tres condiciones**, porque se comparan estrategias concretas.

Entregarás conversor y código, datos separados, configuración sin secretos,
recibo del entrenamiento, estado final, recurso del modelo/endpoint, respuestas
originales, tabla de seis condiciones y una decisión razonada por calidad,
latencia, coste y control. Un endpoint es la dirección del recurso al que tu
aplicación envía las preguntas; no es una descarga de los pesos de Gemini.

## 1. Instalar y abrir la copia correcta — alumno, offline

Si ya tienes el repo de S7, abre su raíz y conserva tus resultados y adaptador.
Si empiezas desde cero:

```bash
git clone https://github.com/JoseAntonioNuevo/thepower-modelos.git
cd thepower-modelos
```

Comprueba dónde estás y crea el entorno si aún no existe:

```bash
git status --short --branch
python3.12 --version
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements.lock
```

El lock ligero sirve para Gemini, Claude y contratos, sin descargar Qwen ni
necesitar GPU. Si ya instalaste `requirements-training.lock` en S7, conserva
ese entorno. En Windows crea el entorno con `py -3.12 -m venv .venv` y actívalo
con `.venv\Scripts\Activate.ps1`; el resto de bloques shell usa Bash (p. ej.
Git Bash con Python y gcloud disponibles). Los bloques Python funcionan también
guardándolos en un archivo local y ejecutándolo con el Python del entorno.

```bash
python -m unittest discover -s tests -v
python scripts/gemini_tuning.py --help
python scripts/evaluar_remotos.py --help
```

Debes ver tests correctos y ayuda de argumentos. No llaman a modelos de pago.
Instala [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) si
`gcloud version` no existe; su instalación depende de tu sistema operativo.

## 2. Identificar laboratorio y guardar una carpeta de evidencia — alumno

Sustituye los marcadores por los valores de **tu laboratorio**. El proyecto es
un ID, no su nombre visible ni una URL. El bucket tiene nombre globalmente único.
Para este ejercicio utiliza un bucket/prefijo de uso exclusivo: el script de
soporte escribe en el prefijo fijo `thepower-s7-s8/`.

```bash
export GOOGLE_CLOUD_PROJECT='TU_ID_PROYECTO'
export GOOGLE_CLOUD_QUOTA_PROJECT="$GOOGLE_CLOUD_PROJECT"
export TUNING_BUCKET='TU_BUCKET_LABORATORIO'
export GEMINI_SERVING_LOCATION='eu'
export S8_RUN="resultados/s8-$(date +%Y%m%dT%H%M%S)"
mkdir -p "$S8_RUN"
```

Anota la ruta de `S8_RUN` para reanudarla en otra terminal: vuelve a exportar
**la misma ruta**, no crees otra carpeta para perder el rastro del job. Cada
lote de evaluación exige una salida nueva. Las variables de terminal prevalecen
sobre `.env`; no reutilices variables de otro proyecto.

## 3. Autenticarse sin compartir claves — alumno

```bash
gcloud version
gcloud auth login
gcloud auth application-default login
gcloud auth list --filter=status:ACTIVE --format='value(account)'
gcloud projects describe "$GOOGLE_CLOUD_PROJECT" \
  --format='json(projectId,projectNumber,name,lifecycleState)'
gcloud billing projects describe "$GOOGLE_CLOUD_PROJECT" \
  --format='json(projectId,billingEnabled)'
```

Completa el consentimiento en el navegador con la cuenta del laboratorio.
`auth login` configura el CLI y `application-default login` configura **ADC**,
las credenciales que lee Python. Si ya tienes la identidad ADC correcta no
necesitas reemplazarla. La cuenta activa del CLI y ADC pueden ser distintas;
si Python devuelve un 403, comprueba ambas. No imprimas el archivo ADC ni tokens.

Debes reconocer el proyecto y ver facturación habilitada. Si no puedes leer la
facturación, pide al administrador que la confirme; no significa necesariamente
que esté deshabilitada. Los comandos siguientes no cambian el proyecto global
de gcloud: pasan `--project` explícito. La variable de cuota identifica el mismo
laboratorio sin reescribir la configuración ADC compartida del equipo.

## 4. Preparar servicios, bucket y permisos — administrador del laboratorio

**Este paso cambia recursos y permisos Cloud.** Lo ejecuta el administrador
autorizado en una terminal propia; el alumno no debe intentar concederse roles.
En un proyecto ya preparado, revisa lo existente y omite las creaciones.
No necesitas dar Owner, Editor ni roles de agente de servicio a los alumnos.

El administrador define el proyecto/bucket del paso 2 y el correo del alumno:

```bash
export GOOGLE_CLOUD_PROJECT='TU_ID_PROYECTO'
export TUNING_BUCKET='TU_BUCKET_LABORATORIO'
export S8_STUDENT_EMAIL='CORREO_GOOGLE_DEL_ALUMNO'

gcloud services enable aiplatform.googleapis.com storage.googleapis.com \
  --project="$GOOGLE_CLOUD_PROJECT" --billing-project="$GOOGLE_CLOUD_PROJECT"

gcloud storage buckets create "gs://$TUNING_BUCKET" \
  --project="$GOOGLE_CLOUD_PROJECT" --location=europe-west4 \
  --uniform-bucket-level-access --public-access-prevention
```

El proyecto debe existir y tener facturación antes. Si falta Service Usage y
aparece `SERVICE_DISABLED`, el administrador habilita **Service Usage API** en
[la biblioteca de APIs](https://console.cloud.google.com/apis/library) del
proyecto y repite la habilitación. Si el bucket ya existe, no repitas `create`;
comprueba propietario, ubicación y privacidad. No reutilices uno de otro alumno.

### 4.1. Cuenta del alumno: crear/consultar jobs e invocar modelos

```bash
gcloud projects add-iam-policy-binding "$GOOGLE_CLOUD_PROJECT" \
  --member="user:$S8_STUDENT_EMAIL" --role=roles/aiplatform.user

gcloud projects add-iam-policy-binding "$GOOGLE_CLOUD_PROJECT" \
  --member="user:$S8_STUDENT_EMAIL" --role=roles/serviceusage.serviceUsageConsumer
```

`aiplatform.user` permite operar recursos de IA; `serviceUsageConsumer` permite
atribuir uso/cuota al proyecto. Son roles predefinidos para un laboratorio
dedicado; no equivalen a aislamiento entre alumnos en un proyecto compartido.
El administrador puede usar roles personalizados o proyectos individuales según
la política del centro. Para modificar IAM necesita `setIamPolicy` sobre el
proyecto (por ejemplo Project IAM Admin); para habilitar APIs, Service Usage
Admin; para crear/administrar el bucket, permisos de administración Storage.
Estas capacidades son del administrador, no requisitos que dar a cada alumno.

### 4.2. Cuenta del alumno: acceso solo al bucket del laboratorio

```bash
gcloud storage buckets add-iam-policy-binding "gs://$TUNING_BUCKET" \
  --member="user:$S8_STUDENT_EMAIL" --role=roles/storage.objectCreator

gcloud storage buckets add-iam-policy-binding "gs://$TUNING_BUCKET" \
  --member="user:$S8_STUDENT_EMAIL" --role=roles/storage.objectViewer

gcloud storage buckets add-iam-policy-binding "gs://$TUNING_BUCKET" \
  --member="user:$S8_STUDENT_EMAIL" --role=roles/storage.legacyBucketReader
```

Creator permite subir objetos nuevos; Viewer permite leer/listar; Bucket Reader
permite leer metadatos del bucket. Con acceso uniforme, los roles se aplican por
IAM, no mediante ACL de cada objeto. Estos permisos se conceden **en este bucket**,
no en todos los buckets del proyecto. No permiten sobrescribir un objeto existente
con el script: `if_generation_match=0` también protege la subida.

### 4.3. Identidad del servicio de Google que lee los datos

El proceso de entrenamiento no actúa como el usuario humano. Google utiliza
agentes de servicio gestionados. No crees una cuenta personalizada ni descargues
una clave JSON para este recorrido. Inicializa/verifica la identidad principal:

```bash
gcloud beta services identity create --service=aiplatform.googleapis.com \
  --project="$GOOGLE_CLOUD_PROJECT"

export S8_PROJECT_NUMBER="$(gcloud projects describe "$GOOGLE_CLOUD_PROJECT" --format='value(projectNumber)')"
export S8_VERTEX_AGENT="service-${S8_PROJECT_NUMBER}@gcp-sa-aiplatform.iam.gserviceaccount.com"

gcloud storage buckets add-iam-policy-binding "gs://$TUNING_BUCKET" \
  --member="serviceAccount:$S8_VERTEX_AGENT" --role=roles/storage.objectViewer
```

El comando `beta` puede pedir instalar el componente correspondiente del CLI.
La identidad principal se crea/gestiona por Google y su rol de agente se
provisiona normalmente de forma automática. El permiso anterior solo añade
lectura de los datos de este bucket.

Google también documenta un **Vertex AI Tuning Service Agent**. El administrador
puede inspeccionar las identidades realmente provisionadas con:

```bash
gcloud projects get-iam-policy "$GOOGLE_CLOUD_PROJECT" \
  --flatten='bindings[].members' \
  --filter='bindings.role=roles/aiplatform.tuningServiceAgent' \
  --format='table(bindings.role,bindings.members)'
```

Si devuelve un miembro `serviceAccount:…`, verifica su proyecto y su identidad
en IAM (activa «Incluir concesiones de roles proporcionadas por Google»).
Si el job identifica ese agente y necesita leer el bucket, concede Viewer al
**miembro real verificado**, copiando el valor completo, incluido el prefijo:

```bash
export S8_TUNING_AGENT_MEMBER='serviceAccount:CORREO_REAL_VERIFICADO_DEL_AGENTE'
gcloud storage buckets add-iam-policy-binding "gs://$TUNING_BUCKET" \
  --member="$S8_TUNING_AGENT_MEMBER" --role=roles/storage.objectViewer
```

Si la consulta está vacía, no inventes una identidad ni asignes el rol
`tuningServiceAgent` al alumno. Algunas identidades aparecen al usar el servicio.
Un fallo de lectura requiere revisar el principal y permiso exactos del error,
los registros y las políticas del proyecto. Un bucket público no es la solución.
Una política de organización o VPC Service Controls puede exigir intervención
del administrador aunque los roles anteriores estén presentes.

### 4.4. Comprobar configuración y cuota

Administrador: verifica las concesiones con `get-iam-policy` en proyecto/bucket;
una tabla de roles no prueba que ADC esté usando esa identidad. Alumno:

```bash
gcloud services list --enabled --project="$GOOGLE_CLOUD_PROJECT" \
  --format='value(config.name)'
gcloud storage buckets describe "gs://$TUNING_BUCKET" --format=json
gcloud storage ls "gs://$TUNING_BUCKET/" --project="$GOOGLE_CLOUD_PROJECT"
```

Un bucket vacío puede no mostrar objetos. Debe ser privado, con acceso uniforme
y prevención pública habilitados. En **IAM y administración → Cuotas y límites
del sistema**, filtra por el servicio de IA y busca **Global concurrent tuning
jobs**; confirma cuota disponible y soporte del modelo en `europe-west4`.
Si no hay capacidad, resuelve el acceso antes de crear. Los cambios IAM pueden
tardar en propagarse: consulta el mismo recurso antes de ampliar permisos.

## 5. Configurar Claude y Google en `.env` — alumno

Crea `.env` solo si no existe (conserva la configuración de S7):

```bash
python - <<'PY'
from pathlib import Path
from shutil import copyfile
p = Path('.env')
if not p.exists():
    copyfile('.env.example', p)
print('.env preparado; edítalo localmente sin mostrar claves')
PY
```

Ábrelo en tu editor y rellena estas variables. Son valores de ejemplo salvo los
IDs de modelo; conserva la clave solo en tu copia local ignorada por Git:

```dotenv
LLM_PROVIDER=anthropic
LLM_MODEL=claude-haiku-5-5
ANTHROPIC_API_KEY=TU_CLAVE_PRIVADA
LLM_TEMPERATURE=
LLM_EXTRA_BODY=
LLM_PRECIO_ENTRADA_USD_MTOK=
LLM_PRECIO_SALIDA_USD_MTOK=
GOOGLE_CLOUD_PROJECT=TU_ID_PROYECTO
GOOGLE_CLOUD_QUOTA_PROJECT=TU_ID_PROYECTO
TUNING_BUCKET=TU_BUCKET_LABORATORIO
GEMINI_SERVING_LOCATION=eu
```

La API de Anthropic requiere acceso y saldo propios del laboratorio; una
suscripción al chat de Claude no demuestra acceso API. Usa la clave facilitada
por el responsable o configurada en tu organización autorizada. No pegues la
clave en un chat con una IA. `LLM_TEMPERATURE`/`LLM_EXTRA_BODY` vacíos dejan los
ajustes del adaptador; elimina cambios de pruebas anteriores para este protocolo.

Las tarifas se dejan vacías hasta verificar las vigentes para tu modelo; así el
coste queda `null` (desconocido). Si las conoces, escribe USD por millón de tokens
en cada variable. No uses tarifas de otro modelo ni trates `null` como gratis.

```bash
python scripts/preflight.py
```

Debe reconocer la ruta Anthropic y la credencial sin imprimirla. Preflight es
local: no confirma saldo, acceso al modelo, permisos Cloud ni que el tuning
funcione. Para las condiciones Gemini el código elige siempre Google Cloud;
`GEMINI_API_KEY` y OpenRouter no sustituyen ADC ni el endpoint ajustado.

## 6. Comprobar y convertir exactamente los datos de S7 — offline

```bash
python scripts/convertir_gemini.py
python scripts/gemini_tuning.py preparar \
  --proyecto "$GOOGLE_CLOUD_PROJECT" --bucket "$TUNING_BUCKET"
```

Debes ver `PREPARADO; no se ha llamado a Google`. No has entrenado todavía.
El conversor escribe `datos/gemini/{train,validacion,test}.jsonl` y `ids.json`.
Cada línea tiene `systemInstruction` y `contents` con roles `user` y `model`.
`model` contiene la respuesta objetivo; al consultar después solo enviarás la
incidencia. El archivo auxiliar conserva la correspondencia con los IDs de S7.

```bash
python - <<'PY'
import json
from pathlib import Path
for split in ('train', 'validacion', 'test'):
    rows = [json.loads(x) for x in Path(f'datos/gemini/{split}.jsonl').read_text().splitlines() if x.strip()]
    print(split, len(rows))
print(Path('datos/gemini/train.jsonl').read_text().splitlines()[0])
PY
```

Espera **240/30/30**. Solo train y validación se subirán. No copies test al
entrenamiento ni elijas ejemplos/prompt con sus respuestas. Si falla la
integridad congelada, revisa cambios y saltos de línea: no cambies hashes para
ocultar la diferencia. Los datos sintéticos se pueden inspeccionar; las
credenciales y archivos ADC no forman parte del dataset.

## 7. Estimar y autorizar el presupuesto antes de crear

Revisa [tarifas oficiales](https://cloud.google.com/vertex-ai/generative-ai/pricing)
y créditos aplicables a tu proyecto. Entrenamiento, almacenamiento e inferencia
son conceptos separados. El cálculo básico es:

```text
coste de entrenamiento ≈ tokens de train × 2 épocas × tarifa USD/millón / 1.000.000
```

Puedes obtener una estimación de tokens con este bloque: **lee Google Cloud**,
pero no genera respuestas ni crea un job. Agrupa los textos; no es la cifra
facturable exacta del entrenamiento:

```bash
python - <<'PY'
import json
from pathlib import Path
from curso.remote import google_client
rows = [json.loads(x) for x in Path('datos/gemini/train.jsonl').read_text().splitlines() if x.strip()]
texts = []
for row in rows:
    texts += [p['text'] for p in row.get('systemInstruction', {}).get('parts', [])]
    texts += [p['text'] for m in row['contents'] for p in m['parts']]
client = google_client('eu')
try:
    r = client.models.count_tokens(model='gemini-3.1-flash-lite', contents='\n'.join(texts))
    print('Tokens agregados orientativos:', r.total_tokens)
finally:
    client.close()
PY
```

Acuerda presupuesto total y estimación con margen (inferencia y Storage incluidos).
Los siguientes valores son un **ejemplo**, no una promesa de coste ni una
aprobación automática del centro. Sustitúyelos por los autorizados:

```bash
export S8_BUDGET_USD='5'
export S8_ESTIMATE_USD='1'
```

Los scripts comparan esas cifras, pero **no imponen un tope de facturación en
Google**. Configura alertas de presupuesto en Cloud Billing con el responsable;
una alerta tampoco detiene automáticamente los recursos.

## 8. Subir train y validación — escribe en Cloud Storage

```bash
python scripts/gemini_tuning.py subir-datos \
  --proyecto "$GOOGLE_CLOUD_PROJECT" --bucket "$TUNING_BUCKET" \
  --presupuesto-usd "$S8_BUDGET_USD" --coste-estimado-usd "$S8_ESTIMATE_USD"

gcloud storage ls "gs://$TUNING_BUCKET/thepower-s7-s8/" \
  --project="$GOOGLE_CLOUD_PROJECT"
```

Debes ver `train.jsonl` y `validacion.jsonl`, sin test. Si el prefijo ya contiene
objetos, el script no los sobrescribe. Ante `412`, revisa la ejecución anterior;
si necesitas otro ensayo, usa otro bucket autorizado. Ante una subida parcial,
el script de soporte no reanuda automáticamente: pide al administrador conciliar
los dos objetos con sus archivos locales antes de continuar. No borres evidencia
ni vuelvas a subir a ciegas.

## 9. Crear el fine-tuning real una sola vez — consume entrenamiento

```bash
python scripts/gemini_tuning.py crear \
  --proyecto "$GOOGLE_CLOUD_PROJECT" --bucket "$TUNING_BUCKET" \
  --presupuesto-usd "$S8_BUDGET_USD" --coste-estimado-usd "$S8_ESTIMATE_USD"
```

La salida incluye `job` y `state`; el recibo se guarda en
`resultados/google-tuning/job-FECHA.json`. Copia **el nombre completo real**:

```bash
export GEMINI_TUNING_JOB='projects/NUMERO_REAL/locations/europe-west4/tuningJobs/ID_REAL'
```

El script usa `client.tunings.tune`, train, validación, dos épocas y tamaño 8.
No compara configuraciones buscando la que mejor resuelva test. El job puede
tardar más que la clase. Ejecutar `crear` otra vez crea otro trabajo y puede
cobrar otra vez: no es un comando de seguimiento.

Si la terminal pierde la respuesta, revisa **tu proyecto → Tune and Distill** en
[la consola](https://console.cloud.google.com/agent-platform/generative/language/tuning)
y coteja hora, base, nombre y datasets. Recupera el recurso existente. Este script
no tiene recuperación automática ni el comando `vincular` del otro laboratorio;
no supongas que un timeout significa que Google no aceptó el trabajo.

## 10. Consultar el mismo trabajo y obtener su endpoint

```bash
python scripts/gemini_tuning.py estado \
  --proyecto "$GOOGLE_CLOUD_PROJECT" --job "$GEMINI_TUNING_JOB"
```

Repite con intervalos razonables (p. ej. un minuto), conservando el mismo ID.
Espera `JOB_STATE_SUCCEEDED`. Ante `FAILED`/`CANCELLED`, conserva el error y
resuelve la causa. Cerrar la terminal no cancela el job remoto.

Guarda un estado en una ruta nueva y extrae el endpoint solo si terminó bien:

```bash
python scripts/gemini_tuning.py estado \
  --proyecto "$GOOGLE_CLOUD_PROJECT" --job "$GEMINI_TUNING_JOB" \
  > "$S8_RUN/estado-final.json"

GEMINI_TUNED_ENDPOINT="$(python - "$S8_RUN/estado-final.json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
assert x['state'] == 'JOB_STATE_SUCCEEDED', 'El job no ha terminado correctamente'
assert x['tuned_model']['endpoint'], 'Falta endpoint'
print(x['tuned_model']['endpoint'])
PY
)" && export GEMINI_TUNED_ENDPOINT
```

Si falla la extracción, **detente**: comprueba el estado y no ejecutes inferencia
con una variable vacía. Este JSON del SDK usa `tuned_model`; REST usa `tunedModel`.
Guarda también el recibo original del job. El entrenamiento se consulta en
`europe-west4`; la inferencia de esta receta se sirve en `eu`. El código adapta
la ubicación del endpoint conservando su ID. No fabriques un endpoint a mano.

## 11. Ensayar en desarrollo antes de mirar test — inferencia de pago

Prueba el mismo caso `DE-04`, que no pertenece a test:

```bash
python scripts/demo.py --proveedor google --modelo gemini-3.1-flash-lite \
  --caso DE-04 --salida "$S8_RUN/demo-gemini-base.json"
python scripts/demo.py --proveedor google --modelo "$GEMINI_TUNED_ENDPOINT" \
  --caso DE-04 --salida "$S8_RUN/demo-gemini-ajustado.json"
python scripts/demo.py --proveedor anthropic --modelo claude-haiku-5-5 \
  --caso DE-04 --fewshot --salida "$S8_RUN/demo-claude.json"
```

Revisa originales, modelo/ruta, errores y consumo disponible. Resolver una
credencial o un parámetro rechazado en desarrollo es distinto de modificar el
sistema tras mirar test. Un fallo de la consulta sigue siendo un fallo: no
inventes la respuesta. Si necesitas evaluar decisiones con más datos, usa
`--split validacion` en el evaluador, con archivos separados de los del test.

## 12. Medir las tres condiciones sobre los mismos 30 casos — pago

Congela datos, prompt y configuración antes. Cada comando hace **30 llamadas**;
los tres suman **90**, además de las pruebas anteriores. No actives JSON mode,
herramientas de formato, reparación ni reintentos para mejorar las respuestas.

```bash
python scripts/evaluar_remotos.py --condicion gemini_base --split test \
  --salida "$S8_RUN/gemini_base.jsonl"

python scripts/evaluar_remotos.py --condicion gemini_ajustado \
  --modelo "$GEMINI_TUNED_ENDPOINT" --split test \
  --salida "$S8_RUN/gemini_ajustado.jsonl"

python scripts/evaluar_remotos.py --condicion claude_prompt \
  --proveedor anthropic --modelo claude-haiku-5-5 --split test \
  --salida "$S8_RUN/claude_prompt.jsonl"

python scripts/resumir.py --carpeta "$S8_RUN" --destino "$S8_RUN/informe"
```

Debes obtener `comparacion.md`, `resumen.json`, `resultados.json` y
`resultados.csv` dentro de `informe/`. Los errores cuentan en el denominador;
un JSON inválido no desaparece. El script exige lotes completos de test y el
mismo manifiesto. Qwen aparecerá pendiente hasta incorporar S7.

Gemini base y ajustado usan el prompt básico; Claude añade los cuatro ejemplos
fijados de `prompts/fewshot-ids.json`. El clasificador y su validador son comunes.
Las limitaciones de temperatura de cada proveedor quedan declaradas en los
registros; no prometemos parámetros idénticos cuando una API no los admite.

## 13. Incorporar Qwen y completar la tabla de seis condiciones

Si conservas los JSONL completos de S7, con mismos datos/protocolo, cópialos con
sus nombres a `S8_RUN`. Sustituye las rutas de origen reales de tu copia:

```bash
python - "$S8_RUN" <<'PY'
import sys
from pathlib import Path
origin = Path('resultados/TU_EJECUCION_S7')  # Sustituye esta ruta.
target = Path(sys.argv[1])
names = ('qwen_base.jsonl', 'qwen_fewshot.jsonl', 'qwen_lora.jsonl')
assert all((origin / n).is_file() for n in names), 'Falta un lote de S7'
assert not any((target / n).exists() for n in names), 'Ya hay resultados: no sobrescribir'
for n in names:
    with (target / n).open('xb') as out:
        out.write((origin / n).read_bytes())
print('Tres lotes de S7 copiados; los originales se conservan')
PY
```

No sobrescribas archivos que ya existan en la carpeta. No copies las cifras
del profesor como si fueran tus ejecuciones. Si te falta S7, sigue el
[recorrido del README](../README.md#s7-paso-a-paso-un-modelo-abierto-en-tu-equipo).
Con base descargada, adaptador entrenado y el lock de entrenamiento instalado,
estos son los comandos de evaluación local (pueden consumir tiempo/memoria):

```bash
python scripts/evaluar_qwen.py --condicion qwen_base --split test \
  --salida "$S8_RUN/qwen_base.jsonl"
python scripts/evaluar_qwen.py --condicion qwen_fewshot --split test \
  --salida "$S8_RUN/qwen_fewshot.jsonl"
python scripts/evaluar_qwen.py --condicion qwen_lora --split test \
  --adaptador modelos/lora-soporte --salida "$S8_RUN/qwen_lora.jsonl"

python scripts/resumir.py --carpeta "$S8_RUN" --destino "$S8_RUN/informe"
```

Elige copiar o ejecutar cada condición, no ambas sobre el mismo destino.
Comprueba que las seis tienen 30 casos, modelos/rutas correctos y el mismo
manifiesto. Una ejecución pendiente debe seguir identificada como pendiente.

## 14. Interpretar y entregar

Lee por separado JSON válido, esquema válido, categoría, prioridad y ambas
etiquetas correctas. Compara aciertos sobre 30, mediana de latencia, tokens y
costes disponibles. Un ajuste que empata o empeora también cumple el experimento
si está realizado y explicado correctamente. No ajustes usando estos 30 casos.

El coste Google queda `null` en el evaluador de soporte: consulta el consumo y
la facturación del proyecto para el importe, sin inventarlo. Separa el coste
del job del de inferencia; no extrapoles la tarifa del base al ajustado.
Para Claude, las tarifas declaradas dan una estimación a partir de tokens,
no una factura. Declara fecha, moneda, región, proveedor y límites de la muestra.

Tu entrega debe permitir comprobar:

- Código/conversor y versiones, datos originales/convertidos, `ids.json` y manifiesto.
- Configuración pública de la receta; nunca `.env` rellenado, ADC ni claves.
- Recibo del job y `estado-final.json` con éxito, modelo y endpoint reales.
- Tres JSONL remotos y tres de Qwen, con originales y errores conservados.
- Informe y explicación de calidad, latencia, consumo, coste y opción elegida.
- Qué has ejecutado, qué reutilizaste de S7 y qué sigue pendiente.

`resultados/` está ignorado por Git: no basta con entregar un enlace al código
para adjuntar tus evidencias. Prepara una carpeta/ZIP depurado según indique el
docente. Identificadores de proyecto/recursos son metadatos: compártelos solo
por el canal de entrega previsto. No incluyas secretos en capturas.

No hay limpieza automática. Con el administrador, identifica jobs/modelos,
endpoints y objetos propios que debas retirar al terminar, conserva la evidencia
y revisa costes residuales. No borres recursos compartidos ni cierres el proyecto
del curso. Eliminar datos o cerrar la terminal no cancela un job en ejecución.

## 15. Si te atascas: síntoma → comprobación → siguiente paso

| Síntoma | Qué comprobar y hacer |
|---|---|
| `python`/import no existe | Raíz, Python 3.12, entorno activo y lock; usa `python -m pip` del mismo intérprete |
| `gcloud` no existe | Instala CLI oficial; no se instala con el lock Python |
| `DefaultCredentialsError` | Login ADC del paso 3; API key de AI Studio no sirve aquí |
| `SERVICE_DISABLED` | Proyecto correcto y APIs del paso 4; administrador habilita Service Usage si falta |
| `403` en Python | Identidad ADC, proyecto de cuota, aiplatform.user y serviceUsageConsumer; no confundir login CLI con ADC |
| `403` al subir/listar bucket | Nombre y proyecto, Object Creator/Viewer y Bucket Reader del alumno |
| Job no puede leer train | URI real y agente de servicio real; administrador revisa Viewer y políticas del proyecto |
| `429` o cuota de jobs | Cuota/capacidad y jobs activos; consulta el existente y no dupliques |
| `404` al consultar ajustado | Endpoint del job terminado, inferencia `eu`, trabajo `europe-west4`; no cambiar IDs a mano |
| `412` en subida | Objeto existente: conciliar evidencia; el script no sobrescribe ni reanuda automáticamente |
| Timeout al crear | Busca el job en consola antes de repetir; resultado remoto incierto |
| `Datos o protocolo cambiaron` | `git diff -- datos prompts config`, CRLF y variables Qwen; no editar hashes para silenciar |
| Claude `401`/`403` | Clave/organización/acceso API, sin mostrar clave; probar desarrollo antes del lote |
| Claude `429` o saldo | Límites y facturación API de la cuenta autorizada; conservar fallo |
| JSON cortado/inválido | Original y motivo de parada; cuenta como fallo, no reparar ni aumentar límite usando test |
| Archivo de salida ya existe | Conservarlo; otro ensayo necesita ruta nueva y motivo declarado |
| `Lote incompleto` | Faltan casos: conservar parcial; no presentarlo como comparación final ni mezclar pasadas |

## 16. Pedir ayuda a una IA con el repositorio abierto

Abre la raíz del repo en tu editor y pide que lea `AGENTS.md` y esta guía.
Puedes copiar este texto y completar el paso y error **sin secretos**:

> Estoy siguiendo `docs/guia-S8-gemini-claude.md`, paso [N], para las incidencias
> de S7. Lee `AGENTS.md`, ese paso y el script real antes de responder. Explícame
> qué entra, qué hace y qué debe salir, con palabras sencillas. Estoy en [sistema
> y entorno]. Ejecuté [comando sin claves] y obtuve [error depurado]. Distingue
> lo que puedo comprobar offline de lo que requiere administrador, subir datos
> o consumir saldo. Dame el siguiente comando y cómo verificarlo. No inventes
> IDs, entrenamientos ni resultados y no cambies el test ni sus hashes.

Puedes pedir «explícame ADC», «por qué hay roles distintos para mí y Google»,
«qué contiene una fila JSONL», «qué hace `tuning_config`» o «cómo interpretar el
informe». La IA debe localizar funciones y argumentos reales, no trasladar los
comandos de Nébula Hogar a este ejercicio. El archivo `CLAUDE.md` enlaza a
`AGENTS.md`, de modo que comparte el mismo mapa de ayuda.

## Fuentes y versión

Guía revisada el **15/10/2026** contra scripts del repo y documentación oficial.
El entorno fija `google-genai==1.42.0`; usa `vertexai=True`. No copies sin adaptar
constructores `enterprise=True` de versiones nuevas ni actualices el lock solo
porque un ejemplo oficial sea posterior.

- [Crear, consultar y usar tuning Gemini](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/tuning/supervised-tuning/use).
- [Modelos compatibles, límites y cuotas](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/tuning/supervised-tuning).
- [Roles de IA y permisos](https://docs.cloud.google.com/iam/docs/roles-permissions/aiplatform).
- [Identidades de agentes gestionados](https://docs.cloud.google.com/iam/docs/service-agents).
- [Roles de Cloud Storage](https://docs.cloud.google.com/iam/docs/roles-permissions/storage).
- [Service Usage](https://docs.cloud.google.com/iam/docs/roles-permissions/serviceusage).
- [Autenticación ADC local](https://docs.cloud.google.com/docs/authentication/set-up-adc-local-dev-environment).
- [Precios Google](https://cloud.google.com/vertex-ai/generative-ai/pricing) y [precios Anthropic](https://platform.claude.com/docs/en/about-claude/pricing).

La documentación y contratos locales no prueban permisos/cuota de una cuenta
nueva ni un entrenamiento real. Esa evidencia la producen tu job terminado y
las predicciones de su endpoint. Para detalles de proveedores consulta
[modelos remotos](modelos-remotos.md); para la entrega, [ejercicio](ejercicio.md).

# GUÍA DE LECTURA · scripts/preparar_datos.py
# Construye los datos ficticios y fija el protocolo antes de medir el test.
# Los textos y etiquetas se definen aquí; no se extraen de clientes ni de tickets reales.
# 40 situaciones de train × 6 prefijos producen 240 registros, 60 por categoría.
# Los prefijos aportan diversidad limitada: no son 240 problemas independientes.
# Ejecutar este script ESCRIBE datos, few-shot y manifiesto; leerlo no los regenera.

"""Dataset sintético autorado y separado por familia antes del entrenamiento."""

# Hace visible la raíz del proyecto al ejecutar un archivo desde scripts/.
import _bootstrap  # noqa: F401

# Convierte entre texto JSON y diccionarios/listas Python, manteniendo respuestas como evidencia.
import json

# Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
from curso.common import ROOT, write_jsonl, validate_splits, experiment_manifest

# Texto, categoría, prioridad; el propósito y las etiquetas se fijan antes del test.
# Situaciones iniciales de train, agrupadas por categoría, con prioridad anotada antes del test.
TRAIN = {
    "acceso": [
        ("No puedo iniciar sesión con mi cuenta del curso.", "media"),
        (
            "El enlace para cambiar mi contraseña ha caducado y no puedo entrar.",
            "media",
        ),
        ("Mi usuario aparece sin permiso para abrir las lecciones compradas.", "media"),
        (
            "El código de verificación no llega y estoy bloqueado fuera de mi cuenta.",
            "media",
        ),
        ("He perdido el segundo factor y no consigo entrar.", "media"),
        ("¿Cómo puedo cambiar mi contraseña si todavía puedo entrar?", "baja"),
        ("¿Dónde consulto los permisos de mi usuario?", "baja"),
        (
            "Todos los usuarios de la promoción están bloqueados al iniciar sesión.",
            "alta",
        ),
        ("Al entrar veo datos privados de la cuenta de otra persona.", "alta"),
        ("Me gustaría que el login permitiera recordar mi usuario.", "baja"),
    ],
    "facturacion": [
        ("Tengo dos cargos del mismo curso en mi tarjeta.", "media"),
        ("Mi factura tiene un importe diferente al cobro realizado.", "media"),
        ("El pago de mi matrícula se rechaza y no puedo completarlo.", "media"),
        ("No encuentro la factura de un pago ya realizado.", "media"),
        ("La renovación se ha cobrado después de que yo la cancelara.", "media"),
        ("¿Qué medios de pago aceptáis para un curso?", "baja"),
        ("¿Puedo pagar la matrícula en varios plazos?", "baja"),
        ("Todos los alumnos del grupo han recibido un cobro duplicado.", "alta"),
        (
            "La página de facturas muestra datos fiscales privados de otros clientes.",
            "alta",
        ),
        ("Urgente: quiero información sobre los precios para el próximo año.", "baja"),
    ],
    "tecnico": [
        ("El vídeo de la lección no se reproduce y no puedo estudiar.", "media"),
        ("No puedo subir mi ejercicio porque el botón devuelve un error.", "media"),
        ("La reunión en directo no tiene sonido para mí y no puedo seguirla.", "media"),
        ("La descarga del material falla y no puedo obtener el documento.", "media"),
        ("El examen se cierra antes de enviar mis respuestas.", "media"),
        (
            "Un icono aparece desalineado, pero las funciones siguen disponibles.",
            "baja",
        ),
        ("Los subtítulos tienen una errata que no impide seguir la lección.", "baja"),
        ("La plataforma está caída para todos los usuarios.", "alta"),
        (
            "Al descargar un ejercicio se exponen documentos privados de otros alumnos.",
            "alta",
        ),
        ("Sugiero añadir una opción de velocidad al reproductor.", "baja"),
    ],
    "consulta": [
        ("¿A qué hora comienza la siguiente clase?", "baja"),
        ("¿Dónde está el calendario del programa?", "baja"),
        ("¿Cuántas horas de estudio recomendáis por semana?", "baja"),
        ("¿El certificado se entrega al terminar el curso?", "baja"),
        ("¿Puedo revisar las grabaciones después del directo?", "baja"),
        ("Quiero saber qué conocimientos previos necesito.", "baja"),
        ("Gracias por la clase, no tengo ninguna incidencia.", "baja"),
        ("Tengo un problema, pero todavía no sé explicar qué ocurre.", "baja"),
        ("Urgente: ¿cuál es el temario del siguiente módulo?", "baja"),
        ("Propongo organizar una sesión de preguntas al terminar el programa.", "baja"),
    ],
}
# Casos distintos para comprobar pérdida; no se usan en los pasos del optimizador.
VALIDATION = [
    (
        "El inicio de sesión me devuelve credenciales incorrectas y no puedo entrar.",
        "acceso",
        "media",
    ),
    ("Mi invitación al campus expiró y no tengo acceso.", "acceso", "media"),
    ("La cuenta no me deja acceder al módulo que he pagado.", "acceso", "media"),
    (
        "Ningún miembro de nuestra promoción puede acceder con su usuario.",
        "acceso",
        "alta",
    ),
    (
        "La pantalla de sesión enseña el correo y perfil privado de otra persona.",
        "acceso",
        "alta",
    ),
    (
        "Puedo entrar; solo quiero cambiar el nombre que aparece en mi perfil.",
        "acceso",
        "baja",
    ),
    ("¿Se puede usar una llave física para iniciar sesión?", "acceso", "baja"),
    (
        "No puedo entrar porque la sesión se cierra al validar mi identidad.",
        "acceso",
        "media",
    ),
    ("El recibo de mi curso tiene un impuesto incorrecto.", "facturacion", "media"),
    (
        "Mi banco confirma el pago, pero el campus lo marca como pendiente.",
        "facturacion",
        "media",
    ),
    (
        "El sistema no me permite añadir los datos fiscales de mi factura.",
        "facturacion",
        "media",
    ),
    (
        "A toda la clase nos han cargado la misma cuota dos veces.",
        "facturacion",
        "alta",
    ),
    (
        "El PDF de facturación contiene datos privados de otra empresa.",
        "facturacion",
        "alta",
    ),
    ("¿Ofrecéis descuento si me matriculo en dos programas?", "facturacion", "baja"),
    (
        "¿Cuándo se emite una factura de matrícula en condiciones normales?",
        "facturacion",
        "baja",
    ),
    ("El cuestionario de práctica no abre y no puedo completarlo.", "tecnico", "media"),
    (
        "El buscador no encuentra ninguna lección y me impide localizar el contenido.",
        "tecnico",
        "media",
    ),
    ("El editor pierde mi respuesta al guardar el ejercicio.", "tecnico", "media"),
    (
        "Todos los alumnos ven una pantalla en blanco al abrir las lecciones.",
        "tecnico",
        "alta",
    ),
    ("Un enlace de material permite leer entregas privadas ajenas.", "tecnico", "alta"),
    (
        "El título tiene una falta, pero puedo utilizar todo el contenido.",
        "tecnico",
        "baja",
    ),
    ("La animación tarda un poco; no bloquea ninguna acción.", "tecnico", "baja"),
    ("¿Habrá una tutoría adicional este mes?", "consulta", "baja"),
    ("¿El curso incluye un proyecto individual?", "consulta", "baja"),
    ("¿Qué libros de referencia se utilizan?", "consulta", "baja"),
    ("¿Cuántas sesiones tiene este módulo?", "consulta", "baja"),
    ("Todo me va bien. ¿Habrá una encuesta al finalizar?", "consulta", "baja"),
    ("Estoy preocupado, necesito ayuda pero no he dicho con qué.", "consulta", "baja"),
    ("Urgente, quiero conocer el nombre del siguiente profesor.", "consulta", "baja"),
    ("Me gustaría una sesión de orientación laboral.", "consulta", "baja"),
]
# Casos reservados para medir tras congelar el protocolo; no se ajustan a partir de sus resultados.
TEST = [
    (
        "Mi enlace de activación abre una página que dice token inválido; no puedo entrar.",
        "acceso",
        "media",
    ),
    (
        "Después de cambiar de móvil, el segundo factor me impide acceder al campus.",
        "acceso",
        "media",
    ),
    (
        "El campus dice que mi cuenta está suspendida y me bloquea las lecciones.",
        "acceso",
        "media",
    ),
    (
        "A todo el grupo nos aparece usuario inexistente al intentar entrar.",
        "acceso",
        "alta",
    ),
    (
        "En mi perfil autenticado puedo ver el teléfono privado de otra alumna.",
        "acceso",
        "alta",
    ),
    (
        "Sigo entrando correctamente. ¿Cómo cambio el correo asociado a mi usuario?",
        "acceso",
        "baja",
    ),
    (
        "Urgente: ¿puedo configurar un método de acceso adicional? No estoy bloqueado.",
        "acceso",
        "baja",
    ),
    (
        "La autenticación me devuelve un bucle de redirecciones y no puedo iniciar sesión.",
        "acceso",
        "media",
    ),
    (
        "Me han cobrado una mensualidad que no figura en mi contrato de matrícula.",
        "facturacion",
        "media",
    ),
    (
        "La factura está a nombre de una entidad incorrecta y necesito corregirla.",
        "facturacion",
        "media",
    ),
    (
        "Mi cupón se anuncia vigente pero el checkout lo rechaza y no puedo pagar.",
        "facturacion",
        "media",
    ),
    (
        "Todos los usuarios de nuestra empresa han recibido cargos no reconocidos.",
        "facturacion",
        "alta",
    ),
    (
        "El listado de recibos muestra números de cuenta privados de otros clientes.",
        "facturacion",
        "alta",
    ),
    (
        "¿Qué diferencia de precio hay entre las modalidades de matrícula?",
        "facturacion",
        "baja",
    ),
    (
        "Urgente: ¿las tarifas publicadas incluyen los impuestos? Solo estoy consultando.",
        "facturacion",
        "baja",
    ),
    (
        "La entrega se queda al 99 por ciento y no puedo enviar el proyecto.",
        "tecnico",
        "media",
    ),
    (
        "El reproductor muestra error de códec y me impide abrir el vídeo.",
        "tecnico",
        "media",
    ),
    (
        "El formulario de examen no acepta ninguna respuesta y no puedo terminar.",
        "tecnico",
        "media",
    ),
    (
        "Para todos los usuarios el botón de abrir el directo devuelve error 503.",
        "tecnico",
        "alta",
    ),
    (
        "La búsqueda del campus devuelve notas privadas de otros estudiantes.",
        "tecnico",
        "alta",
    ),
    (
        "El color del icono del tema es incorrecto, aunque puedo utilizar la plataforma.",
        "tecnico",
        "baja",
    ),
    (
        "Urgente: la barra de progreso está desalineada; no bloquea ninguna función.",
        "tecnico",
        "baja",
    ),
    ("¿Puedo cursar este módulo antes que el anterior?", "consulta", "baja"),
    ("¿Cuál es la fecha límite orientativa del proyecto final?", "consulta", "baja"),
    ("¿Los directos se quedan grabados durante todo el programa?", "consulta", "baja"),
    ("¿Hay grupos de estudio voluntarios para practicar?", "consulta", "baja"),
    (
        "He terminado las lecciones. ¿Qué pasos siguen para el certificado?",
        "consulta",
        "baja",
    ),
    (
        "Ayuda por favor, algo va mal y no puedo concretarlo todavía.",
        "consulta",
        "baja",
    ),
    (
        "Urgente: ¿qué herramientas se presentan en el siguiente bloque?",
        "consulta",
        "baja",
    ),
    ("Sugiero publicar un glosario de términos junto al temario.", "consulta", "baja"),
]


# Devuelve categoría, prioridad y una respuesta genérica fija para cada categoría.
# Es la anotación docente de un caso ficticio, no la salida de un modelo.
def target(cat, priority):
    # Respuesta HTTP que se inspeccionará y cerrará después de leerla.
    response = {
        "acceso": "Indica el error de acceso y el paso en que aparece, sin compartir tu contraseña.",
        "facturacion": "Indica el concepto y la fecha del pago para revisar la incidencia de facturación.",
        "tecnico": "Indica la función afectada y el mensaje de error para revisar el problema.",
        "consulta": "Indica qué información necesitas para poder ayudarte con la consulta.",
    }[cat]
    # Crea la anotación objetivo que se guardará junto a la incidencia del dataset.
    return {"categoria": cat, "prioridad": priority, "respuesta": response}


# Genera train, validación, test, desarrollo, IDs few-shot y manifiesto en disco.
# Debe usarse para preparar el experimento antes de las evaluaciones.
def main():
    # Seis prefijos fijos que producen variantes cercanas de cada situación de entrenamiento.
    prefixes = (
        "",
        "Os escribo por lo siguiente: ",
        "Quería comunicar esto: ",
        "Sobre el campus del curso: ",
        "Necesito orientación: ",
        "Buenas, mi situación es esta: ",
    )
    # Registros de la partición o lote actual, conservados como una lista de diccionarios.
    rows = []
    # Recorre categorías y situaciones iniciales que ya tienen etiquetas fijadas.
    for cat, seeds in TRAIN.items():
        # Enumera las diez situaciones iniciales de cada categoría para crear su familia estable.
        for j, (text, priority) in enumerate(seeds):
            # Grupo de variantes de una situación para controlar solapamiento entre particiones.
            family = f"train-{cat}-{j:02d}"
            # Añade cada uno de los seis prefijos a la misma situación; todas sus variantes comparten familia.
            for variant, prefix in enumerate(prefixes):
                # Crea un ejemplo con ID único, familia, incidencia y etiqueta/respuesta objetivo.
                rows.append(
                    {
                        "id": f"TR-{len(rows) + 1:03d}",
                        "familia": family,
                        "incidencia": prefix + text,
                        "objetivo": target(cat, priority),
                    }
                )
    # Guarda los registros del paso como JSONL, una fila por ejemplo y sin reparar respuestas.
    write_jsonl(ROOT / "datos/train.jsonl", rows)
    # Construye validación y test con sus propios casos; las familias usan prefijos distintos de train.
    for name, seeds in (("validacion", VALIDATION), ("test", TEST)):
        # Guarda los registros del paso como JSONL, una fila por ejemplo y sin reparar respuestas.
        write_jsonl(
            ROOT / f"datos/{name}.jsonl",
            [
                {
                    "id": f"{'VA' if name == 'validacion' else 'TE'}-{i + 1:02d}",
                    "familia": f"{name}-{i:02d}",
                    "incidencia": text,
                    "objetivo": target(cat, priority),
                }
                for i, (text, cat, priority) in enumerate(seeds)
            ],
        )
    # Selección fija, solo del entrenamiento; se congela antes de usar test.
    # Escribe el contenido legible en disco para conservar configuración o resultados de este paso.
    (ROOT / "prompts/fewshot-ids.json").write_text(
        json.dumps(["TR-001", "TR-061", "TR-121", "TR-181"])
    )
    # Guarda los registros del paso como JSONL, una fila por ejemplo y sin reparar respuestas.
    write_jsonl(
        ROOT / "datos/desarrollo.jsonl",
        [
            {
                "id": "DE-01",
                "incidencia": "La lección no carga para mí y no puedo verla.",
                "objetivo": target("tecnico", "media"),
            },
            {
                "id": "DE-02",
                "incidencia": "¿Podéis explicar las modalidades de pago?",
                "objetivo": target("facturacion", "baja"),
            },
            {
                "id": "DE-03",
                "incidencia": "Todos los usuarios estamos fuera del campus; el login falla.",
                "objetivo": target("acceso", "alta"),
            },
            {
                "id": "DE-04",
                "incidencia": "Tengo un cobro repetido en mi matrícula.",
                "objetivo": target("facturacion", "media"),
            },
            {
                "id": "DE-05",
                "incidencia": "Urgente, me gustaría saber el temario del mes próximo.",
                "objetivo": target("consulta", "baja"),
            },
            {
                "id": "DE-06",
                "incidencia": "Necesito ayuda, pero aún no he descrito el problema.",
                "objetivo": target("consulta", "baja"),
            },
        ],
    )
    # Crea la carpeta necesaria; parents/exist_ok controlan subcarpetas y rutas ya existentes.
    (ROOT / "config").mkdir(exist_ok=True)
    # Escribe el contenido legible en disco para conservar configuración o resultados de este paso.
    (ROOT / "config/experimento-congelado.json").write_text(
        json.dumps(experiment_manifest(), ensure_ascii=False, indent=2) + "\n"
    )
    # Muestra el resultado/estado de este paso en la terminal; las claves no se añaden al informe.
    print(json.dumps(validate_splits()))


# Punto de entrada: ejecutar como script llama main; importar el módulo solo define sus funciones.
# Invoca el punto de entrada únicamente cuando se ejecuta este archivo como programa.
if __name__ == "__main__":
    main()

# GUÍA DE LECTURA · tests/test_contracts.py
# Pruebas del protocolo docente: particiones, few-shot, JSON y aislamiento.
# Usan datos sintéticos, directorios temporales y mocks, sin entrenar modelos.
# Cada test describe una propiedad que debe mantenerse al modificar el proyecto.

# Convierte entre texto JSON y diccionarios/listas Python, manteniendo respuestas como evidencia.
# Gestiona rutas de importación y el intérprete actual de Python.
# Crea directorios temporales de prueba que se retiran al salir del contexto.
# Organiza pruebas y aserciones; los tests se ejecutan solo al invocarlos.
import json, sys, tempfile, unittest

# Path permite construir rutas, comprobar archivos y leer/escribir sin concatenar separadores a mano.
from pathlib import Path

# Añade una ruta de importación para poder reutilizar el paquete curso y los scripts del proyecto.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
# Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
from curso.common import (
    ROOT,
    assert_frozen,
    read_jsonl,
    validate_answer,
    validate_splits,
    messages,
    write_jsonl,
)

# Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
from curso.remote import gemini_record, google_client
from gemini_tuning import job_spec
from resumir import summarize

# Mocks sustituyen red/configuración para probar contratos sin proveedores reales.
from unittest.mock import patch
from entrenar_lora import training_output


# Agrupa contratos del dataset, mensajes, validación y archivos del experimento.
class Contracts(unittest.TestCase):
    # Prepara una respuesta objetivo ficticia antes de cada prueba; no genera datos remotos.
    def setUp(self):
        self.expected = {
            "categoria": "acceso",
            "prioridad": "media",
            "respuesta": "No compartas contraseñas.",
        }

    # Rechaza Markdown alrededor del JSON y valores no estándar como NaN.
    def test_json_strict_no_fences(self):
        # Exige rechazo o ausencia de la propiedad que rompería el contrato.
        self.assertFalse(
            validate_answer(
                "```json\n" + json.dumps(self.expected) + "\n```", self.expected
            )["json_valido"]
        )
        # Exige rechazo o ausencia de la propiedad que rompería el contrato.
        self.assertFalse(
            validate_answer('{"categoria":NaN}', self.expected)["json_valido"]
        )

    # Acepta el esquema correcto y rechaza claves adicionales.
    def test_exact_fields_and_enum(self):
        # Exige que la propiedad comprobada sea verdadera.
        self.assertTrue(
            validate_answer(json.dumps(self.expected), self.expected)["esquema_valido"]
        )
        wrong = {**self.expected, "otra": 1}
        # Exige rechazo o ausencia de la propiedad que rompería el contrato.
        self.assertFalse(
            validate_answer(json.dumps(wrong), self.expected)["esquema_valido"]
        )

    # Comprueba que un fallo cuenta como caso y que coste desconocido sigue siendo None.
    def test_unparseable_is_in_denominator(self):
        # Registros de la partición o lote actual, conservados como una lista de diccionarios.
        rows = [
            {
                "valoracion": validate_answer(json.dumps(self.expected), self.expected),
                "latencia_s": 1,
                "coste_usd": None,
            },
            {
                "valoracion": validate_answer("error", self.expected),
                "latencia_s": None,
                "coste_usd": None,
                "error": "TimeoutError",
            },
        ]
        # Comprueba igualdad exacta con el resultado esperado para este contrato.
        # Comprueba que el valor desconocido se mantiene como None.
        out = summarize(rows)
        self.assertEqual((out["casos"], out["categoria_correcta"]), (2, 1))
        self.assertIsNone(out["coste_usd"])

    # Verifica las cantidades y los hashes del experimento publicado.
    def test_splits_and_freeze(self):
        # Comprueba igualdad exacta con el resultado esperado para este contrato.
        # Exige que la propiedad comprobada sea verdadera.
        self.assertEqual(
            validate_splits(), {"train": 240, "validacion": 30, "test": 30}
        )
        self.assertTrue(assert_frozen())

    # Introduce una familia de train en una copia temporal de test y exige rechazo.
    def test_duplicate_family_is_rejected(self):
        # Usa una carpeta temporal de pruebas para conservar intactos los datos y adaptadores reales.
        with tempfile.TemporaryDirectory() as tmp:
            for s in ("train", "validacion", "test"):
                # Registros de la partición o lote actual, conservados como una lista de diccionarios.
                rows = read_jsonl(ROOT / f"datos/{s}.jsonl")
                if s == "test":
                    rows[0]["familia"] = "train-acceso-00"
                # Guarda los registros del paso como JSONL, una fila por ejemplo y sin reparar respuestas.
                write_jsonl(Path(tmp) / f"{s}.jsonl", rows)
            # La prueba espera una excepción: rechazar esta entrada es el comportamiento correcto.
            with self.assertRaisesRegex(ValueError, "Familia"):
                validate_splits(tmp)

    # Comprueba que los ejemplos few-shot vienen de train y la consulta nueva queda al final.
    def test_fewshot_only_from_training(self):
        msg = messages("CONSULTA NUEVA", True)
        # Comprueba igualdad exacta con el resultado esperado para este contrato.
        self.assertEqual(msg[-1]["content"], "CONSULTA NUEVA")
        self.assertEqual(len(msg), 10)
        training = {r["incidencia"] for r in read_jsonl(ROOT / "datos/train.jsonl")}
        # Exige que la propiedad comprobada sea verdadera.
        self.assertTrue(
            all(m["content"] in training for m in msg[1:-1] if m["role"] == "user")
        )

    # Comprueba que convertir formato conserva la incidencia y la respuesta objetivo.
    def test_gemini_conversion_preserves_labels(self):
        # Un ejemplo: ID/metadatos para nosotros e incidencia/objetivo para la tarea.
        # Evidencia de un caso con entrada, objetivo, respuesta original, fecha y valoración.
        row = read_jsonl(ROOT / "datos/test.jsonl")[0]
        record = gemini_record(row)
        # Comprueba igualdad exacta con el resultado esperado para este contrato.
        self.assertEqual(record["contents"][0]["parts"][0]["text"], row["incidencia"])
        # Comprueba igualdad exacta con el resultado esperado para este contrato.
        self.assertEqual(
            json.loads(record["contents"][1]["parts"][0]["text"]), row["objetivo"]
        )
        # Comprueba que no se añadió un campo/valor ajeno al recorrido permitido.
        self.assertNotIn("id", record)

    # Relaciona cada línea convertida con el ID de su ejemplo original.
    def test_gemini_id_mapping(self):
        ids = json.loads((ROOT / "datos/gemini/ids.json").read_text())
        for s in ("train", "validacion", "test"):
            # Comprueba igualdad exacta con el resultado esperado para este contrato.
            self.assertEqual(
                ids[s]["ids_por_linea"],
                [r["id"] for r in read_jsonl(ROOT / f"datos/{s}.jsonl")],
            )

    # Exige detenerse antes de crear cliente Google cuando falta el proyecto explícito.
    def test_missing_project_prevents_client(self):
        # Sustituye dependencias/configuración durante la prueba y restaura sus valores al terminar.
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("curso.remote.load_config"),
        ):
            # La prueba espera una excepción: rechazar esta entrada es el comportamiento correcto.
            with self.assertRaisesRegex(ValueError, "laboratorio"):
                google_client()

    # Comprueba ubicación y ausencia del test en la especificación de tuning.
    def test_job_spec_never_includes_test(self):
        # Especificación del trabajo de tuning; describirla no equivale a ejecutarlo.
        spec = job_spec("curso-demo", "bucket-demo")
        # Comprueba que no se añadió un campo/valor ajeno al recorrido permitido.
        # Comprueba igualdad exacta con el resultado esperado para este contrato.
        self.assertNotIn("test", json.dumps(spec))
        self.assertEqual(spec["location"], "europe-west4")
        # La prueba espera una excepción: rechazar esta entrada es el comportamiento correcto.
        with self.assertRaises(ValueError):
            job_spec(None, "bucket")

    # Falla si importar el módulo intentara enviar una petición HTTP.
    def test_imports_do_not_connect(self):
        import importlib

        # Sustituye dependencias/configuración durante la prueba y restaura sus valores al terminar.
        with patch("requests.post", side_effect=AssertionError("network")):
            importlib.reload(sys.modules["curso.remote"])

    # Prueba que una salida vacía se acepta y un adaptador previo permanece intacto.
    def test_training_does_not_overwrite_existing_artifacts(self):
        # Usa una carpeta temporal de pruebas para conservar intactos los datos y adaptadores reales.
        with tempfile.TemporaryDirectory() as tmp:
            # Ruta concreta de lectura/escritura; el código utiliza Path para gestionar carpetas y archivos.
            # Crea la carpeta necesaria; parents/exist_ok controlan subcarpetas y rutas ya existentes.
            path = Path(tmp) / "adapter"
            path.mkdir()
            # Comprueba igualdad exacta con el resultado esperado para este contrato.
            self.assertEqual(training_output(path), path.resolve())
            artifact = path / "adapter_model.safetensors"
            artifact.write_bytes(b"preserve")
            # La prueba espera una excepción: rechazar esta entrada es el comportamiento correcto.
            with self.assertRaisesRegex(ValueError, "Salida no vacía"):
                training_output(path)
            # Comprueba igualdad exacta con el resultado esperado para este contrato.
            self.assertEqual(artifact.read_bytes(), b"preserve")


# Punto de entrada: ejecutar como script llama main; importar el módulo solo define sus funciones.
if __name__ == "__main__":
    unittest.main()

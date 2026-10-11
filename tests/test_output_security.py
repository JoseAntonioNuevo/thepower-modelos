# GUÍA DE LECTURA · tests/test_output_security.py
# Pruebas de límites de respuestas y preservación de la evidencia.
# Mocks simulan proveedores; una prueba usa gzip servido en loopback.
# Se comprueban redirects, tamaño descomprimido, JSON, CSV y cierre de recursos.
# Ninguna prueba llama a un proveedor remoto ni imprime una clave real.

# Escribe/lee tablas CSV con comillas y separadores correctos.
import csv

# Comprime la respuesta ficticia para comprobar límites de bytes descomprimidos.
import gzip

# Buffers de memoria permiten inspeccionar salidas de pruebas sin escribir archivos reales.
import io

# Convierte entre texto JSON y diccionarios/listas Python, manteniendo respuestas como evidencia.
import json

# Gestiona rutas de importación y el intérprete actual de Python.
import sys

# Crea directorios temporales de prueba que se retiran al salir del contexto.
import tempfile

# Organiza pruebas y aserciones; los tests se ejecutan solo al invocarlos.
import unittest

# Captura stdout en memoria para inspeccionar el informe generado dentro de una prueba.
from contextlib import redirect_stdout

# Path permite construir rutas, comprobar archivos y leer/escribir sin concatenar separadores a mano.
from pathlib import Path

# Mocks sustituyen red/configuración para probar contratos sin proveedores reales.
from unittest.mock import Mock, patch

# Añade una ruta de importación para poder reutilizar el paquete curso y los scripts del proyecto.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
# Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
from curso.common import (
    ROOT,
    MAX_MODEL_TEXT_CHARS,
    assert_frozen,
    read_jsonl,
    validate_answer,
    write_jsonl,
)

# Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
from curso.http import MAX_RESPONSE_BYTES, bounded_json, joined_text, model_ids

# Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
from curso.remote import chat_local, chat_openrouter, chat_google
from resumir import csv_text, main as summarize_main
from preflight import main as preflight_main
from evaluar_remotos import main as evaluate_main


# Fabrica una respuesta HTTP simulada con bloques, cabeceras y métodos observables.
# Permite comprobar límites y cierre sin acceder a un proveedor externo.
def response(data=None, chunks=None, headers=None, status=200):
    # Respuesta simulada cuyo cuerpo/estado controla la prueba para reproducir éxito o error.
    value = Mock(status_code=status, headers=headers or {})
    value.iter_content.return_value = (
        chunks if chunks is not None else [json.dumps(data).encode()]
    )
    return value


# Agrupa pruebas de transporte acotado y conservación de resultados originales.
class OutputSecurity(unittest.TestCase):
    # Prueba límites de cabecera, bytes descomprimidos, plazo y cierre en rutas de error.
    def test_http_header_and_decoded_chunks_are_bounded_and_closed(self):
        oversized = response(headers={"Content-Length": str(MAX_RESPONSE_BYTES + 1)})
        # La prueba espera una excepción: rechazar esta entrada es el comportamiento correcto.
        with self.assertRaisesRegex(ValueError, "bytes"):
            bounded_json(oversized)
        # Comprueba que esta entrada se rechazó antes de ejecutar el paso que debía evitarse.
        oversized.iter_content.assert_not_called()
        # Comprueba que el recurso se utilizó/cerró exactamente una vez en este recorrido.
        oversized.close.assert_called_once()
        for headers in ({}, {"Content-Length": "2"}):
            chunks = response(chunks=[b"x" * 8192] * 129, headers=headers)
            # La prueba espera una excepción: rechazar esta entrada es el comportamiento correcto.
            with self.assertRaisesRegex(ValueError, "bytes"):
                bounded_json(chunks)
            # Comprueba que el recurso se utilizó/cerró exactamente una vez en este recorrido.
            chunks.close.assert_called_once()
        valid = response({"choices": []})
        # Comprueba igualdad exacta con el resultado esperado para este contrato.
        self.assertEqual(bounded_json(valid), {"choices": []})
        # Comprueba que el recurso se utilizó/cerró exactamente una vez en este recorrido.
        valid.close.assert_called_once()

    # Sirve gzip por HTTP local y comprueba el límite sobre su tamaño descomprimido.
    def test_real_compressed_response_limit(self):
        # Servidor HTTP de la biblioteca estándar para la demo en loopback.
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        # Hilos, locks y semáforos permiten atender HTTP y coordinar el modelo compartido.
        from threading import Thread

        # Cliente HTTP; las peticiones se realizan al llamarlo, no por el simple import.
        import requests

        body = gzip.compress(
            json.dumps({"text": "x" * (MAX_RESPONSE_BYTES + 1)}).encode()
        )

        # Handler local de prueba que sirve una respuesta comprimida controlada.
        class Handler(BaseHTTPRequestHandler):
            # Sirve bytes gzip controlados para comprobar que el lector limita el tamaño descomprimido.
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            # Silencia el log del servidor temporal para que la prueba solo muestre su resultado.
            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = requests.get(
                f"http://127.0.0.1:{server.server_port}/", stream=True, timeout=2
            )
            # La prueba espera una excepción: rechazar esta entrada es el comportamiento correcto.
            with self.assertRaisesRegex(ValueError, "bytes"):
                bounded_json(result)
        finally:
            # Cierra el recurso abierto; libera socket/conexión aunque el paso anterior haya fallado.
            server.shutdown()
            server.server_close()
            thread.join()

    # Rechaza catálogos mal formados y listas/bloques de respuesta demasiado grandes.
    def test_catalog_shape_and_part_limits(self):
        # Comprueba igualdad exacta con el resultado esperado para este contrato.
        self.assertEqual(
            model_ids({"data": [{"id": "one"}, {"id": "two"}]}), ["one", "two"]
        )
        for entries in (
            [{}],
            [None],
            [{"id": "x" * 1025}],
            [{"id": "one"}] * 257,
            "bad",
        ):
            # La prueba espera una excepción: rechazar esta entrada es el comportamiento correcto.
            with (
                self.subTest(entries_type=type(entries).__name__),
                self.assertRaises(ValueError),
            ):
                model_ids({"data": entries})
        # Comprueba igualdad exacta con el resultado esperado para este contrato.
        self.assertEqual(
            joined_text([{"text": "uno"}, {"text": "dos"}], "text"), "unodos"
        )
        for parts in (
            [{"text": "x"}] * 65,
            [{"text": "x" * MAX_RESPONSE_BYTES}, {"text": "x"}],
            [{"text": 1}],
        ):
            # La prueba espera una excepción: rechazar esta entrada es el comportamiento correcto.
            with self.assertRaises(ValueError):
                joined_text(parts, "text")

    # Comprueba stream=True, ausencia de redirects y conservación del texto en los tres clientes.
    def test_all_http_clients_stream_and_preserve_normal_text(self):
        # Texto original que generó el modelo; no se repara antes de valorarlo o guardarlo.
        raw = '{"categoria":"consulta","prioridad":"baja","respuesta":"Información."}'
        envelope = {"choices": [{"message": {"content": raw}}]}
        for native in (False, True):
            # Respuesta simulada cuyo cuerpo/estado controla la prueba para reproducir éxito o error.
            value = response(
                {"output": [{"type": "message", "content": raw}]}
                if native
                else envelope
            )
            # Sustituye dependencias/configuración durante la prueba y restaura sus valores al terminar.
            with (
                patch("curso.remote.load_config"),
                patch("requests.post", return_value=value) as post,
            ):
                # Comprueba igualdad exacta con el resultado esperado para este contrato.
                self.assertEqual(
                    chat_local("incidencia", model="demo", native=native)[
                        "respuesta_original"
                    ],
                    raw,
                )
                # Exige que la propiedad comprobada sea verdadera.
                self.assertTrue(post.call_args.kwargs["stream"])
                # Exige rechazo o ausencia de la propiedad que rompería el contrato.
                self.assertFalse(post.call_args.kwargs["allow_redirects"])
        # Respuesta simulada cuyo cuerpo/estado controla la prueba para reproducir éxito o error.
        value = response(envelope)
        # Sustituye dependencias/configuración durante la prueba y restaura sus valores al terminar.
        with (
            patch("curso.remote.load_config"),
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "dummy"}),
            patch("requests.post", return_value=value) as post,
        ):
            # Comprueba igualdad exacta con el resultado esperado para este contrato.
            self.assertEqual(
                chat_openrouter("incidencia", "demo")["respuesta_original"], raw
            )
            # Exige que la propiedad comprobada sea verdadera.
            self.assertTrue(post.call_args.kwargs["stream"])
            # Exige rechazo o ausencia de la propiedad que rompería el contrato.
            self.assertFalse(post.call_args.kwargs["allow_redirects"])
        # Respuesta simulada cuyo cuerpo/estado controla la prueba para reproducir éxito o error.
        value = response({"candidates": [{"content": {"parts": [{"text": raw}]}}]})
        # Sesión simulada: devuelve una respuesta controlada y permite verificar petición y cierre sin Google.
        session = Mock()
        session.post.return_value = value
        # Sustituye dependencias/configuración durante la prueba y restaura sus valores al terminar.
        with (
            patch("curso.remote.load_config"),
            patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "demo"}),
            patch("google.auth.default", return_value=(Mock(), None)),
            patch(
                "google.auth.transport.requests.AuthorizedSession", return_value=session
            ),
        ):
            # Comprueba igualdad exacta con el resultado esperado para este contrato.
            self.assertEqual(
                chat_google("incidencia", "demo")["respuesta_original"], raw
            )
            # Exige que la propiedad comprobada sea verdadera.
            self.assertTrue(session.post.call_args.kwargs["stream"])
            # Exige rechazo o ausencia de la propiedad que rompería el contrato.
            self.assertFalse(session.post.call_args.kwargs["allow_redirects"])
            # Comprueba que el recurso se utilizó/cerró exactamente una vez en este recorrido.
            session.close.assert_called_once()
        too_long = response(
            {"choices": [{"message": {"content": "x" * (MAX_MODEL_TEXT_CHARS + 1)}}]}
        )
        # Sustituye dependencias/configuración durante la prueba y restaura sus valores al terminar.
        with (
            patch("curso.remote.load_config"),
            patch("requests.post", return_value=too_long),
        ):
            # Destino donde esta ejecución guardará sus archivos.
            output = chat_local("incidencia", model="demo", native=False)
        # Comprueba igualdad exacta con el resultado esperado para este contrato.
        self.assertEqual(output["respuesta_original"], "x" * (MAX_MODEL_TEXT_CHARS + 1))
        # Exige rechazo o ausencia de la propiedad que rompería el contrato.
        self.assertFalse(
            validate_answer(output["respuesta_original"], {})["json_valido"]
        )

    # Rechaza redirects antes de leer el cuerpo y respuestas Google sin candidatos.
    def test_redirects_fail_before_body_or_followup_and_google_no_candidates_fails(
        self,
    ):
        # Respuesta simulada cuyo cuerpo/estado controla la prueba para reproducir éxito o error.
        value = response(
            status=302,
            headers={
                "Location": "http://example.invalid",
                "Content-Length": str(MAX_RESPONSE_BYTES * 2),
            },
        )
        # Sustituye dependencias/configuración durante la prueba y restaura sus valores al terminar.
        with (
            patch("curso.remote.load_config"),
            patch("requests.post", return_value=value) as post,
        ):
            # La prueba espera una excepción: rechazar esta entrada es el comportamiento correcto.
            with self.assertRaisesRegex(RuntimeError, "302"):
                chat_local("incidencia", model="demo", native=False)
            # Exige rechazo o ausencia de la propiedad que rompería el contrato.
            self.assertFalse(post.call_args.kwargs["allow_redirects"])
            # Comprueba que esta entrada se rechazó antes de ejecutar el paso que debía evitarse.
            # Comprueba que el recurso se utilizó/cerró exactamente una vez en este recorrido.
            value.iter_content.assert_not_called()
            value.close.assert_called_once()
        # Respuesta simulada cuyo cuerpo/estado controla la prueba para reproducir éxito o error.
        value = response(
            {"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}}
        )
        # Sesión simulada: devuelve una respuesta controlada y permite verificar petición y cierre sin Google.
        session = Mock()
        session.post.return_value = value
        # Sustituye dependencias/configuración durante la prueba y restaura sus valores al terminar.
        with (
            patch("curso.remote.load_config"),
            patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "demo"}),
            patch("google.auth.default", return_value=(Mock(), None)),
            patch(
                "google.auth.transport.requests.AuthorizedSession", return_value=session
            ),
        ):
            # La prueba espera una excepción: rechazar esta entrada es el comportamiento correcto.
            with self.assertRaisesRegex(ValueError, "candidato"):
                chat_google("incidencia", "demo")
            # Comprueba que el recurso se utilizó/cerró exactamente una vez en este recorrido.
            value.close.assert_called_once()
            session.close.assert_called_once()

    # Comprueba que el lector reconoce UTF-8 con BOM, UTF-16 y UTF-32.
    def test_json_bom_and_utf_encodings_remain_supported(self):
        data = {"text": "Información con [{ comillas }]"}
        for encoding in ("utf-8-sig", "utf-16", "utf-32"):
            # Comprueba igualdad exacta con el resultado esperado para este contrato.
            with self.subTest(encoding=encoding):
                self.assertEqual(
                    bounded_json(response(chunks=[json.dumps(data).encode(encoding)])),
                    data,
                )

    # Comprueba que el catálogo y preflight usan lectura acotada y rechazan IDs inválidos.
    def test_catalog_discovery_and_preflight_use_bounded_reader(self):
        # Respuesta HTTP del catálogo; enumera modelos disponibles sin generar una respuesta.
        listing = response({"data": [{"id": "demo"}]})
        answer = response({"choices": [{"message": {"content": "exact text"}}]})
        # Sustituye dependencias/configuración durante la prueba y restaura sus valores al terminar.
        with (
            patch("curso.remote.load_config"),
            patch.dict("os.environ", {}, clear=True),
            patch("requests.get", return_value=listing) as get,
            patch("requests.post", return_value=answer),
        ):
            # Comprueba igualdad exacta con el resultado esperado para este contrato.
            self.assertEqual(
                chat_local("incidencia", native=False)["respuesta_original"],
                "exact text",
            )
            # Exige que la propiedad comprobada sea verdadera.
            self.assertTrue(get.call_args.kwargs["stream"])
            # Exige rechazo o ausencia de la propiedad que rompería el contrato.
            self.assertFalse(get.call_args.kwargs["allow_redirects"])
        malformed = response({"data": [{}]})
        out = io.StringIO()
        # Sustituye dependencias/configuración durante la prueba y restaura sus valores al terminar.
        with (
            patch("sys.argv", ["preflight", "--local"]),
            patch("preflight.load_config"),
            patch("requests.get", return_value=malformed) as get,
            redirect_stdout(out),
        ):
            preflight_main()
        # Comprueba que el resultado contiene la evidencia o estado esperado.
        self.assertIn("inválida", json.loads(out.getvalue())["modelos_locales"])
        # Exige que la propiedad comprobada sea verdadera.
        self.assertTrue(get.call_args.kwargs["stream"])
        # Exige rechazo o ausencia de la propiedad que rompería el contrato.
        self.assertFalse(get.call_args.kwargs["allow_redirects"])

    # Distingue llaves dentro de texto y estructura anidada y aplica límites del validador.
    def test_json_size_depth_and_quoted_braces(self):
        # Resultado/manifestación de referencia con el que se contrasta la ejecución actual.
        expected = {
            "categoria": "consulta",
            "prioridad": "baja",
            "respuesta": '[{{ texto "citado" }}]',
        }
        # Exige que la propiedad comprobada sea verdadera.
        self.assertTrue(
            validate_answer(json.dumps(expected), expected)["esquema_valido"]
        )
        for raw in ("[" * 33 + "0" + "]" * 33, "x" * (MAX_MODEL_TEXT_CHARS + 1), None):
            # Exige rechazo o ausencia de la propiedad que rompería el contrato.
            self.assertFalse(validate_answer(raw, expected)["json_valido"])
        invalid = response(chunks=[b"[" * 33 + b"0" + b"]" * 33])
        # La prueba espera una excepción: rechazar esta entrada es el comportamiento correcto.
        with self.assertRaisesRegex(ValueError, "profundidad"):
            bounded_json(invalid)
        # Comprueba que el recurso se utilizó/cerró exactamente una vez en este recorrido.
        invalid.close.assert_called_once()

    # Comprueba protección de fórmulas en CSV y conservación exacta de los originales en JSON.
    def test_csv_neutralizes_text_and_preserves_json_evidence(self):
        for value in (
            "=1+1",
            "+1",
            "-1",
            "@SUM(1)",
            "  =1+1",
            "\t=1+1",
            "\r=1+1",
            "\n=1+1",
            "=SUM(1,2)\nnext",
        ):
            # Comprueba igualdad exacta con el resultado esperado para este contrato.
            self.assertEqual(csv_text(value), "'" + value)
        # Comprueba igualdad exacta con el resultado esperado para este contrato.
        for value in ("normal", "", 1, -1, None):
            self.assertEqual(csv_text(value), value)
        # Usa una carpeta temporal de pruebas para conservar intactos los datos y adaptadores reales.
        with tempfile.TemporaryDirectory() as tmp:
            # Carpeta que se recorre para leer lotes o generar una salida temporal.
            folder = Path(tmp)
            # Resultados originales reunidos para generar JSON, CSV y resumen.
            records = []
            # Recorre cada registro de la partición elegida; conserva el ID para relacionar entradas y resultados.
            for row in read_jsonl(ROOT / "datos/test.jsonl"):
                records.append(
                    {
                        **row,
                        "split": "test",
                        "condicion": "qwen_base",
                        "experimento": assert_frozen(),
                        "respuesta_original": "=1+1",
                        "provider": " @SUM(1)",
                        "latencia_s": None,
                        "coste_usd": None,
                        "valoracion": validate_answer("=1+1", row["objetivo"]),
                    }
                )
            # Guarda los registros del paso como JSONL, una fila por ejemplo y sin reparar respuestas.
            write_jsonl(folder / "qwen_base.jsonl", records)
            # Sustituye dependencias/configuración durante la prueba y restaura sus valores al terminar.
            with (
                patch(
                    "sys.argv",
                    ["resumir", "--carpeta", tmp, "--destino", tmp + "/report"],
                ),
                redirect_stdout(io.StringIO()),
            ):
                summarize_main()
            # El contexto abre el archivo y lo cierra automáticamente al terminar.
            with (folder / "report/resultados.csv").open(newline="") as file:
                first = next(csv.DictReader(file))
            # Comprueba igualdad exacta con el resultado esperado para este contrato.
            self.assertEqual(first["respuesta_original"], "'=1+1")
            # Comprueba igualdad exacta con el resultado esperado para este contrato.
            self.assertEqual(first["provider"], "' @SUM(1)")
            saved = json.loads((folder / "report/resultados.json").read_text())
            # Comprueba igualdad exacta con el resultado esperado para este contrato.
            self.assertEqual(saved[0]["respuesta_original"], "=1+1")
            # Comprueba igualdad exacta con el resultado esperado para este contrato.
            self.assertEqual(saved[0]["provider"], " @SUM(1)")
            # Comprueba igualdad exacta con el resultado esperado para este contrato.
            self.assertEqual(len(saved), 30)

    # Conserva una respuesta demasiado larga como evidencia, pero la marca inválida.
    def test_overlong_bounded_model_text_survives_evaluation_and_report(self):
        # Texto original que generó el modelo; no se repara antes de valorarlo o guardarlo.
        raw = "x" * (MAX_MODEL_TEXT_CHARS + 1)
        # Usa una carpeta temporal de pruebas para conservar intactos los datos y adaptadores reales.
        with tempfile.TemporaryDirectory() as tmp:
            # Ruta concreta de lectura/escritura; el código utiliza Path para gestionar carpetas y archivos.
            path = Path(tmp) / "claude_prompt.jsonl"
            # Sustituye dependencias/configuración durante la prueba y restaura sus valores al terminar.
            with (
                patch(
                    "sys.argv",
                    [
                        "evaluar",
                        "--condicion",
                        "claude_prompt",
                        "--proveedor",
                        "anthropic",
                        "--modelo",
                        "demo",
                        "--split",
                        "test",
                        "--salida",
                        str(path),
                    ],
                ),
                # Clave ficticia: la comprobación previa de credenciales no llega a la red.
                patch.dict("os.environ", {"ANTHROPIC_API_KEY": "dummy"}),
                patch(
                    "evaluar_remotos.chat",
                    return_value={
                        "respuesta_original": raw,
                        "latencia_s": 1,
                        "coste_usd": None,
                    },
                ),
                redirect_stdout(io.StringIO()),
            ):
                evaluate_main()
            # Registros de la partición o lote actual, conservados como una lista de diccionarios.
            rows = read_jsonl(path)
            # Comprueba igualdad exacta con el resultado esperado para este contrato.
            self.assertEqual(len(rows), 30)
            # Exige que la propiedad comprobada sea verdadera.
            self.assertTrue(
                all(
                    row["respuesta_original"] == raw
                    and not row["valoracion"]["json_valido"]
                    for row in rows
                )
            )
            # Sustituye dependencias/configuración durante la prueba y restaura sus valores al terminar.
            with (
                patch(
                    "sys.argv",
                    ["resumir", "--carpeta", tmp, "--destino", tmp + "/report"],
                ),
                redirect_stdout(io.StringIO()),
            ):
                summarize_main()
            saved = json.loads((Path(tmp) / "report/resultados.json").read_text())
            # Comprueba igualdad exacta con el resultado esperado para este contrato.
            self.assertEqual(saved[0]["respuesta_original"], raw)


# Punto de entrada: ejecutar como script llama main; importar el módulo solo define sus funciones.
if __name__ == "__main__":
    unittest.main()

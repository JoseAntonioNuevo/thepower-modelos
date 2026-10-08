import csv
import gzip
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from curso.common import ROOT, MAX_MODEL_TEXT_CHARS, assert_frozen, read_jsonl, validate_answer, write_jsonl
from curso.http import MAX_RESPONSE_BYTES, bounded_json, joined_text, model_ids
from curso.remote import chat_local, chat_openrouter, chat_google
from resumir import csv_text, main as summarize_main
from preflight import main as preflight_main
from evaluar_remotos import main as evaluate_main


def response(data=None, chunks=None, headers=None, status=200):
    value = Mock(status_code=status, headers=headers or {})
    value.iter_content.return_value = chunks if chunks is not None else [json.dumps(data).encode()]
    return value


class OutputSecurity(unittest.TestCase):
    def test_http_header_and_decoded_chunks_are_bounded_and_closed(self):
        oversized = response(headers={"Content-Length": str(MAX_RESPONSE_BYTES + 1)})
        with self.assertRaisesRegex(ValueError, "bytes"):
            bounded_json(oversized)
        oversized.iter_content.assert_not_called()
        oversized.close.assert_called_once()
        for headers in ({}, {"Content-Length": "2"}):
            chunks = response(chunks=[b"x" * 8192] * 129, headers=headers)
            with self.assertRaisesRegex(ValueError, "bytes"):
                bounded_json(chunks)
            chunks.close.assert_called_once()
        valid = response({"choices": []})
        self.assertEqual(bounded_json(valid), {"choices": []})
        valid.close.assert_called_once()

    def test_real_compressed_response_limit(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from threading import Thread
        import requests
        body = gzip.compress(json.dumps({"text": "x" * (MAX_RESPONSE_BYTES + 1)}).encode())
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args): pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            result = requests.get(f"http://127.0.0.1:{server.server_port}/", stream=True, timeout=2)
            with self.assertRaisesRegex(ValueError, "bytes"):
                bounded_json(result)
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_catalog_shape_and_part_limits(self):
        self.assertEqual(model_ids({"data": [{"id": "one"}, {"id": "two"}]}), ["one", "two"])
        for entries in ([{}], [None], [{"id": "x" * 1025}], [{"id": "one"}] * 257, "bad"):
            with self.subTest(entries_type=type(entries).__name__), self.assertRaises(ValueError):
                model_ids({"data": entries})
        self.assertEqual(joined_text([{"text": "uno"}, {"text": "dos"}], "text"), "unodos")
        for parts in ([{"text": "x"}] * 65, [{"text": "x" * MAX_RESPONSE_BYTES}, {"text": "x"}], [{"text": 1}]):
            with self.assertRaises(ValueError): joined_text(parts, "text")

    def test_all_http_clients_stream_and_preserve_normal_text(self):
        raw = '{"categoria":"consulta","prioridad":"baja","respuesta":"Información."}'
        envelope = {"choices": [{"message": {"content": raw}}]}
        for native in (False, True):
            value = response({"output": [{"type": "message", "content": raw}]} if native else envelope)
            with patch("curso.remote.load_config"), patch("requests.post", return_value=value) as post:
                self.assertEqual(chat_local("incidencia", model="demo", native=native)["respuesta_original"], raw)
                self.assertTrue(post.call_args.kwargs["stream"])
                self.assertFalse(post.call_args.kwargs["allow_redirects"])
        value = response(envelope)
        with patch("curso.remote.load_config"), patch.dict("os.environ", {"OPENROUTER_API_KEY": "dummy"}), patch("requests.post", return_value=value) as post:
            self.assertEqual(chat_openrouter("incidencia", "demo")["respuesta_original"], raw)
            self.assertTrue(post.call_args.kwargs["stream"])
            self.assertFalse(post.call_args.kwargs["allow_redirects"])
        value = response({"candidates": [{"content": {"parts": [{"text": raw}]}}]})
        session = Mock(); session.post.return_value = value
        with patch("curso.remote.load_config"), patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "demo"}), patch("google.auth.default", return_value=(Mock(), None)), patch("google.auth.transport.requests.AuthorizedSession", return_value=session):
            self.assertEqual(chat_google("incidencia", "demo")["respuesta_original"], raw)
            self.assertTrue(session.post.call_args.kwargs["stream"])
            self.assertFalse(session.post.call_args.kwargs["allow_redirects"])
            session.close.assert_called_once()
        too_long = response({"choices": [{"message": {"content": "x" * (MAX_MODEL_TEXT_CHARS + 1)}}]})
        with patch("curso.remote.load_config"), patch("requests.post", return_value=too_long):
            output=chat_local("incidencia", model="demo", native=False)
        self.assertEqual(output["respuesta_original"], "x" * (MAX_MODEL_TEXT_CHARS + 1))
        self.assertFalse(validate_answer(output["respuesta_original"], {})["json_valido"])

    def test_redirects_fail_before_body_or_followup_and_google_no_candidates_fails(self):
        value = response(status=302, headers={"Location": "http://example.invalid", "Content-Length": str(MAX_RESPONSE_BYTES * 2)})
        with patch("curso.remote.load_config"), patch("requests.post", return_value=value) as post:
            with self.assertRaisesRegex(RuntimeError, "302"): chat_local("incidencia", model="demo", native=False)
            self.assertFalse(post.call_args.kwargs["allow_redirects"])
            value.iter_content.assert_not_called(); value.close.assert_called_once()
        value=response({"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}})
        session=Mock(); session.post.return_value=value
        with patch("curso.remote.load_config"), patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "demo"}), patch("google.auth.default", return_value=(Mock(),None)), patch("google.auth.transport.requests.AuthorizedSession", return_value=session):
            with self.assertRaisesRegex(ValueError,"candidato"):chat_google("incidencia","demo")
            value.close.assert_called_once();session.close.assert_called_once()

    def test_json_bom_and_utf_encodings_remain_supported(self):
        data={"text": "Información con [{ comillas }]"}
        for encoding in ("utf-8-sig","utf-16","utf-32"):
            with self.subTest(encoding=encoding):self.assertEqual(bounded_json(response(chunks=[json.dumps(data).encode(encoding)])),data)

    def test_catalog_discovery_and_preflight_use_bounded_reader(self):
        listing = response({"data": [{"id": "demo"}]})
        answer = response({"choices": [{"message": {"content": "exact text"}}]})
        with patch("curso.remote.load_config"), patch.dict("os.environ", {}, clear=True), patch("requests.get", return_value=listing) as get, patch("requests.post", return_value=answer):
            self.assertEqual(chat_local("incidencia", native=False)["respuesta_original"], "exact text")
            self.assertTrue(get.call_args.kwargs["stream"])
            self.assertFalse(get.call_args.kwargs["allow_redirects"])
        malformed = response({"data": [{}]})
        out = io.StringIO()
        with patch("sys.argv", ["preflight", "--local"]), patch("preflight.load_config"), patch("requests.get", return_value=malformed) as get, redirect_stdout(out):
            preflight_main()
        self.assertIn("inválida", json.loads(out.getvalue())["modelos_locales"])
        self.assertTrue(get.call_args.kwargs["stream"])
        self.assertFalse(get.call_args.kwargs["allow_redirects"])

    def test_json_size_depth_and_quoted_braces(self):
        expected = {"categoria": "consulta", "prioridad": "baja", "respuesta": '[{{ texto "citado" }}]'}
        self.assertTrue(validate_answer(json.dumps(expected), expected)["esquema_valido"])
        for raw in ("[" * 33 + "0" + "]" * 33, "x" * (MAX_MODEL_TEXT_CHARS + 1), None):
            self.assertFalse(validate_answer(raw, expected)["json_valido"])
        invalid = response(chunks=[b"[" * 33 + b"0" + b"]" * 33])
        with self.assertRaisesRegex(ValueError, "profundidad"): bounded_json(invalid)
        invalid.close.assert_called_once()

    def test_csv_neutralizes_text_and_preserves_json_evidence(self):
        for value in ("=1+1", "+1", "-1", "@SUM(1)", "  =1+1", "\t=1+1", "\r=1+1", "\n=1+1", '=SUM(1,2)\nnext'):
            self.assertEqual(csv_text(value), "'" + value)
        for value in ("normal", "", 1, -1, None): self.assertEqual(csv_text(value), value)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            records = []
            for row in read_jsonl(ROOT / "datos/test.jsonl"):
                records.append({**row, "split": "test", "condicion": "qwen_base", "experimento": assert_frozen(),
                    "respuesta_original": "=1+1", "provider": " @SUM(1)", "latencia_s": None, "coste_usd": None,
                    "valoracion": validate_answer("=1+1", row["objetivo"])})
            write_jsonl(folder / "qwen_base.jsonl", records)
            with patch("sys.argv", ["resumir", "--carpeta", tmp, "--destino", tmp + "/report"]), redirect_stdout(io.StringIO()):
                summarize_main()
            with (folder / "report/resultados.csv").open(newline="") as file: first = next(csv.DictReader(file))
            self.assertEqual(first["respuesta_original"], "'=1+1")
            self.assertEqual(first["provider"], "' @SUM(1)")
            saved = json.loads((folder / "report/resultados.json").read_text())
            self.assertEqual(saved[0]["respuesta_original"], "=1+1")
            self.assertEqual(saved[0]["provider"], " @SUM(1)")
            self.assertEqual(len(saved), 30)

    def test_overlong_bounded_model_text_survives_evaluation_and_report(self):
        raw="x"*(MAX_MODEL_TEXT_CHARS+1)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"claude_prompt.jsonl"
            with patch("sys.argv",["evaluar","--condicion","claude_prompt","--modelo","demo","--split","test","--salida",str(path)]), patch("evaluar_remotos.chat_openrouter",return_value={"respuesta_original":raw,"latencia_s":1,"coste_usd":None}), redirect_stdout(io.StringIO()):
                evaluate_main()
            rows=read_jsonl(path)
            self.assertEqual(len(rows),30)
            self.assertTrue(all(row["respuesta_original"]==raw and not row["valoracion"]["json_valido"] for row in rows))
            with patch("sys.argv",["resumir","--carpeta",tmp,"--destino",tmp+"/report"]),redirect_stdout(io.StringIO()):
                summarize_main()
            saved=json.loads((Path(tmp)/"report/resultados.json").read_text())
            self.assertEqual(saved[0]["respuesta_original"],raw)


if __name__ == "__main__": unittest.main()

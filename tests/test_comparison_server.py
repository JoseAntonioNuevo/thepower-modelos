"""Contrato HTTP de las dos condiciones y del streaming usado por isbetter.ai."""
import json
import sys
import threading
import unittest
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from servir_comparacion import ALLOWED_ORIGIN, MODEL_IDS, ComparisonServer


class FakeEngine:
    metadata = {"base_model": "Qwen/Qwen3-0.6B", "enable_thinking": False}

    def events(self, model_id, messages):
        yield {"content": model_id}
        yield {"finish_reason": "stop", "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}}


class ComparisonHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ComparisonServer(("127.0.0.1", 0), FakeEngine())
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def body(self, model, stream=False):
        return {"model": model, "messages": [{"role": "user", "content": "Incidencia de prueba"}], "stream": stream}

    def test_discovery_and_origin_scope(self):
        response = requests.get(self.url + "/v1/models", headers={"Origin": ALLOWED_ORIGIN}, timeout=2)
        self.assertEqual([m["id"] for m in response.json()["data"]], list(MODEL_IDS))
        self.assertEqual([m["adapter_enabled"] for m in response.json()["data"]], [False, True])
        self.assertEqual(response.headers["Access-Control-Allow-Origin"], ALLOWED_ORIGIN)
        blocked = requests.get(self.url + "/v1/models", headers={"Origin": "https://otro.example"}, timeout=2)
        self.assertEqual(blocked.status_code, 403)
        self.assertNotIn("Access-Control-Allow-Origin", blocked.headers)

    def test_both_names_dispatch_and_stream_finishes(self):
        for model in MODEL_IDS:
            response = requests.post(self.url + "/v1/chat/completions", json=self.body(model), timeout=2)
            self.assertEqual(response.json()["choices"][0]["message"]["content"], model)
        response = requests.post(self.url + "/v1/chat/completions", json=self.body(MODEL_IDS[1], True), timeout=2)
        events = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(line) for line in events[:-1]]
        self.assertEqual("".join(c["choices"][0]["delta"].get("content", "") for c in chunks), MODEL_IDS[1])
        self.assertEqual(chunks[-1]["usage"]["total_tokens"], 6)

    def test_other_models_and_protocol_changes_rejected(self):
        for change in [{"model": "qwen3.5-4b-mlx"}, {"temperature": 1}, {"max_tokens": 1000}, {"response_format": {"type": "json_object"}}]:
            response = requests.post(self.url + "/v1/chat/completions", json={**self.body(MODEL_IDS[0]), **change}, timeout=2)
            self.assertEqual(response.status_code, 400)

    def test_external_host_is_rejected_even_without_origin(self):
        response = requests.get(self.url + "/v1/models", headers={"Host": "external.example"}, timeout=2)
        self.assertEqual(response.status_code, 403)

    def test_full_inference_queue_is_rejected_and_recovers(self):
        for _ in range(4):
            self.assertTrue(self.server.inference_slots.acquire(blocking=False))
        try:
            response = requests.post(self.url + "/v1/chat/completions", json=self.body(MODEL_IDS[0]), timeout=2)
            self.assertEqual(response.status_code, 429)
        finally:
            for _ in range(4):
                self.server.inference_slots.release()
        response = requests.post(self.url + "/v1/chat/completions", json=self.body(MODEL_IDS[0]), timeout=2)
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()

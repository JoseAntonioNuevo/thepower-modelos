"""API local para comparar el mismo Qwen con y sin el adaptador de soporte.

Una sola instancia, inferencia serializada y parámetros del experimento congelado.
No modifica datos, pesos ni resultados. CORS limitado a isbetter.ai.
"""
import _bootstrap  # noqa: F401

import argparse
import json
import threading
import time
import uuid
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from curso.common import ROOT, MODEL_ID, REVISION, GENERATION, assert_frozen, digest


MODEL_IDS = ("qwen-original", "qwen-lora-soporte")
ALLOWED_ORIGIN = "https://isbetter.ai"


def validate_request(body):
    if not isinstance(body, dict) or body.get("model") not in MODEL_IDS:
        raise ValueError("Selecciona qwen-original o qwen-lora-soporte")
    messages = body.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 24:
        raise ValueError("messages debe contener entre 1 y 24 mensajes")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant"):
            raise ValueError("Rol de mensaje no admitido")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError("Solo se admiten mensajes de texto no vacíos")
    if messages[-1]["role"] != "user":
        raise ValueError("El último mensaje debe ser del usuario")
    if body.get("temperature") not in (None, 0) or body.get("max_tokens") not in (None, GENERATION["max_new_tokens"]):
        raise ValueError("La comparación fija temperature=0 y max_tokens=180")
    if body.get("tools") or body.get("response_format") or body.get("n", 1) != 1:
        raise ValueError("JSON como texto, sin tools ni formato impuesto")
    if not isinstance(body.get("stream", False), bool):
        raise ValueError("stream debe ser booleano")
    return body["model"], messages


class ComparisonEngine:
    def __init__(self, adapter):
        assert_frozen()
        adapter = Path(adapter).resolve()
        config = json.loads((adapter / "adapter_config.json").read_text())
        if config["base_model_name_or_path"] != MODEL_ID or config.get("bias") != "none":
            raise ValueError("El adaptador debe pertenecer al mismo Qwen y no modificar bias")
        from curso.qwen import load_qwen

        self.model, self.tokenizer, self.device = load_qwen(str(adapter))
        self.lock = threading.Lock()
        self.metadata = {
            "base_model": MODEL_ID,
            "revision": REVISION,
            "adapter_sha256": digest(adapter / "adapter_model.safetensors"),
            "device": self.device,
            "generation": GENERATION,
            "enable_thinking": False,
            "inference": "serializada; ambas opciones comparten el mismo modelo",
        }

    def events(self, model_id, messages):
        import torch
        from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
        from curso.qwen import render

        cancelled = threading.Event()

        class Cancelled(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return cancelled.is_set()

        # Incluye generación y restauración del adaptador: las peticiones de la
        # batalla llegan a la vez, pero nunca cambian el adaptador de otra petición.
        with self.lock:
            text = render(self.tokenizer, messages)
            inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)
            prompt_tokens = inputs["input_ids"].shape[1]
            if prompt_tokens + GENERATION["max_new_tokens"] > 4096:
                raise ValueError("La consulta supera el contexto de 4096 tokens de esta demo")
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120)
            completed, errors = [], []

            def generate():
                try:
                    context = self.model.disable_adapter() if model_id == MODEL_IDS[0] else nullcontext()
                    with context, torch.inference_mode():
                        output = self.model.generate(
                            **inputs, **GENERATION, streamer=streamer,
                            pad_token_id=self.tokenizer.eos_token_id,
                            stopping_criteria=StoppingCriteriaList([Cancelled()]),
                        )
                    completed.append(output[0, prompt_tokens:])
                except Exception as exc:
                    errors.append(exc)
                    streamer.end()

            worker = threading.Thread(target=generate, daemon=True)
            worker.start()
            try:
                for fragment in streamer:
                    if fragment:
                        yield {"content": fragment}
            finally:
                cancelled.set()
                # No liberar el lock mientras el hilo todavía cambia el adaptador.
                worker.join()
            if errors:
                raise errors[0]
            count = len(completed[0])
            yield {
                "finish_reason": "length" if count >= GENERATION["max_new_tokens"] else "stop",
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": count, "total_tokens": prompt_tokens + count},
            }


class ComparisonServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 8

    def __init__(self, address, engine):
        self.engine = engine
        self.request_slots = threading.BoundedSemaphore(16)
        self.inference_slots = threading.BoundedSemaphore(4)
        super().__init__(address, ComparisonHandler)

    def process_request(self, request, client_address):
        if not self.request_slots.acquire(blocking=False):
            try:
                request.settimeout(1)
                request.sendall(b"HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.request_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.request_slots.release()


class ComparisonHandler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, fmt, *args):
        # Solo método/estado, nunca prompts, claves ni cuerpos de petición.
        print(fmt % args, flush=True)

    def origin_allowed(self):
        return self.headers.get("Origin") in (None, ALLOWED_ORIGIN)

    def request_allowed(self):
        # Evita que un hostname externo resuelto a loopback se trate como local.
        try:
            destination = urlsplit("//" + self.headers.get("Host", ""))
            local = (destination.hostname in ("localhost", "127.0.0.1")
                     and destination.port == self.server.server_port
                     and destination.username is None and destination.password is None
                     and not destination.path and not destination.query and not destination.fragment)
        except ValueError:
            local = False
        return local and self.origin_allowed()

    def cors_headers(self):
        if self.headers.get("Origin") == ALLOWED_ORIGIN:
            self.send_header("Access-Control-Allow-Origin", ALLOWED_ORIGIN)
            self.send_header("Vary", "Origin")

    def send_json(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        if not self.request_allowed():
            self.send_json(403, {"error": {"message": "Origen no admitido"}})
            return
        self.send_response(204)
        self.cors_headers()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()

    def do_GET(self):
        if not self.request_allowed():
            self.send_json(403, {"error": {"message": "Origen no admitido"}})
            return
        path = urlsplit(self.path).path
        if path == "/v1/models":
            self.send_json(200, {"object": "list", "data": [
                {"id": name, "object": "model", "owned_by": "Qwen", "base_model": MODEL_ID,
                 "adapter_enabled": name == MODEL_IDS[1]} for name in MODEL_IDS
            ]})
        elif path == "/health":
            self.send_json(200, {"status": "ready", "models": MODEL_IDS, **self.server.engine.metadata})
        else:
            self.send_json(404, {"error": {"message": "Ruta no encontrada"}})

    def do_POST(self):
        if not self.request_allowed():
            self.send_json(403, {"error": {"message": "Origen no admitido"}})
            return
        if urlsplit(self.path).path != "/v1/chat/completions":
            self.send_json(404, {"error": {"message": "Ruta no encontrada"}})
            return
        started = False
        events = None
        acquired = False
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError("Tamaño de petición no admitido")
            body = json.loads(self.rfile.read(length))
            model_id, messages = validate_request(body)
            acquired = self.server.inference_slots.acquire(blocking=False)
            if not acquired:
                self.send_json(429, {"error": {"message": "Cola de inferencia completa; espera a terminar la comparación"}})
                return
            identity = {"id": "chatcmpl-" + uuid.uuid4().hex, "created": int(time.time()), "model": model_id}
            events = self.server.engine.events(model_id, messages)
            if body.get("stream", False):
                # Obtener el primer fragmento antes de responder 200 permite
                # devolver un error real de inferencia si falla al comenzar.
                first = next(events)
                self.send_response(200)
                self.cors_headers()
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                started = True

                def chunk(event):
                    delta = {"content": event["content"]} if "content" in event else {}
                    data = {**identity, "object": "chat.completion.chunk", "choices": [
                        {"index": 0, "delta": delta, "finish_reason": event.get("finish_reason")}
                    ]}
                    if "usage" in event:
                        data["usage"] = event["usage"]
                    self.wfile.write(("data: " + json.dumps(data, ensure_ascii=False) + "\n\n").encode())
                    self.wfile.flush()

                chunk(first)
                for event in events:
                    chunk(event)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                parts, last = [], {}
                for event in events:
                    if "content" in event:
                        parts.append(event["content"])
                    else:
                        last = event
                self.send_json(200, {**identity, "object": "chat.completion", "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "".join(parts)},
                     "finish_reason": last["finish_reason"]}
                ], "usage": last["usage"]})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            if started:
                self.wfile.write(("data: " + json.dumps({"error": {"message": str(exc)}}) + "\n\n").encode())
                self.wfile.flush()
            else:
                self.send_json(400 if isinstance(exc, (ValueError, KeyError)) else 500,
                               {"error": {"message": str(exc)}})
        finally:
            if events is not None:
                events.close()
            if acquired:
                self.server.inference_slots.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=1235)
    parser.add_argument("--adaptador", default=str(ROOT / "modelos/lora-soporte"))
    args = parser.parse_args()
    print("Cargando el mismo Qwen y su adaptador de soporte…", flush=True)
    engine = ComparisonEngine(args.adaptador)
    server = ComparisonServer(("127.0.0.1", args.port), engine)
    print(json.dumps({"url": f"http://localhost:{args.port}/v1", "models": MODEL_IDS, **engine.metadata}, ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

# GUÍA DE LECTURA · scripts/servir_comparacion.py
# API de demostración para isbetter.ai en 127.0.0.1:1235 por defecto.
# Dos nombres visibles comparten la misma base: original y base con LoRA.
# El bloqueo impide cambiar el estado del adaptador durante otra generación.
# HTTP puede atender varias conexiones, pero la inferencia es una a una.
# CORS y Host restringen acceso; los semáforos limitan conexiones y consultas en espera.
# No es un servidor público ni un benchmark de velocidad entre condiciones.

"""API local para comparar el mismo Qwen con y sin el adaptador de soporte.

Una sola instancia, inferencia serializada y parámetros del experimento congelado.
No modifica datos, pesos ni resultados. CORS limitado a isbetter.ai.
"""

# Hace visible la raíz del proyecto al ejecutar un archivo desde scripts/.
import _bootstrap  # noqa: F401

# Permite leer opciones de la terminal y generar ayuda del comando.
import argparse

# Convierte entre texto JSON y diccionarios/listas Python, manteniendo respuestas como evidencia.
import json

# Hilos, locks y semáforos permiten atender HTTP y coordinar el modelo compartido.
import threading

# Mide duración y prepara fechas; perf_counter/monotonic sirven para intervalos, no fechas del calendario.
import time

# Genera identificadores distintos para las respuestas de chat.
import uuid

# Context managers cierran/restauran recursos al salir, también cuando hay errores.
from contextlib import nullcontext

# Servidor HTTP de la biblioteca estándar para la demo en loopback.
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Path permite construir rutas, comprobar archivos y leer/escribir sin concatenar separadores a mano.
from pathlib import Path

# Separa ruta, hostname y puerto antes de decidir si una petición está autorizada.
from urllib.parse import urlsplit

# Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
from curso.common import ROOT, MODEL_ID, REVISION, GENERATION, assert_frozen, digest


# Dos alias de la misma base; activar/desactivar LoRA es la única diferencia de pesos.
MODEL_IDS = ("qwen-original", "qwen-lora-soporte")
# Único origen de navegador autorizado a consultar esta demo local.
ALLOWED_ORIGIN = "https://isbetter.ai"


# Valida el cuerpo recibido de isbetter y devuelve nombre de modelo y mensajes.
# Impide cambiar parámetros que romperían la comparación del experimento.
def validate_request(body):
    # Validación previa: Selecciona qwen-original o qwen-lora-soporte.
    if not isinstance(body, dict) or body.get("model") not in MODEL_IDS:
        raise ValueError("Selecciona qwen-original o qwen-lora-soporte")
    # Extrae mensajes del cuerpo HTTP; se validarán roles, longitud y contenido.
    messages = body.get("messages")
    # Validación previa: messages debe contener entre 1 y 24 mensajes.
    if not isinstance(messages, list) or not 1 <= len(messages) <= 24:
        raise ValueError("messages debe contener entre 1 y 24 mensajes")
    # Valida cada mensaje antes de enviarlo a la plantilla/tokenizer del modelo.
    for message in messages:
        # Validación previa: Rol de mensaje no admitido.
        if not isinstance(message, dict) or message.get("role") not in (
            "system",
            "user",
            "assistant",
        ):
            raise ValueError("Rol de mensaje no admitido")
        # Validación previa: Solo se admiten mensajes de texto no vacíos.
        if (
            not isinstance(message.get("content"), str)
            or not message["content"].strip()
        ):
            raise ValueError("Solo se admiten mensajes de texto no vacíos")
    # Validación previa: El último mensaje debe ser del usuario.
    if messages[-1]["role"] != "user":
        raise ValueError("El último mensaje debe ser del usuario")
    # Validación previa: La comparación fija temperature=0 y max_tokens=180.
    if body.get("temperature") not in (None, 0) or body.get("max_tokens") not in (
        None,
        GENERATION["max_new_tokens"],
    ):
        raise ValueError("La comparación fija temperature=0 y max_tokens=180")
    # Validación previa: JSON como texto, sin tools ni formato impuesto.
    if body.get("tools") or body.get("response_format") or body.get("n", 1) != 1:
        raise ValueError("JSON como texto, sin tools ni formato impuesto")
    # Validación previa: stream debe ser booleano.
    if not isinstance(body.get("stream", False), bool):
        raise ValueError("stream debe ser booleano")
    return body["model"], messages


# Motor compartido de Qwen: una instancia y dos condiciones de adaptador.
class ComparisonEngine:
    # Carga una base con su adaptador y prepara el lock y la identidad del motor.
    def __init__(self, adapter):
        # Verifica la identidad del experimento antes de servir las dos condiciones.
        assert_frozen()
        # Resuelve la ruta del adaptador elegido para este servidor.
        adapter = Path(adapter).resolve()
        # Lee la configuración PEFT para comprobar compatibilidad con la base del experimento.
        config = json.loads((adapter / "adapter_config.json").read_text())
        # Validación previa: El adaptador debe pertenecer al mismo Qwen y no modificar bias.
        if (
            config["base_model_name_or_path"] != MODEL_ID
            or config.get("bias") != "none"
        ):
            raise ValueError(
                "El adaptador debe pertenecer al mismo Qwen y no modificar bias"
            )
        # Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
        from curso.qwen import load_qwen

        # Carga una única base con LoRA que ambas condiciones compartirán bajo el lock.
        self.model, self.tokenizer, self.device = load_qwen(str(adapter))
        # Exclusión mutua: ninguna consulta cambia el adaptador mientras otra sigue generando.
        self.lock = threading.Lock()
        # Identifica modelo/revisión/hash del adaptador en /health para relacionar una respuesta con sus pesos.
        self.metadata = {
            "base_model": MODEL_ID,
            "revision": REVISION,
            "adapter_sha256": digest(adapter / "adapter_model.safetensors"),
            "device": self.device,
            "generation": GENERATION,
            "enable_thinking": False,
            "inference": "serializada; ambas opciones comparten el mismo modelo",
        }

    # Produce fragmentos de respuesta y un último evento de consumo/parada.
    # El lock abarca generación y restauración del adaptador para aislar las dos condiciones.
    def events(self, model_id, messages):
        # Motor de tensores y gradientes; durante inferencia se desactivan los gradientes.
        import torch

        # Carga modelos/tokenizers y aporta generación y utilidades de entrenamiento.
        from transformers import (
            TextIteratorStreamer,
            StoppingCriteria,
            StoppingCriteriaList,
        )

        # Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
        from curso.qwen import render

        # Señal compartida para pedir al worker que deje de generar cuando termina o se cierra el consumidor.
        cancelled = threading.Event()

        # Criterio de parada que consulta una señal compartida entre consumidor y worker.
        class Cancelled(StoppingCriteria):
            # Comprueba la señal de cancelación entre pasos de generación y permite detener el worker.
            def __call__(self, input_ids, scores, **kwargs):
                return cancelled.is_set()

        # Incluye generación y restauración del adaptador: las peticiones de la
        # batalla llegan a la vez, pero nunca cambian el adaptador de otra petición.
        # Serializa desde el formateo hasta la restauración del estado PEFT, también durante streaming.
        with self.lock:
            # Texto formateado o decodificado que se pasa al siguiente paso del recorrido.
            text = render(self.tokenizer, messages)
            # Tokeniza el contexto completo y mueve sus tensores al dispositivo del modelo.
            inputs = self.tokenizer(
                text, return_tensors="pt", add_special_tokens=False
            ).to(self.device)
            # Número de tokens de entrada, separado de los tokens nuevos de respuesta.
            prompt_tokens = inputs["input_ids"].shape[1]
            # Validación previa: La consulta supera el contexto de 4096 tokens de esta demo.
            if prompt_tokens + GENERATION["max_new_tokens"] > 4096:
                raise ValueError(
                    "La consulta supera el contexto de 4096 tokens de esta demo"
                )
            # Cola de fragmentos decodificados que comunica el worker de generación con el consumidor HTTP.
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=120
            )
            # Listas de comunicación del worker: tokens finales o excepción, sin perder el error real.
            completed, errors = [], []

            # Worker de inferencia: selecciona original/LoRA, genera y comunica tokens o excepción al consumidor.
            def generate():
                try:
                    # Original desactiva temporalmente LoRA; la otra condición conserva el adaptador cargado.
                    context = (
                        self.model.disable_adapter()
                        if model_id == MODEL_IDS[0]
                        else nullcontext()
                    )
                    # El contexto restaura el estado del adaptador al salir y no se calculan gradientes durante inferencia.
                    with context, torch.inference_mode():
                        # Destino donde esta ejecución guardará sus archivos.
                        output = self.model.generate(
                            **inputs,
                            **GENERATION,
                            streamer=streamer,
                            pad_token_id=self.tokenizer.eos_token_id,
                            stopping_criteria=StoppingCriteriaList([Cancelled()]),
                        )
                    # Conserva solo tokens nuevos, separándolos del prompt, para calcular el consumo final.
                    completed.append(output[0, prompt_tokens:])
                except Exception as exc:
                    # Propaga la excepción del worker al hilo que responde HTTP.
                    errors.append(exc)
                    # Desbloquea al consumidor del streamer cuando la generación terminó con error.
                    streamer.end()

            # Generación y lectura de fragmentos ocurren en hilos separados; el lock sigue protegiendo la base compartida.
            worker = threading.Thread(target=generate, daemon=True)
            # Arranca generación en segundo plano; el hilo actual puede enviar fragmentos conforme llegan.
            worker.start()
            try:
                # Recibe fragmentos de texto conforme el worker de generación los produce.
                for fragment in streamer:
                    # Omite fragmentos vacíos del streamer sin modificar los fragmentos con texto.
                    if fragment:
                        # Entrega un fragmento sin ensamblar todavía toda la respuesta.
                        yield {"content": fragment}
            finally:
                # Solicita detener pasos posteriores si el consumidor se cierra o sale del bucle.
                cancelled.set()
                # No liberar el lock mientras el hilo todavía cambia el adaptador.
                # Espera a que el worker termine antes de liberar el lock y permitir otra condición.
                worker.join()
            # Tras esperar al worker, comunica su fallo en lugar de fingir una respuesta correcta.
            if errors:
                raise errors[0]
            # Cantidad de tokens de respuesta utilizados para informar consumo y motivo de parada.
            count = len(completed[0])
            yield {
                "finish_reason": "length"
                if count >= GENERATION["max_new_tokens"]
                else "stop",
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": count,
                    "total_tokens": prompt_tokens + count,
                },
            }


# Servidor de hilos HTTP con límites independientes de conexiones e inferencias en espera.
class ComparisonServer(ThreadingHTTPServer):
    # Los hilos HTTP no impiden por sí solos que termine el proceso del servidor.
    daemon_threads = True
    # Tamaño de la cola de conexiones del socket antes de atenderlas en hilos.
    request_queue_size = 8

    # Inyecta el motor, crea los límites de concurrencia y registra el handler HTTP.
    def __init__(self, address, engine):
        # Inyecta el motor real o el simulado de tests, conservando el mismo contrato HTTP.
        self.engine = engine
        # Como máximo 16 conexiones atendidas/en curso; se rechaza el exceso sin crear otro hilo.
        self.request_slots = threading.BoundedSemaphore(16)
        # Como máximo cuatro consultas en generación o espera; no implica cuatro inferencias simultáneas.
        self.inference_slots = threading.BoundedSemaphore(4)
        # Inicializa el servidor base y asigna ComparisonHandler como manejador de conexiones.
        super().__init__(address, ComparisonHandler)

    # Reserva una plaza HTTP sin bloquear; si no hay capacidad devuelve 503.
    # Una excepción al crear el hilo devuelve la plaza al semáforo.
    def process_request(self, request, client_address):
        # Reserva plaza sin bloquear; rechaza nuevas conexiones cuando ya hay 16 en curso.
        if not self.request_slots.acquire(blocking=False):
            try:
                # Limita a un segundo la escritura de la respuesta de saturación.
                request.settimeout(1)
                # Informa HTTP 503 sin cuerpo y pide cerrar la conexión rechazada.
                request.sendall(
                    b"HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                )
            finally:
                # Cierra el socket rechazado aunque falle enviar la respuesta.
                self.shutdown_request(request)
            return
        try:
            # Delega en ThreadingHTTPServer la creación del hilo para esta conexión.
            super().process_request(request, client_address)
        except Exception:
            # Devuelve la plaza al semáforo para que una petición posterior pueda utilizarla.
            self.request_slots.release()
            raise

    # Atiende la conexión y libera siempre su plaza HTTP cuando termina.
    def process_request_thread(self, request, client_address):
        try:
            # Ejecuta el handler heredado; el finally posterior devuelve la plaza HTTP.
            super().process_request_thread(request, client_address)
        finally:
            # Devuelve la plaza al semáforo para que una petición posterior pueda utilizarla.
            self.request_slots.release()


# Traduce peticiones HTTP a llamadas del motor y devuelve JSON o eventos SSE.
class ComparisonHandler(BaseHTTPRequestHandler):
    # Prepara la conexión heredada y limita cuánto puede esperar cada lectura.
    def setup(self):
        # Inicializa streams de lectura/escritura del handler antes de fijar timeout.
        super().setup()
        # Evita mantener indefinidamente una conexión con lecturas detenidas.
        self.connection.settimeout(15)

    # Registra método/estado del servidor; no añade prompts ni cuerpos HTTP al log.
    def log_message(self, fmt, *args):
        # Solo método/estado, nunca prompts, claves ni cuerpos de petición.
        # Muestra el resultado/estado de este paso en la terminal; las claves no se añaden al informe.
        print(fmt % args, flush=True)

    # Acepta el origen de isbetter o una petición sin Origin, como curl local.
    def origin_allowed(self):
        return self.headers.get("Origin") in (None, ALLOWED_ORIGIN)

    # Comprueba Host de loopback con el puerto correcto además del Origin.
    # Un dominio externo apuntando a 127.0.0.1 no obtiene acceso por ser loopback.
    def request_allowed(self):
        # Evita que un hostname externo resuelto a loopback se trate como local.
        try:
            # Componentes del Host separados para comprobar nombre, puerto y ausencia de credenciales.
            destination = urlsplit("//" + self.headers.get("Host", ""))
            # Valor local del recorrido actual; su significado concreto depende de la función.
            local = (
                destination.hostname in ("localhost", "127.0.0.1")
                and destination.port == self.server.server_port
                and destination.username is None
                and destination.password is None
                and not destination.path
                and not destination.query
                and not destination.fragment
            )
        except ValueError:
            # Valor local del recorrido actual; su significado concreto depende de la función.
            local = False
        return local and self.origin_allowed()

    # Añade permisos CORS solo para el origen autorizado; otros no reciben esa cabecera.
    def cors_headers(self):
        # Solo añade CORS si el navegador envió exactamente el origen permitido.
        if self.headers.get("Origin") == ALLOWED_ORIGIN:
            # Autoriza exclusivamente a isbetter.ai; no utiliza un comodín.
            self.send_header("Access-Control-Allow-Origin", ALLOWED_ORIGIN)
            # Indica que la respuesta depende del Origin de la petición.
            self.send_header("Vary", "Origin")

    # Serializa payload, envía estado/cabeceras y escribe los bytes de la respuesta HTTP.
    def send_json(self, status, payload):
        # Serializa el objeto y lo codifica en UTF-8 para enviar bytes por HTTP.
        data = json.dumps(payload, ensure_ascii=False).encode()
        # Envía el código HTTP antes de las cabeceras y del cuerpo.
        self.send_response(status)
        # Añade permisos del navegador solo si corresponde al origen validado.
        self.cors_headers()
        # Describe si el cuerpo es JSON completo o un flujo de eventos SSE.
        self.send_header("Content-Type", "application/json; charset=utf-8")
        # Declara la cantidad exacta de bytes de la respuesta JSON completa.
        self.send_header("Content-Length", str(len(data)))
        # Evita que las respuestas de la demo se almacenen en caché HTTP.
        self.send_header("Cache-Control", "no-store")
        # Cierra las cabeceras; a partir de aquí se escribe el cuerpo de respuesta.
        self.end_headers()
        # Envía al cliente los bytes del JSON ya serializado.
        self.wfile.write(data)

    # Responde a la comprobación previa CORS del navegador, sin ejecutar el modelo.
    def do_OPTIONS(self):
        # Comprueba Host y Origin antes de descubrir modelos o ejecutar consultas.
        if not self.request_allowed():
            # Rechaza el origen/destino con un error controlado y sin iniciar inferencia.
            self.send_json(403, {"error": {"message": "Origen no admitido"}})
            return
        # Envía el código HTTP antes de las cabeceras y del cuerpo.
        self.send_response(204)
        # Añade permisos del navegador solo si corresponde al origen validado.
        self.cors_headers()
        # Declara los tres métodos admitidos para el preflight del navegador.
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        # Permite las cabeceras usadas por el cliente compatible de isbetter.
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        # Responde a la comprobación de acceso a red privada que puede pedir el navegador.
        self.send_header("Access-Control-Allow-Private-Network", "true")
        # Cierra las cabeceras; a partir de aquí se escribe el cuerpo de respuesta.
        self.end_headers()

    # Sirve /v1/models para descubrir las dos condiciones y /health para identificar el adaptador.
    def do_GET(self):
        # Comprueba Host y Origin antes de descubrir modelos o ejecutar consultas.
        if not self.request_allowed():
            # Rechaza el origen/destino con un error controlado y sin iniciar inferencia.
            self.send_json(403, {"error": {"message": "Origen no admitido"}})
            return
        # Extrae la ruta HTTP sin mezclar query string con la selección de endpoint.
        path = urlsplit(self.path).path
        # Endpoint de descubrimiento: devuelve los dos alias de una base y su estado de adaptador.
        if path == "/v1/models":
            # Devuelve catálogo/estado o texto completo y consumo con su estructura compatible.
            self.send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": name,
                            "object": "model",
                            "owned_by": "Qwen",
                            "base_model": MODEL_ID,
                            "adapter_enabled": name == MODEL_IDS[1],
                        }
                        for name in MODEL_IDS
                    ],
                },
            )
        # Endpoint de identidad: revisión, hash del adaptador y configuración realmente cargada.
        elif path == "/health":
            # Devuelve catálogo/estado o texto completo y consumo con su estructura compatible.
            self.send_json(
                200,
                {"status": "ready", "models": MODEL_IDS, **self.server.engine.metadata},
            )
        else:
            # Devuelve ruta desconocida; no ejecuta ningún comportamiento alternativo.
            self.send_json(404, {"error": {"message": "Ruta no encontrada"}})

    # Valida la consulta, limita la cola y responde con JSON completo o streaming SSE.
    # Libera recursos aunque el navegador se desconecte o la inferencia falle.
    def do_POST(self):
        # Comprueba Host y Origin antes de descubrir modelos o ejecutar consultas.
        if not self.request_allowed():
            # Rechaza el origen/destino con un error controlado y sin iniciar inferencia.
            self.send_json(403, {"error": {"message": "Origen no admitido"}})
            return
        # Solo admite generación en /v1/chat/completions.
        if urlsplit(self.path).path != "/v1/chat/completions":
            # Devuelve ruta desconocida; no ejecuta ningún comportamiento alternativo.
            self.send_json(404, {"error": {"message": "Ruta no encontrada"}})
            return
        # Marca si ya se enviaron cabeceras de streaming; después solo puede notificarse el error dentro del flujo.
        started = False
        # Generador del motor: produce texto y un último evento con consumo y motivo de parada.
        events = None
        # Indica si esta petición reservó una plaza de inferencia, para liberarla exactamente una vez.
        acquired = False
        try:
            # Cantidad de bytes declarada para el cuerpo; se valida antes de leerlo.
            length = int(self.headers.get("Content-Length", "0"))
            # Validación previa: Tamaño de petición no admitido.
            if not 0 < length <= 65536:
                raise ValueError("Tamaño de petición no admitido")
            # Lee el cuerpo dentro del límite de 64 KiB y lo convierte a un objeto Python.
            body = json.loads(self.rfile.read(length))
            # Aplica contrato de modelos, roles y parámetros antes de reservar una consulta.
            model_id, messages = validate_request(body)
            # Indica si esta petición reservó una plaza de inferencia, para liberarla exactamente una vez.
            acquired = self.server.inference_slots.acquire(blocking=False)
            # Con cuatro consultas en curso/espera, devuelve 429 sin crear más trabajo de generación.
            if not acquired:
                # Informa saturación; el cliente debe esperar, sin cambiar silenciosamente de modelo.
                self.send_json(
                    429,
                    {
                        "error": {
                            "message": "Cola de inferencia completa; espera a terminar la comparación"
                        }
                    },
                )
                return
            # Metadatos compatibles de la respuesta HTTP: identificador, fecha y alias del modelo.
            identity = {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "created": int(time.time()),
                "model": model_id,
            }
            # Generador del motor: produce texto y un último evento con consumo y motivo de parada.
            events = self.server.engine.events(model_id, messages)
            # El cliente decide entre recibir eventos progresivos o una respuesta JSON al terminar.
            if body.get("stream", False):
                # Obtener el primer fragmento antes de responder 200 permite
                # devolver un error real de inferencia si falla al comenzar.
                # Intenta comenzar la generación antes de enviar HTTP 200 para poder comunicar un fallo inicial con su estado real.
                first = next(events)
                # Envía el código HTTP antes de las cabeceras y del cuerpo.
                self.send_response(200)
                # Añade permisos del navegador solo si corresponde al origen validado.
                self.cors_headers()
                # Describe si el cuerpo es JSON completo o un flujo de eventos SSE.
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                # Evita que las respuestas de la demo se almacenen en caché HTTP.
                self.send_header("Cache-Control", "no-store")
                # Cierra las cabeceras; a partir de aquí se escribe el cuerpo de respuesta.
                self.end_headers()
                # Recuerda que ya empezó SSE: un fallo posterior se comunica dentro del flujo.
                started = True

                # Convierte un evento del motor al formato SSE compatible y lo envía inmediatamente.
                def chunk(event):
                    # Cada evento de texto es un delta; el evento final no añade texto.
                    delta = {"content": event["content"]} if "content" in event else {}
                    # Añade metadatos compatibles y el motivo de parada a cada chunk.
                    data = {
                        **identity,
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": event.get("finish_reason"),
                            }
                        ],
                    }
                    # El último evento aporta consumo; se añade solo cuando el motor lo ha medido.
                    if "usage" in event:
                        # Conserva tokens de entrada/salida del evento final.
                        data["usage"] = event["usage"]
                    # Codifica la línea SSE o el error del flujo y lo escribe al cliente.
                    self.wfile.write(
                        (
                            "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"
                        ).encode()
                    )
                    # Envía inmediatamente bytes pendientes para que el navegador muestre el fragmento.
                    self.wfile.flush()

                # Publica el fragmento inicial que ya demostró que podía empezar la inferencia.
                chunk(first)
                # Consume los fragmentos del motor en orden hasta el evento final de consumo/parada.
                for event in events:
                    # Traduce cada evento posterior al formato SSE compatible.
                    chunk(event)
                # Terminador SSE esperado por clientes compatibles; indica fin del flujo, no un fragmento del modelo.
                self.wfile.write(b"data: [DONE]\n\n")
                # Envía inmediatamente bytes pendientes para que el navegador muestre el fragmento.
                self.wfile.flush()
            else:
                # En modo no streaming se acumula texto y se conserva por separado el último evento de métricas.
                parts, last = [], {}
                # Consume los fragmentos del motor en orden hasta el evento final de consumo/parada.
                for event in events:
                    # Distingue un fragmento de respuesta del evento final con consumo/parada.
                    if "content" in event:
                        # Acumula cada fragmento original sin repararlo para la respuesta JSON final.
                        parts.append(event["content"])
                    else:
                        # Guarda el evento final que identifica motivo de parada y tokens.
                        last = event
                # Devuelve catálogo/estado o texto completo y consumo con su estructura compatible.
                self.send_json(
                    200,
                    {
                        **identity,
                        "object": "chat.completion",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": "".join(parts),
                                },
                                "finish_reason": last["finish_reason"],
                            }
                        ],
                        "usage": last["usage"],
                    },
                )
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            # Tras iniciar SSE, comunica el error dentro del flujo; ya no puede cambiar el estado HTTP.
            if started:
                # Codifica la línea SSE o el error del flujo y lo escribe al cliente.
                self.wfile.write(
                    (
                        "data: " + json.dumps({"error": {"message": str(exc)}}) + "\n\n"
                    ).encode()
                )
                # Envía inmediatamente bytes pendientes para que el navegador muestre el fragmento.
                self.wfile.flush()
            else:
                # Errores de entrada reciben 400; fallos internos reciben 500.
                self.send_json(
                    400 if isinstance(exc, (ValueError, KeyError)) else 500,
                    {"error": {"message": str(exc)}},
                )
        finally:
            # Cierra el generador si llegó a crearse, también al desconectarse el navegador.
            if events is not None:
                # Cierra el generador; su finally cancela/espera al worker y libera el lock.
                events.close()
            # Solo devuelve plaza de inferencia si esta petición realmente la había adquirido.
            if acquired:
                # Devuelve la plaza al semáforo para que una petición posterior pueda utilizarla.
                self.server.inference_slots.release()


# Carga el motor y abre el servidor solo en loopback; Ctrl+C cierra el socket.
def main():
    # Parser de argumentos: describe las opciones que admite este comando.
    parser = argparse.ArgumentParser(description=__doc__)
    # Opción de terminal '--port': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    parser.add_argument("--port", type=int, default=1235)
    # Opción de terminal '--adaptador': el parser aplica tipo, valores permitidos y obligatoriedad definidos aquí.
    parser.add_argument("--adaptador", default=str(ROOT / "modelos/lora-soporte"))
    # Argumentos de la terminal convertidos a atributos como args.salida o args.modelo.
    args = parser.parse_args()
    # Muestra el resultado/estado de este paso en la terminal; las claves no se añaden al informe.
    print("Cargando el mismo Qwen y su adaptador de soporte…", flush=True)
    # Carga modelo y adaptador antes de empezar a escuchar consultas.
    engine = ComparisonEngine(args.adaptador)
    # Escucha solo en IPv4 loopback; no abre la demo en todas las interfaces de red.
    server = ComparisonServer(("127.0.0.1", args.port), engine)
    # Muestra el resultado/estado de este paso en la terminal; las claves no se añaden al informe.
    print(
        json.dumps(
            {
                "url": f"http://localhost:{args.port}/v1",
                "models": MODEL_IDS,
                **engine.metadata,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        # Atiende peticiones hasta que Ctrl+C interrumpe el bucle.
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        # Cierra el recurso abierto; libera socket/conexión aunque el paso anterior haya fallado.
        server.server_close()


# Punto de entrada: ejecutar como script llama main; importar el módulo solo define sus funciones.
if __name__ == "__main__":
    # Invoca el punto de entrada únicamente cuando se ejecuta este archivo como programa.
    main()

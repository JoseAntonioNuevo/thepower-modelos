# GUÍA DE LECTURA · tests/test_comparison_server.py
# Pruebas HTTP reales contra un servidor local con un motor simulado.
# Comprueban modelos, streaming, CORS, Host y saturación de la cola.
# El puerto 0 pide un puerto libre al sistema y evita depender del servidor del profesor.
# Estas pruebas no cargan Qwen ni acreditan la calidad de sus respuestas.

"""Contrato HTTP de las dos condiciones y del streaming usado por isbetter.ai."""

# Convierte entre texto JSON y diccionarios/listas Python, manteniendo respuestas como evidencia.
import json

# Gestiona rutas de importación y el intérprete actual de Python.
import sys

# Hilos, locks y semáforos permiten atender HTTP y coordinar el modelo compartido.
import threading

# Organiza pruebas y aserciones; los tests se ejecutan solo al invocarlos.
import unittest

# Path permite construir rutas, comprobar archivos y leer/escribir sin concatenar separadores a mano.
from pathlib import Path

# Cliente HTTP; las peticiones se realizan al llamarlo, no por el simple import.
import requests

# Añade una ruta de importación para poder reutilizar el paquete curso y los scripts del proyecto.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from servir_comparacion import ALLOWED_ORIGIN, MODEL_IDS, ComparisonServer


# Sustituto del modelo real utilizado por las pruebas HTTP; no necesita pesos ni GPU.
class FakeEngine:
    metadata = {"base_model": "Qwen/Qwen3-0.6B", "enable_thinking": False}

    # Motor ficticio: devuelve el alias solicitado y consumo fijo para probar HTTP sin cargar pesos.
    def events(self, model_id, messages):
        yield {"content": model_id}
        yield {
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        }


# Agrupa pruebas de las rutas HTTP utilizando un servidor local y motor simulado.
class ComparisonHTTP(unittest.TestCase):
    # Arranca una sola instancia del servidor simulado para todas las pruebas de esta clase.
    @classmethod
    def setUpClass(cls):
        cls.server = ComparisonServer(("127.0.0.1", 0), FakeEngine())
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    # Detiene el servidor, cierra el socket y espera a que termine su hilo.
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        # Cierra el recurso abierto; libera socket/conexión aunque el paso anterior haya fallado.
        cls.server.server_close()
        cls.thread.join()

    # Construye un cuerpo mínimo de chat para probar el contrato HTTP y el modo stream.
    def body(self, model, stream=False):
        return {
            "model": model,
            "messages": [{"role": "user", "content": "Incidencia de prueba"}],
            "stream": stream,
        }

    # Comprueba catálogo, activación del adaptador y restricción de orígenes del navegador.
    def test_discovery_and_origin_scope(self):
        # Respuesta HTTP que se inspeccionará y cerrará después de leerla.
        response = requests.get(
            self.url + "/v1/models", headers={"Origin": ALLOWED_ORIGIN}, timeout=2
        )
        # Comprueba igualdad exacta con el resultado esperado para este contrato.
        self.assertEqual([m["id"] for m in response.json()["data"]], list(MODEL_IDS))
        # Comprueba igualdad exacta con el resultado esperado para este contrato.
        self.assertEqual(
            [m["adapter_enabled"] for m in response.json()["data"]], [False, True]
        )
        # Comprueba igualdad exacta con el resultado esperado para este contrato.
        self.assertEqual(
            response.headers["Access-Control-Allow-Origin"], ALLOWED_ORIGIN
        )
        blocked = requests.get(
            self.url + "/v1/models",
            headers={"Origin": "https://otro.example"},
            timeout=2,
        )
        # Comprueba igualdad exacta con el resultado esperado para este contrato.
        self.assertEqual(blocked.status_code, 403)
        # Comprueba que no se añadió un campo/valor ajeno al recorrido permitido.
        self.assertNotIn("Access-Control-Allow-Origin", blocked.headers)

    # Comprueba las dos opciones, reconstrucción del streaming y terminador [DONE].
    def test_both_names_dispatch_and_stream_finishes(self):
        # Prueba las dos condiciones del mismo modelo sin descargar pesos nuevos.
        for model in MODEL_IDS:
            # Respuesta HTTP que se inspeccionará y cerrará después de leerla.
            response = requests.post(
                self.url + "/v1/chat/completions", json=self.body(model), timeout=2
            )
            # Comprueba igualdad exacta con el resultado esperado para este contrato.
            self.assertEqual(response.json()["choices"][0]["message"]["content"], model)
        # Respuesta HTTP que se inspeccionará y cerrará después de leerla.
        response = requests.post(
            self.url + "/v1/chat/completions",
            json=self.body(MODEL_IDS[1], True),
            timeout=2,
        )
        # Generador del motor: produce texto y un último evento con consumo y motivo de parada.
        events = [
            line[6:] for line in response.text.splitlines() if line.startswith("data: ")
        ]
        # Comprueba igualdad exacta con el resultado esperado para este contrato.
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(line) for line in events[:-1]]
        # Comprueba igualdad exacta con el resultado esperado para este contrato.
        self.assertEqual(
            "".join(c["choices"][0]["delta"].get("content", "") for c in chunks),
            MODEL_IDS[1],
        )
        # Comprueba igualdad exacta con el resultado esperado para este contrato.
        self.assertEqual(chunks[-1]["usage"]["total_tokens"], 6)

    # Rechaza modelos o parámetros que cambien el protocolo docente.
    def test_other_models_and_protocol_changes_rejected(self):
        for change in [
            {"model": "qwen3.5-4b-mlx"},
            {"temperature": 1},
            {"max_tokens": 1000},
            {"response_format": {"type": "json_object"}},
        ]:
            # Respuesta HTTP que se inspeccionará y cerrará después de leerla.
            response = requests.post(
                self.url + "/v1/chat/completions",
                json={**self.body(MODEL_IDS[0]), **change},
                timeout=2,
            )
            # Comprueba igualdad exacta con el resultado esperado para este contrato.
            self.assertEqual(response.status_code, 400)

    # Rechaza un Host externo aunque la petición llegue al socket de loopback.
    def test_external_host_is_rejected_even_without_origin(self):
        # Respuesta HTTP que se inspeccionará y cerrará después de leerla.
        response = requests.get(
            self.url + "/v1/models", headers={"Host": "external.example"}, timeout=2
        )
        # Comprueba igualdad exacta con el resultado esperado para este contrato.
        self.assertEqual(response.status_code, 403)

    # Llena la cola, exige 429 y comprueba que vuelve a aceptar tras liberar plazas.
    def test_full_inference_queue_is_rejected_and_recovers(self):
        for _ in range(4):
            # Exige que la propiedad comprobada sea verdadera.
            self.assertTrue(self.server.inference_slots.acquire(blocking=False))
        try:
            # Respuesta HTTP que se inspeccionará y cerrará después de leerla.
            response = requests.post(
                self.url + "/v1/chat/completions",
                json=self.body(MODEL_IDS[0]),
                timeout=2,
            )
            # Comprueba igualdad exacta con el resultado esperado para este contrato.
            self.assertEqual(response.status_code, 429)
        finally:
            for _ in range(4):
                # Devuelve la plaza al semáforo para que una petición posterior pueda utilizarla.
                self.server.inference_slots.release()
        # Respuesta HTTP que se inspeccionará y cerrará después de leerla.
        response = requests.post(
            self.url + "/v1/chat/completions", json=self.body(MODEL_IDS[0]), timeout=2
        )
        # Comprueba igualdad exacta con el resultado esperado para este contrato.
        self.assertEqual(response.status_code, 200)


# Punto de entrada: ejecutar como script llama main; importar el módulo solo define sus funciones.
if __name__ == "__main__":
    unittest.main()

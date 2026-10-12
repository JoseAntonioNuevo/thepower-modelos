# GUÍA DE LECTURA · tests/test_comparison_server.py
# Pruebas HTTP reales contra un servidor local con un motor simulado.
# Comprueban modelos, streaming, CORS, Host y saturación de la cola.
# El puerto 0 pide un puerto libre al sistema y evita depender de un servidor que ya esté en marcha.
# Estas pruebas no cargan Qwen ni acreditan la calidad de sus respuestas.

"""Contrato HTTP de las dos condiciones y del streaming usado por isbetter.ai."""

# Convierte entre texto JSON y diccionarios/listas Python, manteniendo respuestas como evidencia.
import json

# Gestiona rutas de importación y el intérprete actual de Python.
import sys

# Hilo en segundo plano para ejecutar el servidor de prueba.
import threading

# Organiza pruebas y aserciones; los tests se ejecutan solo al invocarlos.
import unittest

# Path permite construir rutas, comprobar archivos y leer/escribir sin concatenar separadores a mano.
from pathlib import Path

# Cliente HTTP; las peticiones se realizan al llamarlo, no por el simple import.
import requests

# Añade una ruta de importación para poder reutilizar el paquete curso y los scripts del proyecto.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
# Constantes y clase de servidor de scripts/servir_comparacion.py.
from servir_comparacion import ALLOWED_ORIGIN, MODEL_IDS, ComparisonServer


# Sustituto del modelo real utilizado por las pruebas HTTP; no necesita pesos ni GPU.
class FakeEngine:
    # Metadatos mínimos que /health devolvería.
    metadata = {"base_model": "Qwen/Qwen3-0.6B", "enable_thinking": False}

    # Motor ficticio: devuelve el alias solicitado y consumo fijo para probar HTTP sin cargar pesos.
    def events(self, model_id, messages):
        # Primer evento: un fragmento de texto (el propio alias, para saber qué condición respondió).
        yield {"content": model_id}
        # Evento final: motivo de parada y consumo de tokens.
        yield {
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        }


# Agrupa pruebas de las rutas HTTP utilizando un servidor local y motor simulado.
class ComparisonHTTP(unittest.TestCase):
    # Arranca una sola instancia del servidor simulado para todas las pruebas de esta clase.
    @classmethod
    def setUpClass(cls):
        # Puerto 0: el sistema elige un puerto libre; FakeEngine evita cargar Qwen.
        cls.server = ComparisonServer(("127.0.0.1", 0), FakeEngine())
        # Hilo en segundo plano que atiende peticiones mientras corren las pruebas.
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        # Arranca el hilo; serve_forever queda escuchando.
        cls.thread.start()
        # URL base con el puerto que eligió el sistema.
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    # Detiene el servidor, cierra el socket y espera a que termine su hilo.
    @classmethod
    def tearDownClass(cls):
        # Para el bucle serve_forever.
        cls.server.shutdown()
        # Cierra el socket del servidor de prueba.
        cls.server.server_close()
        # Espera a que el hilo del servidor termine.
        cls.thread.join()

    # Construye un cuerpo mínimo de chat para probar el contrato HTTP y el modo stream.
    def body(self, model, stream=False):
        # Cuerpo de chat mínimo con un único mensaje de usuario.
        return {
            "model": model,
            "messages": [{"role": "user", "content": "Incidencia de prueba"}],
            "stream": stream,
        }

    # Comprueba catálogo, activación del adaptador y restricción de orígenes del navegador.
    def test_discovery_and_origin_scope(self):
        # Petición real al servidor local de prueba, con el Origin de isbetter.ai.
        response = requests.get(
            self.url + "/v1/models", headers={"Origin": ALLOWED_ORIGIN}, timeout=2
        )
        # El catálogo ofrece exactamente los dos alias, en orden.
        self.assertEqual([m["id"] for m in response.json()["data"]], list(MODEL_IDS))
        # Solo qwen-lora-soporte tiene el adaptador activado.
        self.assertEqual(
            [m["adapter_enabled"] for m in response.json()["data"]], [False, True]
        )
        # La respuesta autoriza al navegador de isbetter.ai (CORS).
        self.assertEqual(
            response.headers["Access-Control-Allow-Origin"], ALLOWED_ORIGIN
        )
        # Mismo endpoint desde un origen no autorizado.
        blocked = requests.get(
            self.url + "/v1/models",
            headers={"Origin": "https://otro.example"},
            timeout=2,
        )
        # Otro origen recibe 403 (prohibido).
        self.assertEqual(blocked.status_code, 403)
        # Y no recibe la cabecera que autorizaría al navegador.
        self.assertNotIn("Access-Control-Allow-Origin", blocked.headers)

    # Comprueba las dos opciones, reconstrucción del streaming y terminador [DONE].
    def test_both_names_dispatch_and_stream_finishes(self):
        # Prueba las dos condiciones del mismo modelo sin descargar pesos nuevos.
        for model in MODEL_IDS:
            # Petición sin streaming para cada alias.
            response = requests.post(
                self.url + "/v1/chat/completions", json=self.body(model), timeout=2
            )
            # El motor simulado responde con el alias: cada nombre llegó a su condición.
            self.assertEqual(response.json()["choices"][0]["message"]["content"], model)
        # Misma consulta en modo streaming (SSE) con el alias del adaptador.
        response = requests.post(
            self.url + "/v1/chat/completions",
            json=self.body(MODEL_IDS[1], True),
            timeout=2,
        )
        # Contenido de cada línea SSE «data: …» de la respuesta en streaming.
        events = [
            line[6:] for line in response.text.splitlines() if line.startswith("data: ")
        ]
        # El flujo termina con el marcador [DONE].
        self.assertEqual(events[-1], "[DONE]")
        # Cada línea data: (salvo [DONE]) es un fragmento JSON.
        chunks = [json.loads(line) for line in events[:-1]]
        # Unir los fragmentos reconstruye el texto completo (aquí, el alias pedido).
        self.assertEqual(
            "".join(c["choices"][0]["delta"].get("content", "") for c in chunks),
            MODEL_IDS[1],
        )
        # El último fragmento trae el consumo de tokens.
        self.assertEqual(chunks[-1]["usage"]["total_tokens"], 6)

    # Rechaza modelos o parámetros que cambien el protocolo docente.
    def test_other_models_and_protocol_changes_rejected(self):
        # Cada diccionario cambia un aspecto de la petición válida:
        for change in [
            # un modelo que el servidor no ofrece,
            {"model": "qwen3.5-4b-mlx"},
            # una temperatura distinta de 0,
            {"temperature": 1},
            # un límite de salida distinto de 180
            {"max_tokens": 1000},
            # o un formato JSON impuesto (JSON mode).
            {"response_format": {"type": "json_object"}},
        ]:
            # Petición válida con un único cambio prohibido.
            response = requests.post(
                self.url + "/v1/chat/completions",
                json={**self.body(MODEL_IDS[0]), **change},
                timeout=2,
            )
            # Cada cambio se rechaza con 400 (petición incorrecta).
            self.assertEqual(response.status_code, 400)

    # Rechaza un Host externo aunque la petición llegue al socket de loopback.
    def test_external_host_is_rejected_even_without_origin(self):
        # Host externo aunque la conexión llegue a 127.0.0.1.
        response = requests.get(
            self.url + "/v1/models", headers={"Host": "external.example"}, timeout=2
        )
        # Se rechaza con 403 aunque no haya Origin.
        self.assertEqual(response.status_code, 403)

    # Llena la cola, exige 429 y comprueba que vuelve a aceptar tras liberar plazas.
    def test_full_inference_queue_is_rejected_and_recovers(self):
        # Ocupa las cuatro plazas de inferencia como si hubiera cuatro consultas en curso.
        for _ in range(4):
            # acquire(blocking=False) devuelve True si había plaza libre.
            self.assertTrue(self.server.inference_slots.acquire(blocking=False))
        # try/finally: las plazas se devuelven aunque la prueba falle.
        try:
            # Petición con la cola de inferencia llena.
            response = requests.post(
                self.url + "/v1/chat/completions",
                json=self.body(MODEL_IDS[0]),
                timeout=2,
            )
            # 429: demasiadas peticiones; el servidor no encola más trabajo.
            self.assertEqual(response.status_code, 429)
        finally:
            # Libera las cuatro plazas ocupadas.
            for _ in range(4):
                # Devuelve la plaza al semáforo para que una petición posterior pueda utilizarla.
                self.server.inference_slots.release()
        # Con las plazas liberadas, la misma petición vuelve a funcionar.
        response = requests.post(
            self.url + "/v1/chat/completions", json=self.body(MODEL_IDS[0]), timeout=2
        )
        # 200: respuesta correcta.
        self.assertEqual(response.status_code, 200)


# Punto de entrada: ejecutar como script llama main; importar el módulo solo define sus funciones.
if __name__ == "__main__":
    # Ejecuta las pruebas de este archivo.
    unittest.main()

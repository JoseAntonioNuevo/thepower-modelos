# GUÍA DE LECTURA · tests/test_proveedores.py
# Contratos de la capa de proveedores de curso/remote.py.
# Comprueban cómo se elige la ruta desde .env y qué petición construye cada adaptador:
# Anthropic (SDK oficial), OpenAI, OpenRouter, APIs compatibles y Gemini API.
# Mocks sustituyen SDK y red: ninguna prueba llama a un proveedor ni usa una clave real.

# Convierte entre texto JSON y diccionarios/listas Python, manteniendo respuestas como evidencia.
import json

# Gestiona rutas de importación y el intérprete actual de Python.
import sys

# Crea directorios temporales de prueba que se retiran al salir del contexto.
import tempfile

# Organiza pruebas y aserciones; los tests se ejecutan solo al invocarlos.
import unittest

# Buffers de memoria permiten descartar la salida de los scripts dentro de una prueba.
import io

# Captura stdout en memoria para que los scripts no ensucien la salida de las pruebas.
from contextlib import redirect_stdout

# Path permite construir rutas, comprobar archivos y leer/escribir sin concatenar separadores a mano.
from pathlib import Path

# Objetos simples con atributos, como los que devuelve el SDK de Anthropic.
from types import SimpleNamespace

# Mocks sustituyen red/configuración para probar contratos sin proveedores reales.
from unittest.mock import Mock, patch

# Errores tipados del SDK y objetos HTTP con los que se construyen.
import anthropic
import httpx2

# Añade una ruta de importación para poder reutilizar el paquete curso y los scripts del proyecto.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
# Reutiliza funciones del núcleo compartido para mantener el mismo protocolo entre scripts.
from curso.common import MAX_NEW_TOKENS, basic_prompt
from curso.remote import chat, route
from evaluar_remotos import main as evaluate_main

# Respuesta JSON válida que devuelven los proveedores simulados.
RAW = '{"categoria": "facturacion", "prioridad": "media", "respuesta": "Indica la fecha del pago."}'


# Fabrica una respuesta HTTP simulada con el cuerpo JSON indicado.
def response(data, status=200):
    # Respuesta simulada cuyo cuerpo/estado controla la prueba para reproducir éxito o error.
    value = Mock(status_code=status, headers={})
    value.iter_content.return_value = [json.dumps(data).encode()]
    return value


# Entorno aislado: sin .env real y solo con las variables que indica cada prueba.
def environment(**values):
    return (
        patch("curso.remote.load_config"),
        patch.dict("os.environ", values, clear=True),
    )


# Mensaje con la forma que devuelve client.messages.create del SDK de Anthropic.
def anthropic_message(blocks):
    return SimpleNamespace(
        content=blocks,
        usage=SimpleNamespace(input_tokens=1000, output_tokens=40),
        model="claude-haiku-5-5",
        id="msg_demo",
        _request_id="req_demo",
        stop_reason="end_turn",
    )


# Agrupa los contratos de la capa de proveedores.
class Providers(unittest.TestCase):
    # Sin .env se usa la ruta del curso; LLM_MODEL solo completa el proveedor de LLM_PROVIDER.
    def test_route_from_env_and_arguments(self):
        load, env = environment()
        with load, env:
            self.assertEqual(route(), ("anthropic", "claude-haiku-5-5"))
            self.assertEqual(route("local"), ("local", None))
            # Validación previa: otro proveedor sin modelo no hereda el ID de la ruta del curso.
            with self.assertRaises(ValueError):
                route("openai")
            with self.assertRaises(ValueError):
                route("desconocido", "x")
        load, env = environment(LLM_PROVIDER="OpenAI", LLM_MODEL="gpt-demo")
        with load, env:
            self.assertEqual(route(), ("openai", "gpt-demo"))
            self.assertEqual(route(None, "otro"), ("openai", "otro"))
            # Validación previa: la ruta del curso necesita --modelo si .env apunta a otra.
            with self.assertRaises(ValueError):
                route("anthropic")

    # Claude por la Messages API: system aparte, sin temperatura, sin thinking y sin reintentos.
    def test_anthropic_request_record_and_cost(self):
        load, env = environment(
            ANTHROPIC_API_KEY="dummy",
            LLM_PRECIO_ENTRADA_USD_MTOK="0.10",
            LLM_PRECIO_SALIDA_USD_MTOK="0.50",
        )
        # Cliente simulado: devuelve un bloque thinking vacío seguido del texto de la respuesta.
        client = Mock()
        client.messages.create.return_value = anthropic_message(
            [
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text=RAW),
            ]
        )
        with load, env, patch("anthropic.Anthropic", return_value=client) as factory:
            result = chat("incidencia", fewshot=True)
        # El cliente se crea sin reintentos automáticos y se cierra al terminar.
        self.assertEqual(factory.call_args.kwargs["max_retries"], 0)
        client.close.assert_called_once()
        sent = client.messages.create.call_args.kwargs
        # Petición: modelo del curso, 180 tokens, instrucciones en system y conversación few-shot.
        self.assertEqual(sent["model"], "claude-haiku-5-5")
        self.assertEqual(sent["max_tokens"], MAX_NEW_TOKENS)
        self.assertEqual(sent["system"], basic_prompt())
        self.assertEqual(len(sent["messages"]), 9)
        self.assertEqual(sent["messages"][0]["role"], "user")
        self.assertEqual(
            sent["messages"][-1], {"role": "user", "content": "incidencia"}
        )
        self.assertEqual(sent["thinking"], {"type": "disabled"})
        # Haiku 5.5 rechaza temperaturas no predeterminadas: no se envía ninguna.
        self.assertNotIn("temperature", sent)
        self.assertIsNone(sent["extra_body"])
        # Registro: solo el texto, tokens, coste por tarifa de .env e identificadores.
        self.assertEqual(result["respuesta_original"], RAW)
        self.assertEqual(
            (result["tokens_entrada"], result["tokens_salida"]), (1000, 40)
        )
        self.assertAlmostEqual(result["coste_usd"], (1000 * 0.10 + 40 * 0.50) / 1e6)
        self.assertEqual(result["request_id"], "req_demo")
        self.assertEqual(result["finish_reason"], "end_turn")
        self.assertEqual(result["ruta"], "anthropic")
        self.assertNotIn("system", result["parametros_solicitados"])

    # Sin clave no se crea cliente; un error HTTP se registra sin reintentar.
    def test_anthropic_missing_key_and_status_error(self):
        load, env = environment()
        with load, env, patch("anthropic.Anthropic") as factory:
            with self.assertRaises(ValueError):
                chat("incidencia")
            factory.assert_not_called()
        # Error 429 del servicio, construido como lo haría el SDK.
        request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        error = anthropic.RateLimitError(
            "límite", response=httpx2.Response(429, request=request), body=None
        )
        client = Mock()
        client.messages.create.side_effect = error
        load, env = environment(ANTHROPIC_API_KEY="dummy")
        with load, env, patch("anthropic.Anthropic", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "429"):
                chat("incidencia")
        client.messages.create.assert_called_once()
        client.close.assert_called_once()

    # .env puede cambiar temperatura, thinking y añadir campos del proveedor configurado.
    def test_env_overrides_for_configured_route(self):
        load, env = environment(
            ANTHROPIC_API_KEY="dummy",
            LLM_TEMPERATURE="1",
            LLM_EXTRA_BODY='{"thinking": null, "output_config": {"effort": "low"}}',
        )
        client = Mock()
        client.messages.create.return_value = anthropic_message(
            [SimpleNamespace(type="text", text=RAW)]
        )
        with load, env, patch("anthropic.Anthropic", return_value=client):
            chat("incidencia")
        sent = client.messages.create.call_args.kwargs
        self.assertNotIn("thinking", sent)
        self.assertEqual(sent["temperature"], 1.0)
        self.assertEqual(sent["extra_body"], {"output_config": {"effort": "low"}})
        # Validación previa: campos protegidos o JSON que no es un objeto se rechazan.
        for value in ('{"model": "otro"}', "[1]"):
            load, env = environment(ANTHROPIC_API_KEY="dummy", LLM_EXTRA_BODY=value)
            with load, env, patch("anthropic.Anthropic", return_value=client):
                with self.assertRaises(ValueError):
                    chat("incidencia")

    # OpenAI, APIs compatibles y OpenRouter comparten Chat Completions con nombres propios.
    def test_chat_completions_routes(self):
        envelope = {
            "id": "gen-demo",
            "model": "demo",
            "choices": [{"message": {"content": RAW}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 30},
        }
        load, env = environment(OPENAI_API_KEY="dummy")
        with load, env, patch("requests.post", return_value=response(envelope)) as post:
            result = chat("incidencia", "openai", "gpt-demo")
        sent = post.call_args.kwargs
        # OpenAI: max_completion_tokens, sin temperatura y con autenticación Bearer.
        self.assertEqual(
            post.call_args.args[0], "https://api.openai.com/v1/chat/completions"
        )
        self.assertEqual(sent["json"]["max_completion_tokens"], MAX_NEW_TOKENS)
        self.assertNotIn("max_tokens", sent["json"])
        self.assertNotIn("temperature", sent["json"])
        self.assertEqual(sent["headers"]["Authorization"], "Bearer dummy")
        self.assertTrue(sent["stream"])
        self.assertFalse(sent["allow_redirects"])
        self.assertEqual(result["respuesta_original"], RAW)
        self.assertIsNone(result["coste_usd"])
        # Servidor compatible en el propio equipo: sin clave y con temperatura 0.
        load, env = environment(COMPAT_BASE_URL="http://localhost:11434/v1")
        with load, env, patch("requests.post", return_value=response(envelope)) as post:
            chat("incidencia", "compatible", "llama-demo")
        self.assertEqual(
            post.call_args.args[0], "http://localhost:11434/v1/chat/completions"
        )
        self.assertNotIn("Authorization", post.call_args.kwargs["headers"])
        self.assertEqual(post.call_args.kwargs["json"]["temperature"], 0)
        # Validación previa: sin URL, o con http hacia un servidor remoto, no se envía nada.
        for url in ("", "http://proveedor.example/v1"):
            load, env = environment(COMPAT_BASE_URL=url)
            with load, env, patch("requests.post") as post:
                with self.assertRaises(ValueError):
                    chat("incidencia", "compatible", "x")
                post.assert_not_called()
        # OpenRouter conserva usage.cost como coste informado por la ruta.
        priced = dict(envelope, usage={**envelope["usage"], "cost": 0.0002})
        load, env = environment(OPENROUTER_API_KEY="dummy")
        with load, env, patch("requests.post", return_value=response(priced)) as post:
            result = chat("incidencia", "openrouter", "proveedor/modelo")
        self.assertEqual(post.call_args.kwargs["json"]["usage"], {"include": True})
        self.assertEqual(result["coste_usd"], 0.0002)

    # Gemini API con clave: contents user/model, thinking mínimo y bloques thought excluidos.
    def test_gemini_api_payload_and_reply(self):
        data = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "pienso", "thought": True}, {"text": RAW}]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 300, "candidatesTokenCount": 25},
            "responseId": "resp-demo",
        }
        load, env = environment(GEMINI_API_KEY="dummy")
        with load, env, patch("requests.post", return_value=response(data)) as post:
            result = chat("incidencia", "gemini", "models/gemini-3.1-flash-lite", True)
        sent = post.call_args.kwargs
        self.assertTrue(
            post.call_args.args[0].endswith(
                "/models/gemini-3.1-flash-lite:generateContent"
            )
        )
        self.assertEqual(sent["headers"]["x-goog-api-key"], "dummy")
        roles = [m["role"] for m in sent["json"]["contents"]]
        self.assertEqual(roles, ["user", "model"] * 4 + ["user"])
        self.assertEqual(
            sent["json"]["generationConfig"]["thinkingConfig"],
            {"thinkingLevel": "MINIMAL"},
        )
        self.assertEqual(result["respuesta_original"], RAW)
        self.assertEqual((result["tokens_entrada"], result["tokens_salida"]), (300, 25))
        self.assertEqual(result["finish_reason"], "STOP")
        # La ruta Vertex del ejercicio no admite few-shot.
        load, env = environment(GOOGLE_CLOUD_PROJECT="demo")
        with load, env, self.assertRaises(ValueError):
            chat("incidencia", "google", "gemini-3.1-flash-lite", True)

    # La evaluación usa la ruta de .env para claude_prompt y fija la ruta de Gemini.
    def test_remote_evaluation_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "claude_prompt.jsonl"
            # Sin clave, el lote se detiene antes de llamar ni escribir resultados.
            load, env = environment()
            with (
                load,
                env,
                patch(
                    "sys.argv",
                    ["evaluar", "--condicion", "claude_prompt", "--salida", str(path)],
                ),
                patch("evaluar_remotos.chat") as call,
                patch("sys.stderr", io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                evaluate_main()
            call.assert_not_called()
            self.assertFalse(path.exists())
            load, env = environment(ANTHROPIC_API_KEY="dummy")
            with (
                load,
                env,
                patch(
                    "sys.argv",
                    [
                        "evaluar",
                        "--condicion",
                        "claude_prompt",
                        "--split",
                        "desarrollo",
                        "--salida",
                        str(path),
                    ],
                ),
                patch(
                    "evaluar_remotos.chat",
                    return_value={
                        "respuesta_original": RAW,
                        "latencia_s": 1,
                        "coste_usd": None,
                    },
                ) as call,
                redirect_stdout(io.StringIO()),
            ):
                evaluate_main()
            # Seis casos de desarrollo con Claude Haiku 5.5 por Anthropic y few-shot.
            self.assertEqual(call.call_count, 6)
            self.assertEqual(
                call.call_args.args[1:], ("anthropic", "claude-haiku-5-5", True)
            )
        # Gemini del ejercicio no cambia de proveedor.
        load, env = environment()
        with (
            load,
            env,
            patch(
                "sys.argv",
                [
                    "evaluar",
                    "--condicion",
                    "gemini_base",
                    "--proveedor",
                    "anthropic",
                    "--salida",
                    "x.jsonl",
                ],
            ),
            redirect_stdout(io.StringIO()),
            patch("sys.stderr", io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            evaluate_main()


# Permite ejecutar este archivo directamente además de unittest discover.
if __name__ == "__main__":
    unittest.main()

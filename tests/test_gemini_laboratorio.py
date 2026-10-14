"""Contratos del laboratorio público: proveedor simulado, ninguna llamada Cloud."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from curso import gemini_laboratorio as lab
from curso.common import read_jsonl
from curso.negocio import DATA, GENERATION, dataset_check, grade, mentions, sha


class LaboratoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        quiet = patch("builtins.print")
        quiet.start()
        self.addCleanup(quiet.stop)
        self.run = Path(self.temp.name) / "laboratorio"
        self.cfg = lab.prepare(self.run, "mi-proyecto-clase", "mi-bucket-clase", "nebula-v1", 5)
        self.number = "123456789012"

    def seed_budget_upload(self):
        lab.save_new(self.run / "presupuesto.json", {"total_estimated_usd": 4,
                                                     "config_sha256": sha(self.run / "config.json")})
        lab.save_new(self.run / "subida.json", {"project_number": self.number,
                                               "config_sha256": sha(self.run / "config.json")})

    def job(self, state="JOB_STATE_SUCCEEDED"):
        return {"name": f"projects/{self.number}/locations/europe-west4/tuningJobs/789",
                "state": state, "base_model": lab.BASE,
                "supervised_tuning_spec": {
                    "training_dataset_uri": "gs://mi-bucket-clase/nebula-v1/train.jsonl",
                    "validation_dataset_uri": "gs://mi-bucket-clase/nebula-v1/validacion.jsonl",
                    "hyper_parameters": {"epoch_count": 8}},
                "tuned_model": {"checkpoints": [
                    {"checkpoint_id": str(n), "step": n * 10, "epoch": n,
                     "endpoint": f"projects/{self.number}/locations/eu/endpoints/{1000+n}"}
                    for n in (1, 4, 8)]}}

    def seed_finished(self):
        self.seed_budget_upload()
        lab.save_new(self.run / "job-creado.json", self.job())
        lab.save_new(self.run / "estado-final.json", self.job())

    def test_dataset_and_conversion_are_complete(self):
        result = dataset_check()
        self.assertEqual(result["splits"], {"train": 328, "validacion": 37, "test": 78,
                                           "test-precios": 48, "desarrollo": 2})

    def test_prepare_ignores_global_project_and_preserves_previous_run(self):
        self.assertEqual(self.cfg["project"], "mi-proyecto-clase")
        with patch.dict("os.environ", {"GOOGLE_CLOUD_PROJECT": "otro-proyecto"}):
            self.assertEqual(lab.load_run(self.run)["project"], "mi-proyecto-clase")
        with self.assertRaises(FileExistsError):
            lab.prepare(self.run, "mi-proyecto-clase", "mi-bucket-clase", "nebula-v1", 5)
        for value in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                lab.positive(value, "Presupuesto")

    def test_credentials_use_explicit_quota_without_mutating_global_adc(self):
        creds = Mock()
        with patch("google.auth.default", return_value=(creds, "otro-proyecto")):
            self.assertIs(lab.credentials(self.cfg), creds.with_quota_project.return_value)
        creds.with_quota_project.assert_called_once_with("mi-proyecto-clase")

    def test_sdk_has_one_attempt_and_explicit_project_credentials(self):
        creds = Mock()
        with patch.object(lab, "credentials", return_value=creds), patch("google.genai.Client") as constructor:
            lab.vertex_client(self.cfg)
        options = constructor.call_args.kwargs
        self.assertEqual(options["project"], "mi-proyecto-clase")
        self.assertIs(options["credentials"], creds)
        self.assertEqual(options["http_options"].retry_options.attempts, 1)

    def test_cost_margin_and_budget_stop_before_mutation(self):
        predictor = Mock()
        predictor.request.return_value = {"totalTokens": 133735}
        with patch.object(lab, "Predictor", return_value=predictor):
            value = lab.estimate(self.run, self.cfg)
        self.assertEqual(value["estimated_tokens_per_epoch_with_margin"], 156949)
        self.assertAlmostEqual(value["total_estimated_usd"], 4.266776)
        predictor.close.assert_called_once()
        self.cfg["budget_usd"] = 1
        with self.assertRaisesRegex(ValueError, "fuera del presupuesto"):
            lab.check_budget(self.run, self.cfg)

    def test_mutations_and_inference_require_authorization(self):
        with patch.object(lab, "Predictor") as predictor, patch.object(lab, "vertex_client") as client:
            for call in (lambda: lab.upload(self.run, self.cfg), lambda: lab.create(self.run, self.cfg),
                         lambda: lab.evaluate(self.run, self.cfg, "base_sin_catalogo", "test"),
                         lambda: lab.query(self.run, self.cfg, "base_sin_catalogo", "Precio")):
                with self.assertRaisesRegex(ValueError, "autorizar-gasto"):
                    call()
            predictor.assert_not_called()
            client.assert_not_called()

    def test_upload_rejects_bucket_of_other_project_before_writing(self):
        self.seed_budget_upload()
        predictor = Mock()
        predictor.project_number.return_value = self.number
        storage = Mock()
        storage.get_bucket.return_value.project_number = "999999999999"
        with patch.object(lab, "Predictor", return_value=predictor), patch.object(lab, "credentials"), \
                patch("google.cloud.storage.Client", return_value=storage):
            with self.assertRaisesRegex(ValueError, "no pertenece"):
                lab.upload(self.run, self.cfg, True)
        storage.get_bucket.return_value.blob.assert_not_called()
        storage.close.assert_called_once()

    def test_upload_reuses_matching_objects_but_never_sends_test(self):
        # Hace posible recuperar una subida parcial sin sobrescribir un objeto.
        lab.save_new(self.run / "presupuesto.json", {"total_estimated_usd": 4,
                                                     "config_sha256": sha(self.run / "config.json")})
        predictor, storage, bucket = Mock(), Mock(), Mock()
        predictor.project_number.return_value = self.number
        storage.get_bucket.return_value = bucket
        bucket.name = "mi-bucket-clase"
        bucket.project_number = self.number
        bucket.iam_configuration.uniform_bucket_level_access_enabled = True
        bucket.iam_configuration.public_access_prevention = "enforced"
        storage.list_blobs.return_value = []
        blobs = {}
        for split in ("train", "validacion"):
            blob = Mock()
            blob.name = f"nebula-v1/{split}.jsonl"
            blob.exists.return_value = split == "train"
            blob.download_as_bytes.return_value = (DATA / "gemini" / f"{split}.jsonl").read_bytes()
            blob.generation = "1"
            blobs[blob.name] = blob
        bucket.blob.side_effect = lambda name: blobs[name]
        with patch.object(lab, "Predictor", return_value=predictor), patch.object(lab, "credentials"), \
                patch("google.cloud.storage.Client", return_value=storage):
            receipt = lab.upload(self.run, self.cfg, True)
        self.assertFalse(receipt["test_uploaded"])
        blobs["nebula-v1/train.jsonl"].upload_from_filename.assert_not_called()
        call = blobs["nebula-v1/validacion.jsonl"].upload_from_filename.call_args
        self.assertEqual(call.kwargs["if_generation_match"], 0)
        self.assertEqual({r["name"] for r in receipt["objects"]}, set(blobs))

    def test_create_records_pending_request_and_does_not_retry_unknown_outcome(self):
        self.seed_budget_upload()
        client = Mock()
        client.tunings.tune.side_effect = TimeoutError("Respuesta desconocida")
        with patch.object(lab, "vertex_client", return_value=client):
            with self.assertRaises(TimeoutError):
                lab.create(self.run, self.cfg, True)
            with self.assertRaises(FileExistsError):
                lab.create(self.run, self.cfg, True)
        client.tunings.tune.assert_called_once()
        self.assertTrue((self.run / "creacion-solicitada.json").exists())
        self.assertFalse((self.run / "job-creado.json").exists())
        client.close.assert_called_once()

    def test_create_success_uses_own_objects_and_typed_sdk_config(self):
        self.seed_budget_upload()
        client, job = Mock(), Mock()
        job.model_dump_json.return_value = json.dumps(self.job("JOB_STATE_RUNNING"))
        client.tunings.tune.return_value = job
        with patch.object(lab, "vertex_client", return_value=client):
            result = lab.create(self.run, self.cfg, True)
        self.assertEqual(result["name"], self.job()["name"])
        request = client.tunings.tune.call_args.kwargs
        self.assertEqual(request["training_dataset"].gcs_uri, "gs://mi-bucket-clase/nebula-v1/train.jsonl")
        self.assertEqual(request["config"].epoch_count, 8)
        self.assertFalse(request["config"].export_last_checkpoint_only)
        self.assertTrue((self.run / "job-creado.json").exists())
        client.close.assert_called_once()

    def test_checkpoint_identifier_cannot_escape_results_directory(self):
        self.seed_finished()
        job = self.job()
        job["tuned_model"]["checkpoints"][0]["checkpoint_id"] = "../../otra-ruta"
        (self.run / "estado-final.json").write_text(json.dumps(job))
        with self.assertRaisesRegex(ValueError, "ID de checkpoint"):
            lab.candidates(self.run, self.cfg)

    def test_recover_validates_account_and_dataset(self):
        self.seed_budget_upload()
        lab.save_new(self.run / "creacion-solicitada.json", {})
        client, job = Mock(), Mock()
        job.model_dump_json.return_value = json.dumps(self.job())
        client.tunings.get.return_value = job
        with patch.object(lab, "vertex_client", return_value=client):
            with self.assertRaisesRegex(ValueError, "otro proyecto"):
                lab.status(self.run, self.cfg, "projects/999/locations/europe-west4/tuningJobs/789")
            value = lab.status(self.run, self.cfg, self.job()["name"])
        self.assertEqual(value["state"], "JOB_STATE_SUCCEEDED")
        self.assertTrue((self.run / "job-creado.json").exists())
        self.assertTrue((self.run / "estado-final.json").exists())
        wrong = self.job()
        wrong["supervised_tuning_spec"]["training_dataset_uri"] = "gs://mi-bucket-clase/otro/train.jsonl"
        with self.assertRaisesRegex(ValueError, "protocolo"):
            lab.validate_job(wrong, self.cfg, {"project_number": self.number})

    def test_endpoint_must_belong_to_own_job_project_and_multiregion(self):
        self.seed_finished()
        for wrong in ("projects/999/locations/eu/endpoints/1008",
                      f"projects/{self.number}/locations/europe-west4/endpoints/1008",
                      "https://example.com/endpoint"):
            with self.assertRaises(ValueError):
                lab.endpoint_check(wrong, self.cfg, self.run)

    def test_selection_and_full_comparison_work_without_network(self):
        self.seed_finished()
        candidates = lab.candidates(self.run, self.cfg)["candidates"]
        all_rows = {r["pregunta"]: r for split in ("validacion", "test-precios")
                    for r in read_jsonl(DATA / f"{split}.jsonl")}
        predictor = Mock()
        predictor.generate.side_effect = lambda q, model, context: {
            "respuesta_original": json.dumps(all_rows[q]["objetivo"], ensure_ascii=False),
            "tokens_entrada": 100, "tokens_salida": 40, "parametros_solicitados": GENERATION}
        with patch.object(lab, "Predictor", return_value=predictor):
            for cp in candidates:
                lab.evaluate(self.run, self.cfg, "ajustado_sin_catalogo", "validacion", cp["checkpoint_id"], True)
            selected = lab.choose(self.run, self.cfg)
            # Empate: el posterior, sin consultar el test para elegirlo.
            self.assertEqual(selected["checkpoint_id"], "8")
            for condition in lab.CONDITIONS:
                result = lab.evaluate(self.run, self.cfg, condition, "test-precios", allow=True)
                self.assertEqual(result["caso_completo"], 48)
            with self.assertRaises(FileExistsError):
                lab.evaluate(self.run, self.cfg, "base_sin_catalogo", "test-precios", allow=True)
            with self.assertRaisesRegex(ValueError, "Test ya consultado"):
                lab.choose(self.run, self.cfg)
        comparison = lab.compare(self.run, self.cfg, "test-precios")
        self.assertEqual(comparison["gain_percentage_points"], 0)

    def test_api_errors_count_and_preserve_partial_evaluation(self):
        self.seed_finished()
        lab.save_new(self.run / "seleccion.json", {"endpoint": f"projects/{self.number}/locations/eu/endpoints/1008",
                                                 "config_sha256": sha(self.run / "config.json"),
                                                 "selected_at_utc": lab.now()})
        predictor = Mock()
        predictor.generate.side_effect = RuntimeError("secreto-que-no-debe-guardarse")
        with patch.object(lab, "Predictor", return_value=predictor):
            result = lab.evaluate(self.run, self.cfg, "ajustado_sin_catalogo", "test-precios", allow=True)
        self.assertEqual(result["casos"], 48)
        self.assertEqual(result["errores_api"], 48)
        path = lab.result_path(self.run, "ajustado_sin_catalogo", "test-precios")
        self.assertNotIn("secreto-que-no-debe-guardarse", path.read_text())
        self.assertEqual(result["tokens_entrada"], None)

    def test_test_cannot_open_before_selection(self):
        with patch.object(lab, "Predictor") as predictor:
            with self.assertRaisesRegex(ValueError, "Selecciona"):
                lab.evaluate(self.run, self.cfg, "base_sin_catalogo", "test", allow=True)
        predictor.assert_not_called()

    def test_evaluator_rejects_wrong_price_sku_markdown_and_duplicates(self):
        row = read_jsonl(DATA / "test-precios.jsonl")[35]  # Cuaderno Línea, 9 €.
        good = dict(row["objetivo"])
        self.assertTrue(grade(json.dumps(good), row)["caso_completo"])
        for price in (19, 90, "9,90"):
            wrong = {**good, "respuesta": f"Cuaderno Línea cuesta {price} euros"}
            self.assertFalse(grade(json.dumps(wrong), row)["caso_completo"])
        self.assertFalse(grade("```json\n" + json.dumps(good) + "\n```", row)["caso_completo"])
        self.assertFalse(grade(json.dumps({**good, "productos": ["NH-C02", "NH-C02"]}), row)["esquema_valido"])
        self.assertTrue(mentions("Precio 45,00 €", "45"))
        self.assertFalse(mentions("Precio 145 €", "45"))


if __name__ == "__main__":
    unittest.main()

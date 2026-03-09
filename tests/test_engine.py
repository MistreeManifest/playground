from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib import request

from threshold_memory.engine import ThresholdMemoryEngine
from threshold_memory.server import ThresholdHTTPServer


class ThresholdMemoryEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = Path(self.temp_dir.name) / "memory.sqlite3"
        self.engine = ThresholdMemoryEngine(self.db_path)
        self.engine.initialize()
        self.addCleanup(self.engine.close)

    def write_file(self, name: str, content: str) -> Path:
        path = Path(self.temp_dir.name) / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_seed_document_extracts_terms_from_map(self) -> None:
        map_path = self.write_file(
            "TEST-MAP.md",
            "\n".join(
                [
                    "TEST-MAP",
                    "MASTER TERM INDEX (alphabetical)",
                    "Term\tSection in clean.md\tOne-line definition",
                    "Signal\tA1\tMeaningful information",
                    "Noise\tA2\tDistracting entropy",
                    "CONCEPT GRAPH",
                ]
            ),
        )

        seeded = self.engine.seed_document(map_path, domain="opte")
        self.assertEqual(seeded["term_count"], 2)
        status = self.engine.status()
        self.assertEqual(status["canon_documents"], 1)
        self.assertEqual(status["canon_terms"], 2)

    def test_checkpoint_and_collapse_flow(self) -> None:
        checkpoint = self.engine.save_checkpoint(
            task_key="memory-agent",
            intent="Preserve state before permission wait.",
            active_hypothesis="Checkpointing will reduce drift.",
            next_step="Ask for approval and resume from stored cue.",
            resume_cue="Resume from permission boundary with saved intent.",
            risk_flags=["state-loss"],
            state_snapshot={"step": 3},
        )
        pending = self.engine.next_resume()
        self.assertEqual(pending["checkpoint_id"], checkpoint["checkpoint_id"])

        collapse = self.engine.record_collapse(
            boundary_exceeded="approval gate",
            symptoms="lost next action after waiting",
            checkpoint_id=checkpoint["checkpoint_id"],
        )
        self.assertIn("Acknowledge the collapse", collapse["recovery_protocol"])
        self.assertIsNone(self.engine.next_resume())

    def test_query_returns_checkpoint_and_glyph(self) -> None:
        self.engine.log_episode(
            title="Permission drift",
            content="The agent lost its place after a long approval wait and needed reintegration.",
            phase="break",
            delta_coherence=2.0,
            effort=1.0,
        )
        self.engine.log_episode(
            title="Reintegration pass",
            content="A tighter resume cue restored the thread with lower effort.",
            phase="reintegration",
            delta_coherence=3.0,
            effort=1.0,
        )
        glyph = self.engine.consolidate(limit=2)
        self.assertIsNotNone(glyph)

        self.engine.save_checkpoint(
            task_key="approval-loop",
            intent="Hold state through waiting.",
            active_hypothesis="Resume cues should be explicit.",
            next_step="Continue from checkpoint.",
            resume_cue="Return to the stored next step.",
        )

        results = self.engine.query("resume permission reintegration", phase="break", limit=5)
        kinds = {item["kind"] for item in results}
        self.assertIn("checkpoint", kinds)
        self.assertIn("glyph", kinds)

    def test_seed_from_config(self) -> None:
        first = self.write_file("LIFESTORY-CLEAN.md", "LIFESTORY\nCore premise.")
        second = self.write_file("THRESHOLD-CLEAN.md", "THRESHOLD\nTranslation, not instruction.")
        config_path = self.write_file(
            "config.json",
            json.dumps(
                {
                    "documents": [
                        {"path": str(first), "domain": "lifestory"},
                        {"path": str(second), "domain": "threshold"},
                    ]
                }
            ),
        )

        seeded = self.engine.seed_from_config(config_path)
        self.assertEqual(len(seeded), 2)
        self.assertEqual(self.engine.status()["canon_documents"], 2)

    def test_dashboard_state_exposes_recent_material(self) -> None:
        self.engine.log_episode(
            title="Threshold moment",
            content="The system stayed with the pause instead of rushing past it.",
            phase="tension",
        )
        self.engine.save_checkpoint(
            task_key="threshold-demo",
            intent="Preserve the live thread.",
            active_hypothesis="Naming the next step lowers drift.",
            next_step="Return with the same point of focus.",
            resume_cue="Pick up the exact thread, not a reconstructed copy.",
        )

        state = self.engine.dashboard_state()
        self.assertIn("episodes", state)
        self.assertEqual(state["episodes"][0]["title"], "Threshold moment")
        self.assertEqual(state["checkpoints"][0]["task_key"], "threshold-demo")


class ThresholdServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = Path(self.temp_dir.name) / "server.sqlite3"
        engine = ThresholdMemoryEngine(self.db_path)
        engine.initialize()
        engine.log_episode(
            title="Server seed",
            content="A seeded memory gives the console something to show.",
            phase="symmetry",
            source="test",
        )
        engine.close()

        self.server = ThresholdHTTPServer(("127.0.0.1", 0), self.db_path)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.addCleanup(self.stop_server)

    def stop_server(self) -> None:
        self.server.shutdown()
        self.thread.join(1)
        self.server.server_close()

    def fetch_json(self, path: str, method: str = "GET", payload: dict | None = None) -> dict:
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = request.Request(f"{self.base_url}{path}", data=data, method=method, headers=headers)
        with request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))

    def test_state_endpoint_returns_dashboard_payload(self) -> None:
        payload = self.fetch_json("/api/state")
        self.assertIn("status", payload)
        self.assertEqual(payload["status"]["episodes"], 1)
        self.assertTrue(payload["db_path"].endswith("server.sqlite3"))

    def test_demo_endpoint_mutates_state(self) -> None:
        response = self.fetch_json("/api/demo", method="POST", payload={"action": "full"})
        self.assertTrue(response["ok"])

        state = self.fetch_json("/api/state")
        self.assertGreaterEqual(state["status"]["episodes"], 3)
        self.assertGreaterEqual(state["status"]["glyphs"], 1)

    def test_root_serves_console(self) -> None:
        with request.urlopen(f"{self.base_url}/") as response:
            html = response.read().decode("utf-8")
        self.assertIn("Threshold Console", html)


if __name__ == "__main__":
    unittest.main()

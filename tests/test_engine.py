from __future__ import annotations

import json
import tempfile
import threading
import unittest
from datetime import UTC, datetime, timedelta
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


    def test_episode_includes_freshness_and_confidence(self) -> None:
        result = self.engine.log_episode(
            title="Freshness test",
            content="A new memory should arrive fresh and with initial confidence.",
            phase="symmetry",
        )
        self.assertEqual(result["freshness"], 1.0)
        self.assertEqual(result["confidence"], 0.3)

        episodes = self.engine.list_recent_episodes(limit=1)
        self.assertIn("freshness", episodes[0])
        self.assertIn("confidence", episodes[0])
        self.assertAlmostEqual(episodes[0]["freshness"], 1.0, places=1)
        self.assertEqual(episodes[0]["confidence"], 0.3)

    def test_affirm_boosts_confidence(self) -> None:
        ep = self.engine.log_episode(
            title="Affirmable",
            content="This memory will be affirmed.",
            phase="tension",
            delta_coherence=3.0,
            effort=1.0,
        )
        result = self.engine.affirm("episode", ep["episode_id"])
        self.assertEqual(result["confidence_before"], 0.3)
        self.assertAlmostEqual(result["confidence_after"], 0.55, places=2)

        # Affirm again — confidence rises further
        result2 = self.engine.affirm("episode", ep["episode_id"])
        self.assertAlmostEqual(result2["confidence_after"], 0.8, places=2)

        # Affirm once more — should cap at 1.0
        result3 = self.engine.affirm("episode", ep["episode_id"])
        self.assertLessEqual(result3["confidence_after"], 1.0)

    def test_touch_resets_freshness_not_confidence(self) -> None:
        ep = self.engine.log_episode(
            title="Touchable",
            content="Touch refreshes without affirming.",
            phase="novelty",
        )
        result = self.engine.touch("episode", ep["episode_id"])
        self.assertAlmostEqual(result["freshness"], 1.0, places=1)

        # Confidence unchanged — touch doesn't affirm
        episodes = self.engine.list_recent_episodes(limit=1)
        self.assertEqual(episodes[0]["confidence"], 0.3)

    def test_affirm_works_on_all_kinds(self) -> None:
        cp = self.engine.save_checkpoint(
            task_key="affirm-test",
            intent="Test affirm on checkpoint.",
            active_hypothesis="Affirm works across types.",
            next_step="Check result.",
            resume_cue="Resume here.",
        )
        result = self.engine.affirm("checkpoint", cp["checkpoint_id"])
        self.assertAlmostEqual(result["confidence_after"], 0.55, places=2)

        collapse = self.engine.record_collapse(
            boundary_exceeded="test boundary",
            symptoms="test symptoms",
        )
        result = self.engine.affirm("collapse", collapse["collapse_id"])
        self.assertAlmostEqual(result["confidence_after"], 0.55, places=2)

    def test_glyph_inherits_confidence_from_episodes(self) -> None:
        # Create episodes and affirm one
        ep1 = self.engine.log_episode(
            title="Ep1", content="First episode for glyph.", phase="tension",
            delta_coherence=2.0, effort=1.0,
        )
        self.engine.log_episode(
            title="Ep2", content="Second episode for glyph.", phase="tension",
            delta_coherence=3.0, effort=1.0,
        )
        self.engine.affirm("episode", ep1["episode_id"])  # ep1 confidence → 0.55

        glyph = self.engine.consolidate(limit=2)
        self.assertIsNotNone(glyph)
        # Glyph confidence should be at least 0.5 (floor) and reflect inherited average
        glyphs = self.engine.list_glyphs(limit=1)
        self.assertGreaterEqual(glyphs[0]["confidence"], 0.5)

    def test_sweep_fades_old_unaffirmed_episodes(self) -> None:
        # Create an episode and manually backdate it so it looks old
        ep = self.engine.log_episode(
            title="Ancient memory",
            content="This episode is very old and has never been affirmed.",
            phase="symmetry",
        )
        # Backdate: set touched_at and created_at to 60 days ago
        old_time = (datetime.now(UTC) - timedelta(days=60)).replace(microsecond=0).isoformat()
        self.engine.conn.execute(
            "UPDATE episodes SET touched_at = ?, created_at = ? WHERE id = ?",
            (old_time, old_time, ep["episode_id"]),
        )
        self.engine.conn.commit()

        result = self.engine.sweep()
        self.assertEqual(result["newly_faded"], 1)
        self.assertIn(ep["episode_id"], result["faded_ids"])
        self.assertEqual(result["totals"]["faded"], 1)

    def test_sweep_does_not_fade_fresh_episodes(self) -> None:
        self.engine.log_episode(
            title="Fresh memory",
            content="This just happened.",
            phase="novelty",
        )
        result = self.engine.sweep()
        self.assertEqual(result["newly_faded"], 0)
        self.assertEqual(result["totals"]["live"], 1)

    def test_sweep_anchors_high_confidence_episodes(self) -> None:
        ep = self.engine.log_episode(
            title="Trusted memory",
            content="This has been affirmed repeatedly.",
            phase="reintegration",
        )
        # Affirm 3 times: 0.3 → 0.55 → 0.8 → 1.0
        self.engine.affirm("episode", ep["episode_id"])
        self.engine.affirm("episode", ep["episode_id"])
        self.engine.affirm("episode", ep["episode_id"])

        result = self.engine.sweep()
        self.assertEqual(result["totals"]["anchored"], 1)
        self.assertEqual(result["totals"]["faded"], 0)

    def test_affirm_auto_anchors_at_threshold(self) -> None:
        ep = self.engine.log_episode(
            title="Soon anchored",
            content="Will be anchored by the third affirm.",
            phase="tension",
        )
        self.engine.affirm("episode", ep["episode_id"])  # → 0.55
        result = self.engine.affirm("episode", ep["episode_id"])  # → 0.80
        self.assertTrue(result.get("anchored", False))
        self.assertEqual(result["lifecycle"], "anchored")

        # Verify it persists
        episodes = self.engine.list_recent_episodes(limit=1)
        self.assertEqual(episodes[0]["lifecycle"], "anchored")

    def test_episode_lifecycle_in_output(self) -> None:
        ep = self.engine.log_episode(
            title="Lifecycle test",
            content="Should start as live.",
            phase="symmetry",
        )
        self.assertEqual(ep["lifecycle"], "live")

        episodes = self.engine.list_recent_episodes(limit=1)
        self.assertEqual(episodes[0]["lifecycle"], "live")

    def test_promote_requires_confidence(self) -> None:
        # Create a glyph via consolidation
        self.engine.log_episode(title="A", content="first memory", phase="tension", delta_coherence=2.0, effort=1.0)
        self.engine.log_episode(title="B", content="second memory", phase="tension", delta_coherence=3.0, effort=1.0)
        glyph = self.engine.consolidate(limit=2)
        self.assertIsNotNone(glyph)

        # Try to promote without enough confidence — should be refused
        result = self.engine.promote(glyph["glyph_id"])
        self.assertFalse(result["promoted"])
        self.assertIn("below anchor threshold", result["reason"])

    def test_promote_succeeds_after_affirmation(self) -> None:
        self.engine.log_episode(title="A", content="first memory for promotion", phase="novelty", delta_coherence=4.0, effort=1.0)
        self.engine.log_episode(title="B", content="second memory for promotion", phase="novelty", delta_coherence=3.0, effort=1.0)
        glyph = self.engine.consolidate(limit=2)
        self.assertIsNotNone(glyph)

        # Affirm the glyph until it crosses the threshold
        self.engine.affirm("glyph", glyph["glyph_id"])  # 0.5 → 0.75
        self.engine.affirm("glyph", glyph["glyph_id"])  # 0.75 → 1.0

        result = self.engine.promote(glyph["glyph_id"], domain="tested")
        self.assertTrue(result["promoted"])
        self.assertEqual(result["domain"], "tested")

        # Verify it's now in canon_documents
        status = self.engine.status()
        self.assertEqual(status["canon_documents"], 1)

    def test_promote_is_idempotent(self) -> None:
        self.engine.log_episode(title="C", content="idempotent test", phase="reintegration", delta_coherence=2.0, effort=1.0)
        glyph = self.engine.consolidate(limit=1)
        self.engine.affirm("glyph", glyph["glyph_id"])
        self.engine.affirm("glyph", glyph["glyph_id"])

        first = self.engine.promote(glyph["glyph_id"])
        second = self.engine.promote(glyph["glyph_id"])
        self.assertTrue(first["promoted"])
        self.assertTrue(second["promoted"])
        # Should still be just 1 document (ON CONFLICT updates)
        self.assertEqual(self.engine.status()["canon_documents"], 1)

    def test_query_vitals_in_metadata(self) -> None:
        self.engine.log_episode(
            title="Vitals query test",
            content="Resume permission reintegration drift approval.",
            phase="break",
            delta_coherence=2.0,
            effort=1.0,
        )
        self.engine.save_checkpoint(
            task_key="vitals-test",
            intent="Test vitals in query results.",
            active_hypothesis="Vitals appear in metadata.",
            next_step="Check metadata.",
            resume_cue="Resume from permission boundary.",
        )
        results = self.engine.query("resume permission", phase="break", limit=5)
        for item in results:
            if item["kind"] in ("checkpoint", "glyph", "collapse"):
                self.assertIn("freshness", item["metadata"])
                self.assertIn("confidence", item["metadata"])


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

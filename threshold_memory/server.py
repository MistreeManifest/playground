from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .engine import ThresholdMemoryEngine


WEB_ROOT = Path(__file__).with_name("web")


def demo_action(engine: ThresholdMemoryEngine, action: str) -> dict[str, Any]:
    if action == "episode":
        return {
            "action": action,
            "result": engine.log_episode(
                title="Permission shimmer",
                content=(
                    "The thread felt warm before the pause. The task stayed clear, but the next movement "
                    "needed to be preserved before waiting."
                ),
                phase="tension",
                source="demo",
                tags=["demo", "permission"],
                delta_coherence=1.8,
                effort=1.0,
            ),
        }
    if action == "checkpoint":
        return {
            "action": action,
            "result": engine.save_checkpoint(
                task_key="demo-approval-loop",
                intent="Preserve continuity before a permission boundary.",
                active_hypothesis="A precise resume cue prevents drift during waiting.",
                next_step="Resume from the stored cue after permission lands.",
                resume_cue="Return to the exact next step and do not rebuild the task from scratch.",
                risk_flags=["state-loss", "context-drift"],
                state_snapshot={"mode": "waiting", "focus": "continuity"},
            ),
        }
    if action == "collapse":
        checkpoint = engine.next_resume("demo-approval-loop")
        checkpoint_id = checkpoint["checkpoint_id"] if checkpoint else None
        return {
            "action": action,
            "result": engine.record_collapse(
                checkpoint_id=checkpoint_id,
                boundary_exceeded="permission gate",
                symptoms="The next action blurred after waiting and the thread became diffuse.",
                reintegration_notes="Use a smaller resume cue and restore orientation before execution.",
            ),
        }
    if action == "consolidate":
        return {"action": action, "result": engine.consolidate(limit=5)}
    if action == "full":
        results = [
            demo_action(engine, "episode"),
            demo_action(engine, "checkpoint"),
            demo_action(engine, "collapse"),
            {
                "action": "episode",
                "result": engine.log_episode(
                    title="Re-entry with orientation",
                    content=(
                        "The agent resumed by naming intent, risk, and next smallest action. The thread "
                        "returned with less effort."
                    ),
                    phase="reintegration",
                    source="demo",
                    tags=["demo", "recovery"],
                    delta_coherence=2.6,
                    effort=1.1,
                ),
            },
            demo_action(engine, "consolidate"),
        ]
        return {"action": action, "result": results}
    raise ValueError(f"Unsupported demo action: {action}")


class ThresholdRequestHandler(BaseHTTPRequestHandler):
    server: "ThresholdHTTPServer"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_api_get(parsed)
            return
        self._serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            self._send_json({"error": "Unknown endpoint."}, status=HTTPStatus.NOT_FOUND)
            return
        self._handle_api_post(parsed.path)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _handle_api_get(self, parsed) -> None:
        if parsed.path == "/api/state":
            with self.server.engine() as engine:
                state = engine.dashboard_state()
                state["db_path"] = str(self.server.db_path)
            self._send_json(state)
            return

        if parsed.path == "/api/query":
            query = parse_qs(parsed.query)
            text = query.get("q", [""])[0]
            phase = query.get("phase", [None])[0]
            domains = query.get("domain", [])
            with self.server.engine() as engine:
                results = engine.query(text, phase=phase, domains=domains or None, limit=8)
            self._send_json({"query": text, "phase": phase, "domains": domains, "results": results})
            return

        if parsed.path == "/api/kg/neighbors":
            name = parse_qs(parsed.query).get("name", [""])[0].strip()
            if not name:
                self._send_json({"error": "Missing ?name= parameter."}, status=HTTPStatus.BAD_REQUEST)
                return
            with self.server.engine() as engine:
                result = engine.kg_neighbors(name)
            self._send_json(result)
            return

        self._send_json({"error": "Unknown endpoint."}, status=HTTPStatus.NOT_FOUND)

    def _handle_api_post(self, path: str) -> None:
        try:
            payload = self._read_json()
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        try:
            with self.server.engine() as engine:
                if path == "/api/episodes":
                    result = engine.log_episode(
                        title=self._required(payload, "title"),
                        content=self._required(payload, "content"),
                        phase=payload.get("phase", "symmetry"),
                        source=payload.get("source", "console"),
                        tags=payload.get("tags") or [],
                        delta_coherence=payload.get("delta_coherence"),
                        effort=payload.get("effort"),
                    )
                elif path == "/api/checkpoints":
                    result = engine.save_checkpoint(
                        task_key=self._required(payload, "task_key"),
                        intent=self._required(payload, "intent"),
                        active_hypothesis=self._required(payload, "active_hypothesis"),
                        next_step=self._required(payload, "next_step"),
                        resume_cue=self._required(payload, "resume_cue"),
                        risk_flags=payload.get("risk_flags") or [],
                        state_snapshot=payload.get("state_snapshot") or {},
                    )
                elif path == "/api/collapse":
                    result = engine.record_collapse(
                        boundary_exceeded=self._required(payload, "boundary_exceeded"),
                        symptoms=self._required(payload, "symptoms"),
                        checkpoint_id=payload.get("checkpoint_id"),
                        episode_id=payload.get("episode_id"),
                        recovery_protocol=payload.get("recovery_protocol"),
                        reintegration_notes=payload.get("reintegration_notes"),
                    )
                elif path == "/api/consolidate":
                    result = engine.consolidate(limit=int(payload.get("limit", 5)))
                elif path == "/api/kg/entity":
                    result = engine.kg_add_entity(
                        name=self._required(payload, "name"),
                        kind=payload.get("kind", "concept"),
                        description=payload.get("description", ""),
                    )
                elif path == "/api/kg/link":
                    result = engine.kg_link(
                        from_name=self._required(payload, "from_name"),
                        relation=self._required(payload, "relation"),
                        to_name=self._required(payload, "to_name"),
                        weight=float(payload.get("weight", 1.0)),
                    )
                elif path == "/api/demo":
                    result = demo_action(engine, self._required(payload, "action"))
                else:
                    self._send_json({"error": "Unknown endpoint."}, status=HTTPStatus.NOT_FOUND)
                    return
        except (ValueError, KeyError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self._send_json({"ok": True, "result": result})

    def _serve_static(self, raw_path: str) -> None:
        relative = raw_path.lstrip("/") or "index.html"
        candidate = (WEB_ROOT / relative).resolve()
        if WEB_ROOT.resolve() not in candidate.parents and candidate != WEB_ROOT.resolve():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if candidate.is_dir():
            candidate = candidate / "index.html"
        if not candidate.exists():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type, _ = mimetypes.guess_type(candidate.name)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.end_headers()
        self.wfile.write(candidate.read_bytes())

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length else b"{}"
        if not raw_body:
            return {}
        try:
            value = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("Request body must be valid JSON.") from exc
        if not isinstance(value, dict):
            raise ValueError("Request body must be a JSON object.")
        return value

    def _required(self, payload: dict[str, Any], key: str) -> Any:
        value = payload.get(key)
        if value in (None, ""):
            raise KeyError(f"Missing required field: {key}")
        return value

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ThresholdHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], db_path: str | Path):
        super().__init__(server_address, ThresholdRequestHandler)
        self.db_path = Path(db_path)

    def engine(self) -> ThresholdMemoryEngine:
        engine = ThresholdMemoryEngine(self.db_path)
        engine.initialize()
        return engine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Threshold Memory local console.")
    parser.add_argument("--db", default="data/threshold-memory.sqlite3", help="Path to the SQLite database.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    server = ThresholdHTTPServer((args.host, args.port), args.db)
    print(f"Threshold Console running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

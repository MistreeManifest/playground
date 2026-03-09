from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import PHASE_CHOICES, ThresholdMemoryEngine


def add_db_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", default="data/threshold-memory.sqlite3", help="Path to the SQLite database.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Threshold memory prototype.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create the database schema.")
    add_db_argument(init_parser)

    seed_parser = subparsers.add_parser("seed-file", help="Ingest a single canonical markdown file.")
    add_db_argument(seed_parser)
    seed_parser.add_argument("path", help="Markdown file to ingest.")
    seed_parser.add_argument("--domain", help="Optional explicit domain name.")

    seed_config_parser = subparsers.add_parser("seed-config", help="Ingest canonical files from a JSON config.")
    add_db_argument(seed_config_parser)
    seed_config_parser.add_argument("config", help="JSON config with a documents array.")

    episode_parser = subparsers.add_parser("episode", help="Store an episodic memory entry.")
    add_db_argument(episode_parser)
    episode_parser.add_argument("--title", required=True)
    episode_parser.add_argument("--content", required=True)
    episode_parser.add_argument("--phase", default="symmetry", choices=PHASE_CHOICES)
    episode_parser.add_argument("--source", default="manual")
    episode_parser.add_argument("--tag", action="append", default=[])
    episode_parser.add_argument("--delta-coherence", type=float)
    episode_parser.add_argument("--effort", type=float)

    checkpoint_parser = subparsers.add_parser("checkpoint", help="Persist a permission checkpoint.")
    add_db_argument(checkpoint_parser)
    checkpoint_parser.add_argument("--task-key", required=True)
    checkpoint_parser.add_argument("--intent", required=True)
    checkpoint_parser.add_argument("--hypothesis", required=True)
    checkpoint_parser.add_argument("--next-step", required=True)
    checkpoint_parser.add_argument("--resume-cue", required=True)
    checkpoint_parser.add_argument("--risk", action="append", default=[])
    checkpoint_parser.add_argument("--state", help="Optional JSON object describing state snapshot.")

    resume_parser = subparsers.add_parser("resume", help="Show the latest pending checkpoint.")
    add_db_argument(resume_parser)
    resume_parser.add_argument("--task-key")

    resolve_parser = subparsers.add_parser("resolve", help="Mark a checkpoint as resolved.")
    add_db_argument(resolve_parser)
    resolve_parser.add_argument("checkpoint_id", type=int)
    resolve_parser.add_argument("--status", default="resolved")

    collapse_parser = subparsers.add_parser("collapse", help="Record a collapse or failed run.")
    add_db_argument(collapse_parser)
    collapse_parser.add_argument("--boundary", required=True)
    collapse_parser.add_argument("--symptoms", required=True)
    collapse_parser.add_argument("--checkpoint-id", type=int)
    collapse_parser.add_argument("--episode-id", type=int)
    collapse_parser.add_argument("--protocol")
    collapse_parser.add_argument("--notes")

    consolidate_parser = subparsers.add_parser("consolidate", help="Condense recent episodes into a glyph.")
    add_db_argument(consolidate_parser)
    consolidate_parser.add_argument("--limit", type=int, default=5)

    query_parser = subparsers.add_parser("query", help="Retrieve relevant memory.")
    add_db_argument(query_parser)
    query_parser.add_argument("text", help="Search query.")
    query_parser.add_argument("--phase", choices=PHASE_CHOICES)
    query_parser.add_argument("--domain", action="append")
    query_parser.add_argument("--limit", type=int, default=6)

    affirm_parser = subparsers.add_parser("affirm", help="Affirm a memory — boost confidence and refresh.")
    add_db_argument(affirm_parser)
    affirm_parser.add_argument("kind", choices=["episode", "checkpoint", "collapse", "glyph"])
    affirm_parser.add_argument("item_id", type=int)

    touch_parser = subparsers.add_parser("touch", help="Touch a memory — reset freshness without changing confidence.")
    add_db_argument(touch_parser)
    touch_parser.add_argument("kind", choices=["episode", "checkpoint", "collapse", "glyph"])
    touch_parser.add_argument("item_id", type=int)

    promote_parser = subparsers.add_parser("promote", help="Promote a high-confidence glyph into a canon document.")
    add_db_argument(promote_parser)
    promote_parser.add_argument("glyph_id", type=int)
    promote_parser.add_argument("--domain", default="emergent", help="Domain for the new canon document.")

    sweep_parser = subparsers.add_parser("sweep", help="Fade old, unaffirmed episodes. Anchor high-confidence ones.")
    add_db_argument(sweep_parser)

    status_parser = subparsers.add_parser("status", help="Show table counts.")
    add_db_argument(status_parser)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    engine = ThresholdMemoryEngine(Path(args.db))
    try:
        engine.initialize()
        result = run_command(engine, args)
    finally:
        engine.close()
    print(json.dumps(result, indent=2, sort_keys=True))


def run_command(engine: ThresholdMemoryEngine, args: argparse.Namespace) -> object:
    if args.command == "init":
        return {"status": "initialized", "db": str(args.db)}
    if args.command == "seed-file":
        return engine.seed_document(args.path, args.domain)
    if args.command == "seed-config":
        return engine.seed_from_config(args.config)
    if args.command == "episode":
        return engine.log_episode(
            title=args.title,
            content=args.content,
            phase=args.phase,
            source=args.source,
            tags=args.tag,
            delta_coherence=args.delta_coherence,
            effort=args.effort,
        )
    if args.command == "checkpoint":
        state = json.loads(args.state) if args.state else {}
        return engine.save_checkpoint(
            task_key=args.task_key,
            intent=args.intent,
            active_hypothesis=args.hypothesis,
            next_step=args.next_step,
            resume_cue=args.resume_cue,
            risk_flags=args.risk,
            state_snapshot=state,
        )
    if args.command == "resume":
        return engine.next_resume(task_key=args.task_key)
    if args.command == "resolve":
        engine.resolve_checkpoint(args.checkpoint_id, status=args.status)
        return {"checkpoint_id": args.checkpoint_id, "status": args.status}
    if args.command == "collapse":
        return engine.record_collapse(
            boundary_exceeded=args.boundary,
            symptoms=args.symptoms,
            checkpoint_id=args.checkpoint_id,
            episode_id=args.episode_id,
            recovery_protocol=args.protocol,
            reintegration_notes=args.notes,
        )
    if args.command == "consolidate":
        return engine.consolidate(limit=args.limit)
    if args.command == "query":
        return engine.query(args.text, phase=args.phase, domains=args.domain, limit=args.limit)
    if args.command == "affirm":
        return engine.affirm(args.kind, args.item_id)
    if args.command == "touch":
        return engine.touch(args.kind, args.item_id)
    if args.command == "promote":
        return engine.promote(args.glyph_id, domain=args.domain)
    if args.command == "sweep":
        return engine.sweep()
    if args.command == "status":
        return engine.status()
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()

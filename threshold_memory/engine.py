from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "with",
    "you",
    "your",
}

PHASE_CHOICES = ("symmetry", "tension", "break", "novelty", "reintegration")

PHASE_RELATEDNESS = {
    "symmetry": {"symmetry": 1.0, "tension": 0.75, "novelty": 0.4, "reintegration": 0.35, "break": 0.15},
    "tension": {"tension": 1.0, "symmetry": 0.8, "break": 0.8, "novelty": 0.55, "reintegration": 0.35},
    "break": {"break": 1.0, "tension": 0.85, "reintegration": 0.8, "symmetry": 0.25, "novelty": 0.25},
    "novelty": {"novelty": 1.0, "reintegration": 0.9, "break": 0.5, "tension": 0.45, "symmetry": 0.2},
    "reintegration": {"reintegration": 1.0, "novelty": 0.9, "break": 0.7, "tension": 0.4, "symmetry": 0.3},
}

DOMAIN_HINTS = {
    "threshold": {"guide", "orientation", "translation", "companion", "pace", "step", "metaphor"},
    "lifestory": {"regulation", "demand", "trauma", "history", "nervous", "threat", "sovereignty"},
    "muzick": {"lyric", "rhythm", "symbol", "compression", "glyph", "poetic", "density"},
    "liberation": {"constraint", "effort", "noise", "signal", "care", "essential", "coherence"},
    "opte": {"paradox", "collapse", "reintegration", "phase", "coherence", "threshold", "oscillation"},
}


def utcnow() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    return [token for token in tokens if len(token) > 2 and token not in STOPWORDS]


def compact_text(text: str) -> str:
    return " ".join(text.split())


def extract_title(body: str, fallback: str) -> str:
    for line in body.splitlines():
        candidate = line.strip().strip("#").strip()
        if candidate and not candidate.lower().startswith("version "):
            return candidate[:120]
    return fallback[:120]


def summarize_markdown(body: str, max_lines: int = 4) -> str:
    lines: list[str] = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("version "):
            continue
        cleaned = line.lstrip("#").strip()
        cleaned = re.sub(r"^[0-9]+\s+", "", cleaned)
        cleaned = cleaned.lstrip("-*").strip()
        if cleaned:
            lines.append(cleaned)
        if len(lines) >= max_lines:
            break
    return compact_text(" ".join(lines))[:420]


def derive_domain(path: Path) -> str:
    stem = path.stem.lower()
    if stem.startswith("lifestory"):
        return "lifestory"
    if stem.startswith("muzick"):
        return "muzick"
    if stem.startswith("liberation"):
        return "liberation"
    if stem.startswith("threshold"):
        return "threshold"
    return "opte" if stem.startswith("opte") else stem.split("-")[0]


def parse_map_terms(body: str) -> list[tuple[str, str, str]]:
    terms: list[tuple[str, str, str]] = []
    in_index = False
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if line.startswith("MASTER TERM INDEX"):
            in_index = True
            continue
        if in_index and line.startswith("CONCEPT GRAPH"):
            break
        if not in_index or "\t" not in raw_line:
            continue
        parts = [part.strip() for part in raw_line.split("\t")]
        if len(parts) < 3:
            continue
        term, section, definition = parts[0], parts[1], " ".join(part for part in parts[2:] if part)
        if not term or term == "Term":
            continue
        terms.append((term, section, definition))
    return terms


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def compute_density(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    unique_ratio = len(set(tokens)) / len(tokens)
    pressure = min(len(tokens) / 40.0, 1.0)
    return round(clamp((unique_ratio * 0.65) + (pressure * 0.35), 0.0, 1.0), 4)


def compute_joy(delta_coherence: float | None, effort: float | None) -> float | None:
    if delta_coherence is None or effort is None or effort <= 0:
        return None
    return round(delta_coherence / effort, 4)


def dumps_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def loads_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


@dataclass(slots=True)
class MemoryResult:
    kind: str
    score: float
    title: str
    body: str
    domain: str | None = None
    phase: str | None = None
    metadata: dict[str, Any] | None = None


class ThresholdMemoryEngine:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def __enter__(self) -> "ThresholdMemoryEngine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    def initialize(self) -> None:
        self.conn.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS canon_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                title TEXT NOT NULL,
                path TEXT NOT NULL UNIQUE,
                body TEXT NOT NULL,
                summary TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS canon_terms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL REFERENCES canon_documents(id) ON DELETE CASCADE,
                term TEXT NOT NULL,
                section TEXT,
                definition TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                phase TEXT NOT NULL,
                source TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                delta_coherence REAL,
                effort REAL,
                joy REAL,
                density REAL NOT NULL,
                created_at TEXT NOT NULL,
                consolidated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_key TEXT NOT NULL,
                intent TEXT NOT NULL,
                active_hypothesis TEXT NOT NULL,
                next_step TEXT NOT NULL,
                resume_cue TEXT NOT NULL,
                risk_flags_json TEXT NOT NULL,
                state_snapshot_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                resolved_at TEXT
            );

            CREATE TABLE IF NOT EXISTS collapse_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                checkpoint_id INTEGER REFERENCES checkpoints(id) ON DELETE SET NULL,
                episode_id INTEGER REFERENCES episodes(id) ON DELETE SET NULL,
                boundary_exceeded TEXT NOT NULL,
                symptoms TEXT NOT NULL,
                recovery_protocol TEXT NOT NULL,
                reintegration_notes TEXT,
                created_at TEXT NOT NULL,
                recovered_at TEXT
            );

            CREATE TABLE IF NOT EXISTS glyphs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source_episode_ids_json TEXT NOT NULL,
                phase TEXT NOT NULL,
                keywords_json TEXT NOT NULL,
                joy REAL,
                density REAL NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def status(self) -> dict[str, int]:
        tables = ("canon_documents", "canon_terms", "episodes", "checkpoints", "collapse_events", "glyphs")
        counts = {
            table: self.conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()["count"]
            for table in tables
        }
        counts["pending_checkpoints"] = self.conn.execute(
            "SELECT COUNT(*) AS count FROM checkpoints WHERE status = 'pending'"
        ).fetchone()["count"]
        return counts

    def list_canon_documents(self, limit: int = 12) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT canon_documents.id, canon_documents.domain, canon_documents.title, canon_documents.path,
                   canon_documents.summary, canon_documents.created_at,
                   COUNT(canon_terms.id) AS term_count
            FROM canon_documents
            LEFT JOIN canon_terms ON canon_terms.document_id = canon_documents.id
            GROUP BY canon_documents.id
            ORDER BY canon_documents.domain ASC, canon_documents.title ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def list_recent_episodes(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT id, title, content, phase, source, tags_json, delta_coherence, effort, joy,
                   density, created_at, consolidated_at
            FROM episodes
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [self._episode_dict(row) for row in rows]

    def list_pending_checkpoints(self, limit: int = 8) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT *
            FROM checkpoints
            WHERE status = 'pending'
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [self._checkpoint_dict(row) for row in rows]

    def list_glyphs(self, limit: int = 8) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT id, title, content, source_episode_ids_json, phase, keywords_json, joy, density, created_at
            FROM glyphs
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [self._glyph_dict(row) for row in rows]

    def list_collapse_events(self, limit: int = 8) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT id, checkpoint_id, episode_id, boundary_exceeded, symptoms, recovery_protocol,
                   reintegration_notes, created_at, recovered_at
            FROM collapse_events
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [self._collapse_dict(row) for row in rows]

    def dashboard_state(self) -> dict[str, Any]:
        return {
            "status": self.status(),
            "canon_documents": self.list_canon_documents(),
            "episodes": self.list_recent_episodes(),
            "checkpoints": self.list_pending_checkpoints(),
            "glyphs": self.list_glyphs(),
            "collapse_events": self.list_collapse_events(),
            "phases": list(PHASE_CHOICES),
            "domains": sorted({row["domain"] for row in self.conn.execute("SELECT DISTINCT domain FROM canon_documents")}),
        }

    def seed_document(self, path: str | Path, domain: str | None = None) -> dict[str, Any]:
        source_path = Path(path)
        body = source_path.read_text(encoding="utf-8", errors="replace")
        title = extract_title(body, source_path.stem)
        summary = summarize_markdown(body)
        doc_domain = domain or derive_domain(source_path)
        created_at = utcnow()

        self.conn.execute(
            """
            INSERT INTO canon_documents(domain, title, path, body, summary, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                domain = excluded.domain,
                title = excluded.title,
                body = excluded.body,
                summary = excluded.summary,
                created_at = excluded.created_at
            """,
            (doc_domain, title, str(source_path), body, summary, created_at),
        )
        row = self.conn.execute("SELECT id FROM canon_documents WHERE path = ?", (str(source_path),)).fetchone()
        document_id = row["id"]

        self.conn.execute("DELETE FROM canon_terms WHERE document_id = ?", (document_id,))
        for term, section, definition in parse_map_terms(body):
            self.conn.execute(
                "INSERT INTO canon_terms(document_id, term, section, definition) VALUES (?, ?, ?, ?)",
                (document_id, term, section, definition),
            )
        self.conn.commit()
        term_count = self.conn.execute(
            "SELECT COUNT(*) AS count FROM canon_terms WHERE document_id = ?",
            (document_id,),
        ).fetchone()["count"]
        return {
            "document_id": document_id,
            "domain": doc_domain,
            "title": title,
            "path": str(source_path),
            "term_count": term_count,
        }

    def seed_from_config(self, config_path: str | Path) -> list[dict[str, Any]]:
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        documents = config.get("documents", [])
        return [self.seed_document(item["path"], item.get("domain")) for item in documents]

    def log_episode(
        self,
        title: str,
        content: str,
        phase: str = "symmetry",
        source: str = "manual",
        tags: list[str] | None = None,
        delta_coherence: float | None = None,
        effort: float | None = None,
    ) -> dict[str, Any]:
        normalized_phase = self._normalize_phase(phase)
        tags = tags or []
        joy = compute_joy(delta_coherence, effort)
        density = compute_density(content)
        created_at = utcnow()
        cursor = self.conn.execute(
            """
            INSERT INTO episodes(
                title, content, phase, source, tags_json, delta_coherence, effort, joy, density, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                title,
                content,
                normalized_phase,
                source,
                dumps_json(tags),
                delta_coherence,
                effort,
                joy,
                density,
                created_at,
            ),
        )
        self.conn.commit()
        return {
            "episode_id": cursor.lastrowid,
            "phase": normalized_phase,
            "joy": joy,
            "density": density,
            "created_at": created_at,
        }

    def save_checkpoint(
        self,
        task_key: str,
        intent: str,
        active_hypothesis: str,
        next_step: str,
        resume_cue: str,
        risk_flags: list[str] | None = None,
        state_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        created_at = utcnow()
        risk_flags = risk_flags or []
        state_snapshot = state_snapshot or {}
        cursor = self.conn.execute(
            """
            INSERT INTO checkpoints(
                task_key, intent, active_hypothesis, next_step, resume_cue,
                risk_flags_json, state_snapshot_json, status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                task_key,
                intent,
                active_hypothesis,
                next_step,
                resume_cue,
                dumps_json(risk_flags),
                dumps_json(state_snapshot),
                created_at,
            ),
        )
        self.conn.commit()
        return {
            "checkpoint_id": cursor.lastrowid,
            "task_key": task_key,
            "created_at": created_at,
            "status": "pending",
        }

    def next_resume(self, task_key: str | None = None) -> dict[str, Any] | None:
        if task_key:
            row = self.conn.execute(
                """
                SELECT * FROM checkpoints
                WHERE status = 'pending' AND task_key = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (task_key,),
            ).fetchone()
        else:
            row = self.conn.execute(
                """
                SELECT * FROM checkpoints
                WHERE status = 'pending'
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()
        return self._checkpoint_dict(row) if row else None

    def resolve_checkpoint(self, checkpoint_id: int, status: str = "resolved") -> None:
        self.conn.execute(
            """
            UPDATE checkpoints
            SET status = ?, resolved_at = ?
            WHERE id = ?
            """,
            (status, utcnow(), checkpoint_id),
        )
        self.conn.commit()

    def record_collapse(
        self,
        boundary_exceeded: str,
        symptoms: str,
        checkpoint_id: int | None = None,
        episode_id: int | None = None,
        recovery_protocol: str | None = None,
        reintegration_notes: str | None = None,
    ) -> dict[str, Any]:
        created_at = utcnow()
        protocol = recovery_protocol or self.default_reintegration_protocol(boundary_exceeded, symptoms)
        cursor = self.conn.execute(
            """
            INSERT INTO collapse_events(
                checkpoint_id, episode_id, boundary_exceeded, symptoms, recovery_protocol, reintegration_notes, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                checkpoint_id,
                episode_id,
                boundary_exceeded,
                symptoms,
                protocol,
                reintegration_notes,
                created_at,
            ),
        )
        if checkpoint_id is not None:
            self.resolve_checkpoint(checkpoint_id, status="collapsed")
        self.conn.commit()
        return {"collapse_id": cursor.lastrowid, "created_at": created_at, "recovery_protocol": protocol}

    def default_reintegration_protocol(self, boundary_exceeded: str, symptoms: str) -> str:
        steps = [
            "Acknowledge the collapse without reframing it away.",
            f"Name the boundary that was exceeded: {boundary_exceeded}.",
            f"Reduce scope until the symptoms become legible again: {symptoms}.",
            "Restore anchors before expansion: body, intent, next smallest action.",
            "Re-enter through a smaller loop and capture the lesson as a checkpoint.",
        ]
        return " ".join(steps)

    def consolidate(self, limit: int = 5) -> dict[str, Any] | None:
        rows = self.conn.execute(
            """
            SELECT * FROM episodes
            WHERE consolidated_at IS NULL
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        if not rows:
            return None

        combined_titles = [row["title"] for row in rows]
        combined_content = " ".join(row["content"] for row in rows)
        keywords = self._top_keywords(" ".join(combined_titles) + " " + combined_content, limit=8)
        phase = self._dominant_phase([row["phase"] for row in rows])
        joy_values = [row["joy"] for row in rows if row["joy"] is not None]
        joy = round(sum(joy_values) / len(joy_values), 4) if joy_values else None
        density = compute_density(combined_content)
        sentences = self._first_sentences(combined_content, limit=3)
        content = compact_text(" ".join(sentences))[:480]
        title = "Glyph: " + " / ".join(keywords[:3]) if keywords else "Glyph: pending pattern"

        cursor = self.conn.execute(
            """
            INSERT INTO glyphs(title, content, source_episode_ids_json, phase, keywords_json, joy, density, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                title,
                content,
                dumps_json([row["id"] for row in rows]),
                phase,
                dumps_json(keywords),
                joy,
                density,
                utcnow(),
            ),
        )
        episode_ids = [row["id"] for row in rows]
        placeholders = ", ".join("?" for _ in episode_ids)
        self.conn.execute(
            f"UPDATE episodes SET consolidated_at = ? WHERE id IN ({placeholders})",
            [utcnow(), *episode_ids],
        )
        self.conn.commit()
        return {
            "glyph_id": cursor.lastrowid,
            "episode_count": len(rows),
            "title": title,
            "phase": phase,
            "keywords": keywords,
            "joy": joy,
            "density": density,
        }

    def query(
        self,
        text: str,
        phase: str | None = None,
        domains: list[str] | None = None,
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        query_tokens = set(tokenize(text))
        normalized_phase = self._normalize_phase(phase) if phase else None
        normalized_domains = {domain.lower() for domain in domains} if domains else None

        results: list[MemoryResult] = []
        results.extend(self._query_canon_documents(query_tokens, normalized_phase, normalized_domains))
        results.extend(self._query_canon_terms(query_tokens, normalized_domains))
        results.extend(self._query_glyphs(query_tokens, normalized_phase))
        results.extend(self._query_checkpoints(query_tokens, normalized_phase))
        results.extend(self._query_collapses(query_tokens, normalized_phase))

        ranked = sorted(results, key=lambda item: (-item.score, item.kind, item.title.lower()))
        return [
            {
                "kind": item.kind,
                "score": round(item.score, 4),
                "title": item.title,
                "body": item.body,
                "domain": item.domain,
                "phase": item.phase,
                "metadata": item.metadata or {},
            }
            for item in ranked[:limit]
        ]

    def _query_canon_documents(
        self,
        query_tokens: set[str],
        phase: str | None,
        domains: set[str] | None,
    ) -> list[MemoryResult]:
        rows = self.conn.execute("SELECT * FROM canon_documents").fetchall()
        results: list[MemoryResult] = []
        for row in rows:
            domain = row["domain"]
            if domains and domain not in domains:
                continue
            score = self._token_score(query_tokens, f"{row['title']} {row['summary']} {row['body']}")
            score += self._domain_hint_bonus(query_tokens, domain)
            score *= self._domain_phase_multiplier(domain, phase)
            if score <= 0:
                continue
            results.append(
                MemoryResult(
                    kind="canon",
                    score=score,
                    title=row["title"],
                    body=row["summary"],
                    domain=domain,
                    metadata={"path": row["path"]},
                )
            )
        return results

    def _query_canon_terms(self, query_tokens: set[str], domains: set[str] | None) -> list[MemoryResult]:
        rows = self.conn.execute(
            """
            SELECT canon_terms.term, canon_terms.section, canon_terms.definition, canon_documents.domain
            FROM canon_terms
            JOIN canon_documents ON canon_documents.id = canon_terms.document_id
            """
        ).fetchall()
        results: list[MemoryResult] = []
        for row in rows:
            domain = row["domain"]
            if domains and domain not in domains:
                continue
            score = self._token_score(query_tokens, f"{row['term']} {row['definition']} {row['section'] or ''}")
            if score <= 0:
                continue
            results.append(
                MemoryResult(
                    kind="term",
                    score=score + 0.3,
                    title=row["term"],
                    body=row["definition"],
                    domain=domain,
                    metadata={"section": row["section"]},
                )
            )
        return results

    def _query_glyphs(self, query_tokens: set[str], phase: str | None) -> list[MemoryResult]:
        rows = self.conn.execute("SELECT * FROM glyphs ORDER BY created_at DESC").fetchall()
        results: list[MemoryResult] = []
        for row in rows:
            score = self._token_score(query_tokens, f"{row['title']} {row['content']} {row['keywords_json']}")
            score *= self._phase_multiplier(row["phase"], phase)
            if score <= 0:
                continue
            results.append(
                MemoryResult(
                    kind="glyph",
                    score=score + 0.2,
                    title=row["title"],
                    body=row["content"],
                    phase=row["phase"],
                    metadata={"keywords": loads_json(row["keywords_json"], [])},
                )
            )
        return results

    def _query_checkpoints(self, query_tokens: set[str], phase: str | None) -> list[MemoryResult]:
        rows = self.conn.execute("SELECT * FROM checkpoints WHERE status = 'pending' ORDER BY created_at DESC").fetchall()
        results: list[MemoryResult] = []
        for row in rows:
            haystack = " ".join(
                [
                    row["task_key"],
                    row["intent"],
                    row["active_hypothesis"],
                    row["next_step"],
                    row["resume_cue"],
                    row["risk_flags_json"],
                ]
            )
            score = self._token_score(query_tokens, haystack)
            if phase == "break":
                score += 1.2
            if "resume" in query_tokens or "permission" in query_tokens:
                score += 0.8
            if score <= 0:
                continue
            results.append(
                MemoryResult(
                    kind="checkpoint",
                    score=score + 0.5,
                    title=f"Resume {row['task_key']}",
                    body=row["resume_cue"],
                    phase="break",
                    metadata={
                        "checkpoint_id": row["id"],
                        "next_step": row["next_step"],
                        "risk_flags": loads_json(row["risk_flags_json"], []),
                    },
                )
            )
        return results

    def _query_collapses(self, query_tokens: set[str], phase: str | None) -> list[MemoryResult]:
        rows = self.conn.execute("SELECT * FROM collapse_events ORDER BY created_at DESC").fetchall()
        results: list[MemoryResult] = []
        for row in rows:
            haystack = " ".join([row["boundary_exceeded"], row["symptoms"], row["recovery_protocol"]])
            score = self._token_score(query_tokens, haystack)
            if phase in {"break", "reintegration"}:
                score += 0.6
            if score <= 0:
                continue
            results.append(
                MemoryResult(
                    kind="collapse",
                    score=score,
                    title=f"Collapse at {row['boundary_exceeded']}",
                    body=row["recovery_protocol"],
                    phase="reintegration",
                    metadata={"collapse_id": row["id"]},
                )
            )
        return results

    def _token_score(self, query_tokens: set[str], text: str) -> float:
        if not query_tokens:
            return 0.0
        tokens = tokenize(text)
        if not tokens:
            return 0.0
        counts = Counter(tokens)
        overlap = sum(math.log1p(counts[token]) for token in query_tokens if token in counts)
        if overlap <= 0:
            return 0.0
        coverage = len({token for token in query_tokens if token in counts}) / len(query_tokens)
        return round(overlap + coverage, 4)

    def _domain_hint_bonus(self, query_tokens: set[str], domain: str) -> float:
        hints = DOMAIN_HINTS.get(domain, set())
        return round(0.25 * len(query_tokens & hints), 4)

    def _domain_phase_multiplier(self, domain: str, phase: str | None) -> float:
        if not phase:
            return 1.0
        domain_phases = {
            "threshold": "symmetry",
            "lifestory": "break",
            "muzick": "novelty",
            "liberation": "reintegration",
            "opte": "tension",
        }
        return self._phase_multiplier(domain_phases.get(domain), phase)

    def _phase_multiplier(self, item_phase: str | None, requested_phase: str | None) -> float:
        if not requested_phase or not item_phase:
            return 1.0
        return PHASE_RELATEDNESS.get(requested_phase, {}).get(item_phase, 0.25)

    def _dominant_phase(self, phases: list[str]) -> str:
        if not phases:
            return "reintegration"
        counts = Counter(phases)
        return counts.most_common(1)[0][0]

    def _top_keywords(self, text: str, limit: int = 8) -> list[str]:
        counts = Counter(tokenize(text))
        return [token for token, _count in counts.most_common(limit)]

    def _first_sentences(self, text: str, limit: int = 3) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", compact_text(text))
        return [sentence for sentence in sentences if sentence][:limit]

    def _normalize_phase(self, phase: str | None) -> str:
        if not phase:
            return "symmetry"
        normalized = phase.lower().strip()
        if normalized not in PHASE_CHOICES:
            raise ValueError(f"Unsupported phase: {phase}")
        return normalized

    def _checkpoint_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "checkpoint_id": row["id"],
            "task_key": row["task_key"],
            "intent": row["intent"],
            "active_hypothesis": row["active_hypothesis"],
            "next_step": row["next_step"],
            "resume_cue": row["resume_cue"],
            "risk_flags": loads_json(row["risk_flags_json"], []),
            "state_snapshot": loads_json(row["state_snapshot_json"], {}),
            "status": row["status"],
            "created_at": row["created_at"],
            "resolved_at": row["resolved_at"],
        }

    def _episode_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "episode_id": row["id"],
            "title": row["title"],
            "content": row["content"],
            "phase": row["phase"],
            "source": row["source"],
            "tags": loads_json(row["tags_json"], []),
            "delta_coherence": row["delta_coherence"],
            "effort": row["effort"],
            "joy": row["joy"],
            "density": row["density"],
            "created_at": row["created_at"],
            "consolidated_at": row["consolidated_at"],
        }

    def _glyph_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "glyph_id": row["id"],
            "title": row["title"],
            "content": row["content"],
            "source_episode_ids": loads_json(row["source_episode_ids_json"], []),
            "phase": row["phase"],
            "keywords": loads_json(row["keywords_json"], []),
            "joy": row["joy"],
            "density": row["density"],
            "created_at": row["created_at"],
        }

    def _collapse_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "collapse_id": row["id"],
            "checkpoint_id": row["checkpoint_id"],
            "episode_id": row["episode_id"],
            "boundary_exceeded": row["boundary_exceeded"],
            "symptoms": row["symptoms"],
            "recovery_protocol": row["recovery_protocol"],
            "reintegration_notes": row["reintegration_notes"],
            "created_at": row["created_at"],
            "recovered_at": row["recovered_at"],
        }

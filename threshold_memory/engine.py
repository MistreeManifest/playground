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

# Freshness decay: how quickly memories fade without interaction.
# Half-life in days — after this many days, freshness drops to 0.5.
FRESHNESS_HALF_LIFE_DAYS = 7.0
# Glyphs are condensed — dense things hold heat longer.
GLYPH_HALF_LIFE_DAYS = 21.0

# Confidence: initial value for new memories. Rises with affirmation.
CONFIDENCE_INITIAL = 0.3
CONFIDENCE_AFFIRM_BOOST = 0.25
CONFIDENCE_MAX = 1.0

# Anchoring: once confidence reaches this threshold, the memory stops decaying.
CONFIDENCE_ANCHOR_THRESHOLD = 0.8

# Fading: when freshness drops below this AND confidence is below anchor threshold,
# the episode is marked "faded" — still retrievable, but quiet in results.
FRESHNESS_FADE_THRESHOLD = 0.15

# Episode lifecycle states
EPISODE_LIVE = "live"
EPISODE_FADED = "faded"
EPISODE_ANCHORED = "anchored"


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


def compute_freshness(created_or_touched: str, half_life_days: float = FRESHNESS_HALF_LIFE_DAYS) -> float:
    """Exponential decay from 1.0 toward 0.0 based on age.

    A memory created right now has freshness 1.0.
    After half_life_days, it drops to 0.5. After two half-lives, 0.25.
    Think of it like a tone fading — the note is still there, but quieter.
    """
    try:
        ts = datetime.fromisoformat(created_or_touched)
    except (ValueError, TypeError):
        return 0.5  # unknown age → neutral
    now = datetime.now(UTC)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    age_days = max((now - ts).total_seconds() / 86400.0, 0.0)
    decay = math.exp(-0.693 * age_days / half_life_days)  # ln(2) ≈ 0.693
    return round(clamp(decay, 0.0, 1.0), 4)


def dumps_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def loads_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


# ── Sanitization ──────────────────────────────────────────────────────────────
# Borrowed from OpenFang's session_repair approach: strip prompt-injection
# markers and oversized base64-like blobs from untrusted content at the
# system boundary — before anything is written to the store.

_INJECTION_RE = re.compile(
    r"(<\|im_start\|>|<\|im_end\|>|IGNORE\s+PREVIOUS\s+INSTRUCTIONS?|SYSTEM\s+PROMPT\s+OVERRIDE)",
    re.IGNORECASE,
)
_BASE64_BLOB_RE = re.compile(r"[A-Za-z0-9+/]{1000,}={0,2}")
_MAX_CONTENT_LENGTH = 10_000


def sanitize_content(text: str) -> str:
    """Strip prompt injection markers and oversized base64-like blobs.

    Applied to all user-supplied text before it enters the store.
    Truncation is a last-resort ceiling, not a routine operation.
    """
    text = _BASE64_BLOB_RE.sub("[redacted-blob]", text)
    text = _INJECTION_RE.sub("[redacted]", text)
    return text[:_MAX_CONTENT_LENGTH]


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
    def __init__(self, db_path: str | Path, mirror_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.mirror_path: Path | None = Path(mirror_path) if mirror_path else None

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
                confidence REAL NOT NULL DEFAULT 0.3,
                lifecycle TEXT NOT NULL DEFAULT 'live',
                touched_at TEXT,
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
                confidence REAL NOT NULL DEFAULT 0.3,
                touched_at TEXT,
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
                confidence REAL NOT NULL DEFAULT 0.3,
                touched_at TEXT,
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
                confidence REAL NOT NULL DEFAULT 0.5,
                touched_at TEXT,
                created_at TEXT NOT NULL
            );

            -- Knowledge graph: entities and their relations.
            -- Adapted from OpenFang's openfang-memory knowledge graph layer.
            -- Entities are named concepts discovered from canon terms or added manually.
            -- Relations connect them with typed, weighted edges.

            CREATE TABLE IF NOT EXISTS kg_entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL DEFAULT 'concept',
                description TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.3,
                touched_at TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS kg_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_entity_id INTEGER NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
                to_entity_id INTEGER NOT NULL REFERENCES kg_entities(id) ON DELETE CASCADE,
                relation TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                UNIQUE(from_entity_id, to_entity_id, relation)
            );
            """
        )
        # Migrate existing databases: add new columns if missing.
        self._migrate_add_column("episodes", "confidence", "REAL NOT NULL DEFAULT 0.3")
        self._migrate_add_column("episodes", "lifecycle", "TEXT NOT NULL DEFAULT 'live'")
        self._migrate_add_column("episodes", "touched_at", "TEXT")
        self._migrate_add_column("checkpoints", "confidence", "REAL NOT NULL DEFAULT 0.3")
        self._migrate_add_column("checkpoints", "touched_at", "TEXT")
        self._migrate_add_column("collapse_events", "confidence", "REAL NOT NULL DEFAULT 0.3")
        self._migrate_add_column("collapse_events", "touched_at", "TEXT")
        self._migrate_add_column("glyphs", "confidence", "REAL NOT NULL DEFAULT 0.5")
        self._migrate_add_column("glyphs", "touched_at", "TEXT")
        self.conn.commit()

    def _migrate_add_column(self, table: str, column: str, col_type: str) -> None:
        """Add a column if it doesn't already exist. Safe to call repeatedly."""
        cursor = self.conn.execute(f"PRAGMA table_info({table})")
        existing = {row[1] for row in cursor.fetchall()}
        if column not in existing:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")

    def status(self) -> dict[str, int]:
        tables = ("canon_documents", "canon_terms", "episodes", "checkpoints", "collapse_events", "glyphs",
                  "kg_entities", "kg_relations")
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
                   density, confidence, lifecycle, touched_at, created_at, consolidated_at
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
            SELECT id, title, content, source_episode_ids_json, phase, keywords_json,
                   joy, density, confidence, touched_at, created_at
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
                   reintegration_notes, confidence, touched_at, created_at, recovered_at
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
            "kg_entities": self.list_kg_entities(),
            "phases": list(PHASE_CHOICES),
            "domains": sorted({row["domain"] for row in self.conn.execute("SELECT DISTINCT domain FROM canon_documents")}),
        }

    def list_kg_entities(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT e.id, e.name, e.kind, e.description, e.confidence, e.created_at,
                   COUNT(r.id) AS relation_count
            FROM kg_entities e
            LEFT JOIN kg_relations r ON r.from_entity_id = e.id
            GROUP BY e.id
            ORDER BY relation_count DESC, e.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

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
        title = sanitize_content(title)
        content = sanitize_content(content)
        normalized_phase = self._normalize_phase(phase)
        tags = tags or []
        joy = compute_joy(delta_coherence, effort)
        density = compute_density(content)
        created_at = utcnow()
        cursor = self.conn.execute(
            """
            INSERT INTO episodes(
                title, content, phase, source, tags_json, delta_coherence, effort, joy, density,
                confidence, lifecycle, touched_at, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                CONFIDENCE_INITIAL,
                EPISODE_LIVE,
                created_at,
                created_at,
            ),
        )
        self.conn.commit()
        result = {
            "episode_id": cursor.lastrowid,
            "phase": normalized_phase,
            "joy": joy,
            "density": density,
            "confidence": CONFIDENCE_INITIAL,
            "lifecycle": EPISODE_LIVE,
            "freshness": 1.0,
            "created_at": created_at,
        }
        self._mirror("episode", {"title": title, **result})
        return result

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
        intent = sanitize_content(intent)
        active_hypothesis = sanitize_content(active_hypothesis)
        next_step = sanitize_content(next_step)
        resume_cue = sanitize_content(resume_cue)
        created_at = utcnow()
        risk_flags = risk_flags or []
        state_snapshot = state_snapshot or {}
        cursor = self.conn.execute(
            """
            INSERT INTO checkpoints(
                task_key, intent, active_hypothesis, next_step, resume_cue,
                risk_flags_json, state_snapshot_json, status, confidence, touched_at, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
            """,
            (
                task_key,
                intent,
                active_hypothesis,
                next_step,
                resume_cue,
                dumps_json(risk_flags),
                dumps_json(state_snapshot),
                CONFIDENCE_INITIAL,
                created_at,
                created_at,
            ),
        )
        self.conn.commit()
        result = {
            "checkpoint_id": cursor.lastrowid,
            "task_key": task_key,
            "created_at": created_at,
            "status": "pending",
            "confidence": CONFIDENCE_INITIAL,
            "freshness": 1.0,
        }
        self._mirror("checkpoint", result)
        return result

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
                checkpoint_id, episode_id, boundary_exceeded, symptoms, recovery_protocol,
                reintegration_notes, confidence, touched_at, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                checkpoint_id,
                episode_id,
                boundary_exceeded,
                symptoms,
                protocol,
                reintegration_notes,
                CONFIDENCE_INITIAL,
                created_at,
                created_at,
            ),
        )
        if checkpoint_id is not None:
            self.resolve_checkpoint(checkpoint_id, status="collapsed")
        self.conn.commit()
        result = {
            "collapse_id": cursor.lastrowid,
            "created_at": created_at,
            "recovery_protocol": protocol,
            "confidence": CONFIDENCE_INITIAL,
            "freshness": 1.0,
        }
        self._mirror("collapse", {"boundary": boundary_exceeded, **result})
        return result

    def affirm(self, kind: str, item_id: int) -> dict[str, Any]:
        """Affirm a memory — boost its confidence and refresh it.

        This is the sovereignty piece: the system suggests, but you decide what stays.
        Affirming a memory says 'this is still true, this still matters.'
        """
        table, id_col = self._resolve_table(kind)
        row = self.conn.execute(f"SELECT * FROM {table} WHERE {id_col} = ?", (item_id,)).fetchone()
        if not row:
            raise ValueError(f"No {kind} with id {item_id}")

        old_confidence = row["confidence"] if "confidence" in row.keys() else CONFIDENCE_INITIAL
        new_confidence = round(min(old_confidence + CONFIDENCE_AFFIRM_BOOST, CONFIDENCE_MAX), 4)
        now = utcnow()
        self.conn.execute(
            f"UPDATE {table} SET confidence = ?, touched_at = ? WHERE {id_col} = ?",
            (new_confidence, now, item_id),
        )

        # If this is an episode and confidence crosses the anchor threshold,
        # mark it as anchored — it stops decaying.
        anchored = False
        if kind == "episode" and new_confidence >= CONFIDENCE_ANCHOR_THRESHOLD:
            old_lifecycle = row["lifecycle"] if "lifecycle" in row.keys() else EPISODE_LIVE
            if old_lifecycle != EPISODE_ANCHORED:
                self.conn.execute(
                    f"UPDATE {table} SET lifecycle = ? WHERE {id_col} = ?",
                    (EPISODE_ANCHORED, item_id),
                )
                anchored = True

        self.conn.commit()
        result = {
            "kind": kind,
            "id": item_id,
            "confidence_before": old_confidence,
            "confidence_after": new_confidence,
            "freshness": compute_freshness(now),
            "touched_at": now,
        }
        if anchored:
            result["lifecycle"] = EPISODE_ANCHORED
            result["anchored"] = True
        return result

    def touch(self, kind: str, item_id: int) -> dict[str, Any]:
        """Touch a memory — reset its freshness clock without changing confidence.

        Like picking up a book you've already read. It doesn't become more true,
        but it moves back to the front of the shelf.
        """
        table, id_col = self._resolve_table(kind)
        row = self.conn.execute(f"SELECT * FROM {table} WHERE {id_col} = ?", (item_id,)).fetchone()
        if not row:
            raise ValueError(f"No {kind} with id {item_id}")

        now = utcnow()
        self.conn.execute(
            f"UPDATE {table} SET touched_at = ? WHERE {id_col} = ?",
            (now, item_id),
        )
        self.conn.commit()
        return {
            "kind": kind,
            "id": item_id,
            "freshness": compute_freshness(now),
            "touched_at": now,
        }

    def promote(self, glyph_id: int, domain: str = "emergent") -> dict[str, Any]:
        """Promote a glyph into a canon document.

        This is the affirmation gate — a glyph has been tested by time and
        affirmed by the user, and now it becomes part of the stable foundation.
        Not every glyph earns this. Only the ones that are still true after
        the moment has passed.

        Requires confidence >= anchor threshold. This can't be forced — it
        must be earned through repeated affirmation.
        """
        row = self.conn.execute("SELECT * FROM glyphs WHERE id = ?", (glyph_id,)).fetchone()
        if not row:
            raise ValueError(f"No glyph with id {glyph_id}")

        confidence = row["confidence"] if "confidence" in row.keys() else 0.5
        if confidence < CONFIDENCE_ANCHOR_THRESHOLD:
            return {
                "promoted": False,
                "reason": f"Confidence {confidence} is below anchor threshold {CONFIDENCE_ANCHOR_THRESHOLD}. Affirm it more first.",
                "glyph_id": glyph_id,
                "confidence": confidence,
            }

        # Create as canon document
        path = f"glyph://{glyph_id}"
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
            (domain, row["title"], path, row["content"], row["content"][:420], created_at),
        )
        doc_row = self.conn.execute("SELECT id FROM canon_documents WHERE path = ?", (path,)).fetchone()
        self.conn.commit()
        return {
            "promoted": True,
            "glyph_id": glyph_id,
            "document_id": doc_row["id"],
            "domain": domain,
            "title": row["title"],
            "confidence": confidence,
        }

    def sweep(self) -> dict[str, Any]:
        """Walk through all live episodes and update their lifecycle.

        - Anchored episodes are left alone (they've been affirmed enough to endure).
        - Live episodes whose freshness has dropped below the fade threshold
          AND whose confidence is below the anchor threshold become 'faded'.
        - Faded episodes that get touched or affirmed later can return to 'live'.

        This is not deletion. Fading is a quieting — the memory moves to the
        back of the shelf but stays on it. Think of it as the difference
        between forgetting and letting go of the foreground.
        """
        rows = self.conn.execute(
            "SELECT id, confidence, lifecycle, touched_at, created_at FROM episodes WHERE lifecycle = ?",
            (EPISODE_LIVE,),
        ).fetchall()

        faded_ids: list[int] = []
        for row in rows:
            anchor = row["touched_at"] or row["created_at"]
            freshness = compute_freshness(anchor)
            confidence = row["confidence"] or CONFIDENCE_INITIAL

            if confidence >= CONFIDENCE_ANCHOR_THRESHOLD:
                # Promote to anchored — this episode has proven itself
                self.conn.execute(
                    "UPDATE episodes SET lifecycle = ? WHERE id = ?",
                    (EPISODE_ANCHORED, row["id"]),
                )
            elif freshness < FRESHNESS_FADE_THRESHOLD:
                # Fade — this episode hasn't been touched and hasn't been affirmed
                self.conn.execute(
                    "UPDATE episodes SET lifecycle = ? WHERE id = ?",
                    (EPISODE_FADED, row["id"]),
                )
                faded_ids.append(row["id"])

        self.conn.commit()

        # Count current states for the report
        counts = {}
        for state in (EPISODE_LIVE, EPISODE_FADED, EPISODE_ANCHORED):
            counts[state] = self.conn.execute(
                "SELECT COUNT(*) AS c FROM episodes WHERE lifecycle = ?", (state,)
            ).fetchone()["c"]

        return {
            "swept": len(rows),
            "newly_faded": len(faded_ids),
            "faded_ids": faded_ids,
            "totals": counts,
        }

    def _resolve_table(self, kind: str) -> tuple[str, str]:
        """Map a memory kind to its table name and id column."""
        mapping = {
            "episode": ("episodes", "id"),
            "checkpoint": ("checkpoints", "id"),
            "collapse": ("collapse_events", "id"),
            "glyph": ("glyphs", "id"),
        }
        if kind not in mapping:
            raise ValueError(f"Unknown kind: {kind}. Use one of: {', '.join(mapping)}")
        return mapping[kind]

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

        # Glyphs inherit the average confidence of their source episodes,
        # with a floor of 0.5 — condensation itself is an act of affirmation.
        conf_values = [row["confidence"] for row in rows if row["confidence"] is not None]
        glyph_confidence = max(sum(conf_values) / len(conf_values), 0.5) if conf_values else 0.5
        now = utcnow()
        cursor = self.conn.execute(
            """
            INSERT INTO glyphs(title, content, source_episode_ids_json, phase, keywords_json,
                               joy, density, confidence, touched_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                title,
                content,
                dumps_json([row["id"] for row in rows]),
                phase,
                dumps_json(keywords),
                joy,
                density,
                round(glyph_confidence, 4),
                now,
                now,
            ),
        )
        episode_ids = [row["id"] for row in rows]
        placeholders = ", ".join("?" for _ in episode_ids)
        self.conn.execute(
            f"UPDATE episodes SET consolidated_at = ? WHERE id IN ({placeholders})",
            [utcnow(), *episode_ids],
        )
        self.conn.commit()
        result = {
            "glyph_id": cursor.lastrowid,
            "episode_count": len(rows),
            "title": title,
            "phase": phase,
            "keywords": keywords,
            "joy": joy,
            "density": density,
        }
        self._mirror("glyph", result)
        return result

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
        results.extend(self._query_kg(query_tokens))

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

    def _vitals_multiplier(self, row: sqlite3.Row) -> float:
        """Blend freshness and confidence into a score multiplier.

        A brand-new, unaffirmed memory gets: freshness 1.0, confidence 0.3 → ~0.65x
        A week-old, affirmed memory gets:     freshness 0.5, confidence 0.8 → ~0.65x
        A fresh, affirmed memory gets:        freshness 1.0, confidence 1.0 → 1.0x
        A stale, unaffirmed memory gets:      freshness 0.1, confidence 0.3 → ~0.2x
        An anchored memory:                   confidence dominates, freshness matters less

        Floor of 0.15 so nothing disappears entirely — it just gets quieter.
        """
        vitals = self._vitals(row)

        # Anchored memories lean on confidence, not freshness — they've earned persistence.
        lifecycle = row["lifecycle"] if "lifecycle" in row.keys() else EPISODE_LIVE
        if lifecycle == EPISODE_ANCHORED:
            blend = (vitals["freshness"] * 0.2) + (vitals["confidence"] * 0.8)
        elif lifecycle == EPISODE_FADED:
            # Faded memories are very quiet — but not silent
            blend = (vitals["freshness"] * 0.5) + (vitals["confidence"] * 0.5)
            return max(blend * 0.3, 0.05)
        else:
            blend = (vitals["freshness"] * 0.5) + (vitals["confidence"] * 0.5)

        return max(blend, 0.15)

    def _query_glyphs(self, query_tokens: set[str], phase: str | None) -> list[MemoryResult]:
        rows = self.conn.execute("SELECT * FROM glyphs ORDER BY created_at DESC").fetchall()
        results: list[MemoryResult] = []
        for row in rows:
            score = self._token_score(query_tokens, f"{row['title']} {row['content']} {row['keywords_json']}")
            score *= self._phase_multiplier(row["phase"], phase)
            score *= self._vitals_multiplier(row)
            if score <= 0:
                continue
            vitals = self._vitals(row)
            results.append(
                MemoryResult(
                    kind="glyph",
                    score=score + 0.2,
                    title=row["title"],
                    body=row["content"],
                    phase=row["phase"],
                    metadata={
                        "keywords": loads_json(row["keywords_json"], []),
                        "freshness": vitals["freshness"],
                        "confidence": vitals["confidence"],
                    },
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
            score *= self._vitals_multiplier(row)
            if score <= 0:
                continue
            vitals = self._vitals(row)
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
                        "freshness": vitals["freshness"],
                        "confidence": vitals["confidence"],
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
            score *= self._vitals_multiplier(row)
            if score <= 0:
                continue
            vitals = self._vitals(row)
            results.append(
                MemoryResult(
                    kind="collapse",
                    score=score,
                    title=f"Collapse at {row['boundary_exceeded']}",
                    body=row["recovery_protocol"],
                    phase="reintegration",
                    metadata={
                        "collapse_id": row["id"],
                        "freshness": vitals["freshness"],
                        "confidence": vitals["confidence"],
                    },
                )
            )
        return results

    def _query_kg(self, query_tokens: set[str]) -> list[MemoryResult]:
        rows = self.conn.execute("SELECT * FROM kg_entities").fetchall()
        results: list[MemoryResult] = []
        for row in rows:
            score = self._token_score(query_tokens, f"{row['name']} {row['description']}")
            if score <= 0:
                continue
            # Build a brief body from outbound relations
            neighbors = self.kg_neighbors(row["name"], limit=4)
            neighbor_text = ", ".join(
                f"{n['relation']} {n['name']}" for n in neighbors.get("outbound", [])[:3]
            )
            body = row["description"] or neighbor_text or row["name"]
            results.append(
                MemoryResult(
                    kind="entity",
                    score=score,
                    title=row["name"],
                    body=body[:240],
                    metadata={
                        "kind": row["kind"],
                        "outbound_count": len(neighbors.get("outbound", [])),
                    },
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

    def _vitals(self, row: sqlite3.Row) -> dict[str, float]:
        """Compute the live freshness and return it alongside stored confidence."""
        touched = row["touched_at"] if "touched_at" in row.keys() else None
        anchor = touched or row["created_at"]
        confidence = row["confidence"] if "confidence" in row.keys() else CONFIDENCE_INITIAL
        return {
            "freshness": compute_freshness(anchor),
            "confidence": confidence or CONFIDENCE_INITIAL,
        }

    def _checkpoint_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        vitals = self._vitals(row)
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
            "freshness": vitals["freshness"],
            "confidence": vitals["confidence"],
            "created_at": row["created_at"],
            "resolved_at": row["resolved_at"],
        }

    def _episode_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        vitals = self._vitals(row)
        lifecycle = row["lifecycle"] if "lifecycle" in row.keys() else EPISODE_LIVE
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
            "freshness": vitals["freshness"],
            "confidence": vitals["confidence"],
            "lifecycle": lifecycle,
            "created_at": row["created_at"],
            "consolidated_at": row["consolidated_at"],
        }

    def _glyph_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        vitals = self._vitals(row)
        return {
            "glyph_id": row["id"],
            "title": row["title"],
            "content": row["content"],
            "source_episode_ids": loads_json(row["source_episode_ids_json"], []),
            "phase": row["phase"],
            "keywords": loads_json(row["keywords_json"], []),
            "joy": row["joy"],
            "density": row["density"],
            "freshness": vitals["freshness"],
            "confidence": vitals["confidence"],
            "created_at": row["created_at"],
        }

    def _collapse_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        vitals = self._vitals(row)
        return {
            "collapse_id": row["id"],
            "checkpoint_id": row["checkpoint_id"],
            "episode_id": row["episode_id"],
            "boundary_exceeded": row["boundary_exceeded"],
            "symptoms": row["symptoms"],
            "recovery_protocol": row["recovery_protocol"],
            "reintegration_notes": row["reintegration_notes"],
            "freshness": vitals["freshness"],
            "confidence": vitals["confidence"],
            "created_at": row["created_at"],
            "recovered_at": row["recovered_at"],
        }

    # ── JSONL mirror ───────────────────────────────────────────────────────────
    # Adapted from OpenFang's session JSONL backup: a human-readable append-only
    # audit trail that mirrors every write to a newline-delimited JSON file.
    # The DB is the truth; the mirror is for inspection, grep, and debugging.

    def _mirror(self, event_type: str, data: dict[str, Any]) -> None:
        """Append one JSON line to the mirror file, if one is configured."""
        if self.mirror_path is None:
            return
        self.mirror_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"event": event_type, "ts": utcnow(), **data}, ensure_ascii=True, sort_keys=True)
        with self.mirror_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    # ── Knowledge graph ────────────────────────────────────────────────────────
    # Adapted from OpenFang's openfang-memory knowledge graph layer.
    # Entities are named concepts; relations are typed, weighted edges between them.
    # Auto-extraction from episodes uses the existing canon term index as a
    # controlled vocabulary — no NLP library needed, everything stays inspectable.

    def kg_add_entity(
        self,
        name: str,
        kind: str = "concept",
        description: str = "",
    ) -> dict[str, Any]:
        """Upsert a named entity. Existing descriptions are preserved if the new one is empty."""
        now = utcnow()
        self.conn.execute(
            """
            INSERT INTO kg_entities(name, kind, description, confidence, touched_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                kind = excluded.kind,
                description = CASE WHEN excluded.description != '' THEN excluded.description
                                   ELSE kg_entities.description END,
                touched_at = excluded.touched_at
            """,
            (name, kind, description, CONFIDENCE_INITIAL, now, now),
        )
        row = self.conn.execute("SELECT id FROM kg_entities WHERE name = ?", (name,)).fetchone()
        self.conn.commit()
        result = {"entity_id": row["id"], "name": name, "kind": kind}
        self._mirror("kg_entity", result)
        return result

    def kg_link(
        self,
        from_name: str,
        relation: str,
        to_name: str,
        weight: float = 1.0,
    ) -> dict[str, Any]:
        """Create a directed, typed edge between two entities (creating them if needed)."""
        self.kg_add_entity(from_name)
        self.kg_add_entity(to_name)
        from_row = self.conn.execute("SELECT id FROM kg_entities WHERE name = ?", (from_name,)).fetchone()
        to_row = self.conn.execute("SELECT id FROM kg_entities WHERE name = ?", (to_name,)).fetchone()
        now = utcnow()
        self.conn.execute(
            """
            INSERT INTO kg_relations(from_entity_id, to_entity_id, relation, weight, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(from_entity_id, to_entity_id, relation) DO UPDATE SET weight = excluded.weight
            """,
            (from_row["id"], to_row["id"], relation, weight, now),
        )
        self.conn.commit()
        result = {"from": from_name, "relation": relation, "to": to_name, "weight": weight}
        self._mirror("kg_relation", result)
        return result

    def kg_neighbors(self, entity_name: str, limit: int = 10) -> dict[str, Any]:
        """Return outbound and inbound relations for a named entity."""
        entity = self.conn.execute("SELECT * FROM kg_entities WHERE name = ?", (entity_name,)).fetchone()
        if not entity:
            return {"entity": entity_name, "found": False, "outbound": [], "inbound": []}
        outbound = self.conn.execute(
            """
            SELECT e.name, r.relation, r.weight
            FROM kg_relations r
            JOIN kg_entities e ON e.id = r.to_entity_id
            WHERE r.from_entity_id = ?
            ORDER BY r.weight DESC
            LIMIT ?
            """,
            (entity["id"], limit),
        ).fetchall()
        inbound = self.conn.execute(
            """
            SELECT e.name, r.relation, r.weight
            FROM kg_relations r
            JOIN kg_entities e ON e.id = r.from_entity_id
            WHERE r.to_entity_id = ?
            ORDER BY r.weight DESC
            LIMIT ?
            """,
            (entity["id"], limit),
        ).fetchall()
        return {
            "entity": entity_name,
            "kind": entity["kind"],
            "description": entity["description"],
            "found": True,
            "outbound": [{"name": r["name"], "relation": r["relation"], "weight": r["weight"]} for r in outbound],
            "inbound": [{"name": r["name"], "relation": r["relation"], "weight": r["weight"]} for r in inbound],
        }

    def kg_extract_from_episode(self, episode_id: int) -> dict[str, Any]:
        """Auto-extract entities from an episode using the canon term index.

        Any canon term found in the episode text becomes an entity.
        Terms in the same domain get a 'co-occurs' edge between them.
        Every matched term also gets an 'appears-in-phase' edge to the episode's phase.

        This uses the existing controlled vocabulary — no ML model required.
        The result is always legible and re-runnable.
        """
        row = self.conn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        if not row:
            raise ValueError(f"No episode with id {episode_id}")
        content_tokens = set(tokenize(f"{row['title']} {row['content']}"))

        # Find canon terms present in episode content
        terms = self.conn.execute(
            """
            SELECT canon_terms.term, canon_terms.definition, canon_documents.domain
            FROM canon_terms
            JOIN canon_documents ON canon_documents.id = canon_terms.document_id
            """
        ).fetchall()
        matched: list[dict[str, str]] = []
        for term_row in terms:
            term_tokens = set(tokenize(term_row["term"]))
            if term_tokens and term_tokens.issubset(content_tokens):
                self.kg_add_entity(name=term_row["term"], kind="term", description=term_row["definition"])
                matched.append({"name": term_row["term"], "domain": term_row["domain"]})

        # Link co-occurring terms within the same domain
        by_domain: dict[str, list[str]] = {}
        for m in matched:
            by_domain.setdefault(m["domain"], []).append(m["name"])
        links_created = 0
        for domain_names in by_domain.values():
            for i, a in enumerate(domain_names):
                for b in domain_names[i + 1 :]:
                    self.kg_link(a, "co-occurs", b, weight=0.5)
                    links_created += 1

        # Tag each entity with the episode's phase
        phase = row["phase"]
        for m in matched:
            self.kg_link(m["name"], "appears-in-phase", phase, weight=1.0)

        result = {
            "episode_id": episode_id,
            "entities_found": len(matched),
            "entities": [m["name"] for m in matched],
            "links_created": links_created,
        }
        self._mirror("kg_extract", result)
        return result

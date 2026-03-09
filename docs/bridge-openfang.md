# Threshold Memory × OpenFang: A bridge proposal

This document describes how Threshold Memory and OpenFang can enrich each other.
It is written as a contribution proposal for OpenFang — a record of what each
system does distinctively, and where the seam between them is most productive.

---

## What each system does

**OpenFang** (`github.com/RightNow-AI/openfang`) is a production-grade Agent
Operating System written in Rust. It handles LLM orchestration, tool execution in
a WASM sandbox, 40 messaging adapters, 27 LLM providers, cross-channel canonical
sessions, and a three-layer memory substrate (structured SQLite, semantic search,
knowledge graph). Its consolidation engine is explicitly labeled Phase 1 — it
decays confidence over time but does not yet merge or condense memories.

**Threshold Memory** (`threshold_memory/`) is an inspectable Python prototype
focused on continuity at interruption boundaries. It treats approval gates,
failures, and interrupted runs as first-class memory events rather than edge
cases to suppress.

---

## Where Threshold Memory fills OpenFang's gaps

### 1. Structured approval-gate checkpoints

OpenFang persists sessions to SQLite and JSONL. When an agent pauses at an
approval gate, the session is there — but nothing marks the pause structurally.
The agent must reconstruct its intent from raw message history when it resumes.

Threshold Memory's checkpoint is a six-field state-preservation record:

| Field | Purpose |
|---|---|
| `intent` | Why this task exists |
| `active_hypothesis` | What the agent currently believes is true |
| `next_step` | The concrete action to take when resumed |
| `resume_cue` | A terse instruction to the resuming agent |
| `risk_flags` | What could go wrong during the wait |
| `state_snapshot` | Arbitrary JSON of live state at pause time |

This makes interruption legible. The resuming agent reads a resume cue rather
than re-reading the whole conversation.

**Proposed integration point:** OpenFang's `agent_loop.rs` fires a
`BeforePromptBuild` hook before each LLM call. An approval-gate hook could
call a checkpoint write before yielding control, and resume by injecting the
resume cue into the next prompt.

### 2. Collapse events: failures as learnable memory

OpenFang's `session_repair.rs` is excellent at fixing structural corruption
(orphaned tool results, role alternation violations, prompt injection). It
detects that something went wrong and repairs the message history.

What it does not do is record *what* went wrong as a learnable event. The
failure is corrected and discarded.

Threshold Memory's collapse event captures:

- `boundary_exceeded` — what the agent hit
- `symptoms` — what the failure looked like
- `recovery_protocol` — a reintegration script for the next run
- `reintegration_notes` — free-form analysis

Collapse events are stored, retrieved, and weighed in future queries. An agent
querying memory before resuming a task gets collapse events from similar past
runs in its context — which is a much cheaper signal than repeating the failure.

**Proposed integration point:** `session_repair.rs` already tracks `RepairStats`
(orphaned results removed, empty messages eliminated, etc.). When `RepairStats`
shows non-trivial repair (e.g. synthetic insertions > 0), emit a collapse event
to the memory layer with the repair stats as symptoms.

### 3. Phase-aware retrieval

OpenFang retrieves memories by vector similarity (semantic store) or LIKE
matching (structured store). Both treat all memories as equally relevant to the
current moment.

Threshold Memory uses a five-phase state model:

```
symmetry → tension → break → novelty → reintegration
```

Each phase has a relatedness matrix to the others. A query issued during
`break` (task interrupted, agent paused) weights memories from prior `break`
and `reintegration` phases more heavily than memories from `symmetry`. This
surfaces the most contextually appropriate memories rather than the most
semantically similar ones.

**Proposed integration point:** The `Memory` trait in `openfang-memory` could
accept an optional `phase_hint` parameter. Phase-aware retrieval runs as a
post-filter or score modifier after semantic search, requiring no changes to
the vector backend.

### 4. Glyph condensation (OpenFang's Phase 2)

OpenFang's `consolidation.rs` states explicitly:

```rust
memories_merged: 0, // Phase 1: no merging
```

The planned Phase 2 is merging similar memories. Threshold Memory's glyph
condensation is a working reference implementation:

- Collects N unconsolidated episodes
- Extracts dominant phase, top keywords, joy/density signals
- Produces a glyph — a dense, high-signal summary stored as a first-class
  memory type with its own confidence and freshness
- Glyphs can be promoted into the canonical knowledge base after user affirmation

Glyphs are not lossy summaries. The source episode ids are preserved, so the
glyph can be traced back to its origin material.

**Proposed integration point:** Implement glyph consolidation in
`openfang-memory/src/consolidation.rs` as Phase 2. The Threshold Memory
Python implementation is a direct reference. The key additions are:

- A `glyphs` table (title, content, keywords, phase, source_ids, confidence)
- A consolidation pass that groups episodes by time window and dominant phase
- Promotion path from glyph → canonical knowledge base entry

### 5. Affirmation-based confidence anchoring

OpenFang's consolidation decays confidence over time based on `accessed_at`.
This is a useful signal, but it treats all memories as equally erasable. Some
memories should become permanent — not because they are accessed frequently,
but because a human or agent has affirmed them as true.

Threshold Memory's affirmation model:

- Each memory starts at `confidence: 0.3`
- Each affirmation adds `+0.25`, capped at `1.0`
- Once confidence reaches `0.8` (the anchor threshold), the memory stops decaying
- Anchored memories are retrievable indefinitely, regardless of access frequency

This gives the user sovereignty over the memory: the system suggests decay,
but a human can say "this stays."

**Proposed integration point:** Add an `affirm(memory_id)` endpoint to
OpenFang's REST API (it already has 140+ endpoints). Affirmation sets
`confidence = min(confidence + boost, 1.0)` and records an `affirmed_at`
timestamp. The consolidation pass skips anchored memories.

---

## What OpenFang brings to Threshold Memory

This prototype has adopted four patterns from OpenFang that strengthen it:

### Cross-channel canonical sessions

OpenFang's canonical session layer merges memories from Telegram, Discord, Slack,
and 37 other channels into a single cross-channel narrative. Threshold Memory's
`source` field on episodes is the start of this — it tracks where a memory came
from. The natural next step is a canonical session concept that groups episodes
from different sources into one coherent thread.

### Knowledge graph

OpenFang's knowledge graph (entities + typed relations) is a step beyond
episodic storage. Threshold Memory now has a minimal knowledge graph layer
(added in this prototype pass): named entities, typed weighted edges, and
auto-extraction from episodes using the canonical term index. The design is
deliberately inspectable — no ML model required, everything is legible SQL.

### JSONL mirror

OpenFang mirrors sessions to disk in JSONL format for human-readable inspection.
Threshold Memory now does the same: every write event (episode, checkpoint,
collapse, glyph) is appended to a configurable JSONL file. The DB is the truth;
the mirror is for grep, audit, and debugging.

### Prompt injection sanitization

OpenFang's `session_repair.rs` strips injection markers (`<|im_start|>`,
`IGNORE PREVIOUS INSTRUCTIONS`) and oversized base64 blobs from tool results
before they enter the conversation. Threshold Memory now applies the same
sanitization to all incoming episode content and checkpoint fields — before
anything is written to the store.

---

## Proposed contribution to OpenFang

The most valuable contribution from this prototype to OpenFang is a pull request
that adds:

1. A `checkpoints` table to `openfang-memory` with the six-field schema
2. A `collapse_events` table with boundary/symptoms/recovery fields
3. A `save_checkpoint()` call in `agent_loop.rs` at approval gate boundaries
4. A collapse record call in `session_repair.rs` when non-trivial repair occurs
5. A `glyphs` table and Phase 2 consolidation pass
6. An `affirm(memory_id)` REST endpoint

Items 1–4 require no changes to the WASM sandbox, LLM drivers, or channel
adapters — they are purely additive to the memory layer.

Items 5–6 replace the no-op Phase 2 stub in consolidation with a working
implementation.

---

## Reference implementation

This prototype is the reference. Everything is:

- Runnable locally with no external dependencies (SQLite + Python stdlib)
- Covered by 43 tests (engine, server, sanitization, KG, mirror)
- Inspectable at every layer (the console at `threshold_memory/server.py`)
- Small enough to read in an afternoon

The design philosophy is deliberate: legibility first, embeddings later only if
they don't break inspection and control.

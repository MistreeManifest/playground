# Threshold Memory Prototype

This project is a local, inspectable prototype for a paradox-tolerant memory layer.

It is built around four motions:

- remember: keep a canonical corpus and episodic history
- checkpoint: persist task state before permission or approval waits
- recover: treat failed runs as collapse events with a reintegration protocol
- condense: turn diffuse episodes into dense memory glyphs

The prototype is intentionally small. It uses SQLite and the Python standard library only.

## Why this exists

Most agent memory stacks are good at storage and weak at continuity.

They can keep notes, but they often lose the thread at the exact places where continuity matters most:

- approval gates
- long waits
- interrupted runs
- failures that should become learning instead of discard

This prototype treats those moments as first-class memory events.

## Threshold Console

There is now a small local web console for watching the memory loop move in real time.

Run it with:

```powershell
python -m threshold_memory.server --db data/threshold-memory.sqlite3 --host 127.0.0.1 --port 8765
```

Then open:

```text
http://127.0.0.1:8765
```

The console gives you:

- a canon panel for the seeded source corpus
- a live episode stream
- a checkpoint panel for permission waits
- a collapse journal with recovery protocols
- a glyph panel for condensed patterns
- query and demo controls for feeling the loop end to end

## Design layers

- `threshold`: interaction protocol and translation layer
- `lifestory`: boundary model and regulation-aware memory
- `muzick`: compression layer for dense, high-signal phrasing
- `liberation`: ethic of constriction, where lower effort and higher coherence matter
- `opte`: controller for phases, collapse recovery, and state sovereignty

## Project structure

- `threshold_memory/engine.py`: SQLite schema and retrieval logic
- `threshold_memory/cli.py`: command line interface
- `threshold_memory/server.py`: lightweight local HTTP server for the console
- `threshold_memory/web/`: static frontend for the Threshold Console
- `seeds/mistree_corpus.json`: optional seed config for the provided markdown corpus
- `tests/test_engine.py`: basic coverage for seeding, checkpointing, collapse logging, consolidation, and retrieval

## Quick start

Initialize the database:

```powershell
python -m threshold_memory.cli init --db data/threshold-memory.sqlite3
```

Seed the provided corpus:

```powershell
python -m threshold_memory.cli seed-config --db data/threshold-memory.sqlite3 seeds/mistree_corpus.json
```

Store an episode:

```powershell
python -m threshold_memory.cli episode `
  --db data/threshold-memory.sqlite3 `
  --title "Permission wait drift" `
  --content "The agent lost the next action after waiting on approval." `
  --phase break `
  --delta-coherence 2 `
  --effort 1
```

Create a checkpoint before a permission gate:

```powershell
python -m threshold_memory.cli checkpoint `
  --db data/threshold-memory.sqlite3 `
  --task-key approval-loop `
  --intent "Preserve state before asking for permission." `
  --hypothesis "Explicit resume cues reduce drift." `
  --next-step "Resume from the saved cue once approval arrives." `
  --resume-cue "Return to the saved next step and do not re-scan the whole task." `
  --risk state-loss
```

Resume the latest pending checkpoint:

```powershell
python -m threshold_memory.cli resume --db data/threshold-memory.sqlite3
```

Record a collapse:

```powershell
python -m threshold_memory.cli collapse `
  --db data/threshold-memory.sqlite3 `
  --checkpoint-id 1 `
  --boundary "approval gate" `
  --symptoms "lost next action after waiting"
```

Condense recent episodes into a glyph:

```powershell
python -m threshold_memory.cli consolidate --db data/threshold-memory.sqlite3 --limit 5
```

Query memory with optional phase bias:

```powershell
python -m threshold_memory.cli query `
  --db data/threshold-memory.sqlite3 `
  "resume permission reintegration" `
  --phase break `
  --limit 5
```

## Retrieval model

Retrieval is intentionally simple:

- lexical overlap
- domain hint bonuses
- phase-aware weighting across `symmetry`, `tension`, `break`, `novelty`, and `reintegration`

The goal here is not perfect semantic search. The goal is legibility and a structure that another agent stack can adapt.

## Notes

- `seed-config` will ingest the supplied markdown corpus directly from `C:\Users\Mistree\Documents\...`
- map files contribute indexed terms and definitions
- checkpoints are stored separately from episodes so waiting does not blur into narrative memory
- collapse events capture the boundary exceeded and a recovery protocol

## Next useful evolutions

- add embeddings only if inspection and control remain intact
- attach source links and freshness metadata to every retrieved item
- let the consolidator promote only user-affirmed truths into canonical memory
- plug the checkpoint flow into an actual agent runtime

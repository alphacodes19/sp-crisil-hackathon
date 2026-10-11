"""
Minimal read-only signal API.

Endpoints (Section 14 of the master context):
    GET /health
    GET /signals
    GET /signals/{ticker}
    GET /events/{event_id}

Design notes:
- This module is purely a query layer over the already-persisted
  data/signals.jsonl. It has NO dependency on entity resolution,
  sentiment, event classification, or any Hugging Face model - so
  importing/starting the API never triggers a model download. Signals
  must already exist on disk (see scripts/generate_replay_signals.py).
- `SIGNALS_PATH` is a module-level variable (not baked into closures),
  specifically so tests can point it at a temporary file via
  monkeypatching (`monkeypatch.setattr(main, "SIGNALS_PATH", tmp)`)
  without needing a real pipeline run or any app/dependency-injection
  machinery.
- A missing signals file is treated the same as an empty one: `/signals`
  and `/signals/{ticker}` return `[]`, not an error - per requirement
  to "handle a missing or empty JSONL file gracefully".
- Reading `/signals/{ticker}` for a ticker with zero matches is a
  normal, valid outcome (empty list, HTTP 200) - it is NOT a 404. A 404
  is reserved for `/events/{event_id}` when that specific event_id does
  not exist, which IS an error condition for that endpoint.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException

from src.impact.signal_builder import DEFAULT_SIGNALS_PATH, Signal
from src.utils.io import read_jsonl

app = FastAPI(title="SP & CRISIL Hackathon - Risk Signal API", version="0.1.0")

# Mutable on purpose - see module docstring ("Design notes").
SIGNALS_PATH: Path = DEFAULT_SIGNALS_PATH


def _load_signals() -> List[Signal]:
    path = SIGNALS_PATH
    if not Path(path).exists():
        return []
    try:
        return list(read_jsonl(path, Signal))
    except Exception as exc:  # noqa: BLE001 - a corrupt signals file is a server-side
        # problem, not a client error; surface it as 500 rather than crashing
        # the process or silently returning an empty/partial list.
        raise HTTPException(
            status_code=500, detail=f"Failed to read signals file: {exc}"
        ) from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/signals", response_model=List[Signal])
def get_signals() -> List[Signal]:
    return _load_signals()


@app.get("/signals/{ticker}", response_model=List[Signal])
def get_signals_by_ticker(ticker: str) -> List[Signal]:
    if not ticker.strip():
        raise HTTPException(status_code=422, detail="ticker must not be empty")
    ticker_upper = ticker.strip().upper()
    signals = _load_signals()
    return [s for s in signals if any(e.ticker.upper() == ticker_upper for e in s.entities)]


@app.get("/events/{event_id}", response_model=Signal)
def get_event(event_id: str) -> Signal:
    if not event_id.strip():
        raise HTTPException(status_code=422, detail="event_id must not be empty")
    signals = _load_signals()
    match = next((s for s in signals if s.event_id == event_id), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"No signal found for event_id '{event_id}'")
    return match

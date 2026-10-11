# Event-Driven Risk Intelligence & Portfolio Stress Testing

S&P Global & CRISIL Campus Hackathon 2026 submission. Converts
unstructured news/filings into explainable risk signals, then
(Module B) shows their effect on a synthetic wholesale portfolio.

**187 automated tests pass** in this build (see Testing below).
Everything that follows is reproducible from a clean checkout with the
exact commands given.

## Requirements

- Python 3.11+ (built/tested on 3.12)
- `pip install -r requirements.txt`
  - Core (pydantic, requests, pytest), API (fastapi, uvicorn, httpx),
    and dashboard (streamlit, plotly) deps are lightweight and
    sufficient to **run the full test suite** and the API/dashboard
    against already-generated signals.
  - `transformers`/`torch` are also listed but are only needed to
    generate signals from real model inference (see "Generate
    signals" below) - the test suite itself injects stub inference
    and never needs them installed.

## Install

```bash
git clone <this-repo>
cd sp-crisil-hackathon
python3 -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

## Run the tests

```bash
pytest tests/ -v
```

Expect **187 passed**. The suite never downloads model weights or
calls a live API - every ML component is exercised through injected
stub inference functions (see each module's docstring).

## Generate signals (replay mode - no live news/model API calls at demo time)

Requires `transformers`/`torch` installed and, the first time,
internet access to download FinBERT + the zero-shot model's weights.
Run this **once, ahead of time**, not during the live demo:

```bash
python scripts/generate_replay_signals.py            # real GDELT + SEC replay -> data/signals.jsonl
python scripts/generate_synthetic_demo_signals.py    # SYNTHETIC fixture only -> data/signals_synthetic_demo.jsonl
```

These write two **separate** files on purpose - the synthetic,
entirely fictional demo fixture is never merged into the real-source
replay file. The dashboard only loads the synthetic file when you
explicitly tick its sidebar checkbox.

(`data/portfolio/portfolio.csv` and `data/scenarios/scenario_library.json`
are already committed; regenerate the portfolio only if you want to,
via `python scripts/generate_portfolio.py` - it's a fixed-seed
deterministic generator, so re-running it reproduces the same file.)

## Run the API

```bash
uvicorn src.api.main:app --reload
```

Endpoints: `GET /health`, `GET /signals`, `GET /signals/{ticker}`,
`GET /events/{event_id}`. Reads `data/signals.jsonl` only - starting
the API never triggers a model download.

## Run the dashboard (Module B)

```bash
streamlit run dashboard/app.py
```

Reads persisted signals + `data/portfolio/portfolio.csv` +
`data/scenarios/scenario_library.json` only; never loads an ML model.
Tick "Include synthetic high-impact demo fixture" in the sidebar to
also see the fictional ZZDEMO scenario.

## Datasets - what's real vs. synthetic

| Data | Status | Location |
|---|---|---|
| GDELT news articles | Real, public, hand-captured for deterministic replay | `data/replay/gdelt/sample.jsonl` |
| SEC 8-K filings | Real, public, hand-captured for deterministic replay | `data/replay/sec/sample.jsonl` |
| High-impact demo fixture | **Entirely fictional** - fake company "ZZDEMO", fake `.test`-domain publishers, every document body states "SYNTHETIC...NOT REAL NEWS" | `data/replay/synthetic/high_impact_demo.jsonl` (loader: `src/ingestion/synthetic_demo.py`) |
| Entity alias table | Real company names/tickers (public), plus one clearly-labeled fictional entry (ZZDEMO) | `data/entities/alias_table.csv` |
| Portfolio | **Entirely synthetic positions** on real company *names* (used only as realistic labels) plus the fictional ZZDEMO issuer; deterministic fixed-seed generator | `data/portfolio/portfolio.csv` (generator: `scripts/generate_portfolio.py`) |
| Scenario library | Stylized, documented assumptions - **not** Basel-compliant, empirically calibrated, or regulatory-prescribed | `data/scenarios/scenario_library.json` |

The synthetic fixture/portfolio are never silently mixed into the
ordinary real-source replay: they live in separate files, separate
signal files, and are only loaded by the dashboard on an explicit,
off-by-default toggle.

## What's provisional, heuristic, or unverified

- **Impact score** (1-10): a documented, weighted heuristic
  (`src/impact/impact_engine.py`) - explicitly *not* calibrated, not a
  probability.
- **Event resolution**: conservative deterministic clustering
  (`src/events/resolution.py`) - may miss some true duplicates by
  design (precision over recall), not an embedding-based method.
- **Exposure targeting / scenario shocks** (`src/stress/stress_engine.py`,
  `data/scenarios/scenario_library.json`): stylized assumptions, stated
  as such in-file.
- **Real model inference**: FinBERT/zero-shot classification and
  long-running server processes (`uvicorn`, `streamlit run`) were not
  executable in this project's network-restricted build sandbox.
  Verified instead via dependency-injected stub inference (unit/
  integration tests) and in-process test harnesses (FastAPI
  `TestClient`, Streamlit `AppTest`) - run the two "Generate signals"
  commands above on a normal machine with internet before relying on
  real inference.

## Repository layout

```
src/            ingestion, entity resolution, NLP, event resolution,
                impact engine, signal builder, API, portfolio stress engine
data/           replay fixtures, entity table, portfolio, scenarios
scripts/        deterministic generators + CLI entry points
dashboard/      Streamlit + Plotly app (Module B)
tests/          187 tests
docs/           architecture.png, presentation.pdf, demo_script.md
```

## Scope

Built: ingestion -> entity resolution -> sentiment + event
classification -> event resolution -> impact score -> exposure
mapping -> portfolio stress test -> dashboard.

Not built (explicitly out of scope this phase): Module A (tactical
rebalancing), VaR, reverse stress testing, advanced derivatives
pricing, Docker/distributed infrastructure.

## Reference: https://chatgpt.com/c/6ac75dbf-a22c-83e8-9dda-2406ec0561f2 
## https://claude.ai/chat/4722a3a2-0e05-4504-8f91-d22b3bbc51e4


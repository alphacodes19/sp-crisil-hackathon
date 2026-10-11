import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.entities.resolver import EntityResolver, resolve_entities
from src.utils.schema import UnifiedDocument


def _doc(title: str, text: str) -> UnifiedDocument:
    return UnifiedDocument(
        document_id="test-doc-1",
        source_type="news",
        source_name="GDELT",
        publisher="test",
        timestamp="2026-10-08T10:00:00Z",
        title=title,
        text=text,
        url=None,
    )


def test_matches_ticker_symbol():
    doc = _doc("Chipmaker surges", "NVDA shares rallied after the announcement.")
    matches = resolve_entities(doc)
    assert len(matches) == 1
    m = matches[0]
    assert m.canonical_name == "NVIDIA Corporation"
    assert m.ticker == "NVDA"
    assert m.match_type == "ticker"


def test_matches_exact_canonical_name():
    doc = _doc("Filing update", "Tesla Inc. announced a new credit facility today.")
    matches = resolve_entities(doc)
    assert len(matches) == 1
    m = matches[0]
    assert m.canonical_name == "Tesla Inc."
    assert m.ticker == "TSLA"
    assert m.match_type == "exact_name"


def test_matches_known_alias():
    doc = _doc("Tech news", "Google announced a new data center investment.")
    matches = resolve_entities(doc)
    assert len(matches) == 1
    m = matches[0]
    assert m.canonical_name == "Alphabet Inc."
    assert m.ticker == "GOOGL"
    assert m.match_type == "alias"
    assert m.matched_alias == "google"


def test_matches_multiple_distinct_entities_in_one_document():
    doc = _doc(
        "Partnership news",
        "Microsoft and NVIDIA announced an expanded AI infrastructure partnership.",
    )
    matches = resolve_entities(doc)
    canonical_names = {m.canonical_name for m in matches}
    assert canonical_names == {"Microsoft Corporation", "NVIDIA Corporation"}
    assert len(matches) == 2


def test_case_and_punctuation_normalization():
    # Mixed case, extra punctuation, still resolves to Apple Inc.
    doc = _doc("Earnings", "APPLE INC.! reported quarterly results, beating estimates.")
    matches = resolve_entities(doc)
    assert len(matches) == 1
    assert matches[0].canonical_name == "Apple Inc."
    assert matches[0].ticker == "AAPL"


def test_no_match_returns_empty_list():
    doc = _doc("Unrelated story", "A local bakery won an award for its sourdough bread.")
    matches = resolve_entities(doc)
    assert matches == []


def test_single_entity_returned_once_even_with_repeated_mentions():
    doc = _doc(
        "Tesla coverage",
        "Tesla Inc. shares moved. Tesla Inc. also confirmed TSLA production numbers.",
    )
    matches = resolve_entities(doc)
    assert len(matches) == 1
    assert matches[0].canonical_name == "Tesla Inc."
    # ticker mention co-occurs, so the higher-priority "ticker" match wins
    assert matches[0].match_type == "ticker"


def test_fuzzy_fallback_catches_minor_misspelling():
    resolver = EntityResolver(enable_fuzzy_fallback=True, fuzzy_threshold=0.85)
    doc = _doc("Typo test", "Microsft unveiled a new cloud offering this week.")
    matches = resolver.resolve(doc)
    assert len(matches) == 1
    assert matches[0].canonical_name == "Microsoft Corporation"
    assert matches[0].match_type == "fuzzy"


def test_fuzzy_fallback_disabled_means_no_match_on_misspelling():
    resolver = EntityResolver(enable_fuzzy_fallback=False)
    doc = _doc("Typo test", "Microsft unveiled a new cloud offering this week.")
    matches = resolver.resolve(doc)
    assert matches == []

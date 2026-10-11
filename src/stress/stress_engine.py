"""
Module B: portfolio stress engine.

Chain: Risk signal -> exposure matching -> scenario selection -> asset
valuation -> portfolio loss -> loss attribution.

Reuses, without duplicating, the existing pipeline's output types:
`Signal` (src/impact/signal_builder.py) is the input; nothing here
re-implements entity resolution, sentiment, event classification,
event resolution, or the impact engine. This module is purely
downstream of a Signal that already has an `impact_score`.

All monetary/risk numbers in data/portfolio/portfolio.csv and
data/scenarios/scenario_library.json are SYNTHETIC/STYLIZED
assumptions - see those files' own documentation. This module computes
with them but does not itself fabricate or recalibrate them, and never
claims the output is empirically calibrated, Basel-compliant, or a
measure of probability of loss (impact_score, which drives severity
scaling here, carries that same caveat from src/impact/impact_engine.py).

Sign convention (consistent across every asset type): `stress_loss` is
POSITIVE when the stress scenario reduces value, and `market_value_after
= market_value_before - stress_loss` always. A negative `stress_loss`
is possible (e.g. a scenario could imply a GAIN for a specific
position, such as a payer swap in a rate-up scenario) and is not
clamped - only the ASSET VALUE itself is floored at zero for cash
instruments (bonds/loans/equities) where a negative value would be
nonsensical; derivative MTM (swaps/FX forwards) is never floored,
since a negative mark-to-market is a normal, valid state.

Exposure targeting (see `_determine_tier` and `stress_asset`): every
asset's relevance to a given signal is decided by, in priority order:
  1. direct_issuer_match  - the asset's ticker is one of the signal's
     resolved entity tickers (full shock weight).
  2. sector_contagion     - the asset shares a sector with a directly-
     matched issuer found in the portfolio (reduced weight).
  3. country_contagion    - the asset shares a country with a directly-
     matched issuer found in the portfolio (smaller reduced weight).
  4. systemic_baseline    - ONLY for scenarios flagged "systemic" in
     the scenario library: a smaller flat equity/credit shock applied
     to assets that didn't match tiers 1-3, reflecting genuine
     market-wide (not issuer-specific) risk-on/off effects.
  5. systemic_rate / systemic_fx / systemic_commodity - ONLY for
     "systemic" scenarios: their rate_shock_bps / fx_return_shock /
     commodity_return_shock fields apply at FULL magnitude, regardless
     of tier, to the asset types/sectors those fields are genuinely
     relevant to (bonds & swaps for rates, FX forwards for fx,
     Energy/Materials-sector assets for commodity) - a central bank
     decision or a broad FX move is not issuer-specific, so it is not
     diluted by the issuer-tier logic above.
  6. unaffected           - none of the above apply; the asset is left
     completely untouched (zero stress_loss), which is the default for
     "idiosyncratic" scenarios and any asset outside all of the above
     for "systemic" ones. Unrelated assets never receive an arbitrary
     large issuer-specific shock.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, FrozenSet, List, Literal, Optional

from pydantic import BaseModel

DEFAULT_PORTFOLIO_PATH = Path(__file__).resolve().parents[2] / "data" / "portfolio" / "portfolio.csv"
DEFAULT_SCENARIO_LIBRARY_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "scenarios" / "scenario_library.json"
)

AssetType = Literal["CORPORATE_BOND", "CORPORATE_LOAN", "EQUITY", "INTEREST_RATE_SWAP", "FX_FORWARD"]
ScenarioBreadth = Literal["idiosyncratic", "systemic"]

# --- exposure-targeting weights (documented assumptions) -------------------
DIRECT_MULTIPLIER = 1.0
SECTOR_CONTAGION_MULTIPLIER = 0.35
COUNTRY_CONTAGION_MULTIPLIER = 0.15
SYSTEMIC_BASELINE_MULTIPLIER = 0.5
COMMODITY_SECTORS: FrozenSet[str] = frozenset({"Energy", "Materials"})
# A commodity-sector bond/loan also picks up extra credit-spread/PD
# stress under a COMMODITY scenario (commodity exposure historically
# correlates with credit stress for those issuers) - these two
# constants convert the scenario's commodity_return_shock (a price
# return) into that extra credit stress. Documented, not measured.
COMMODITY_BOND_SPREAD_BPS_PER_UNIT_SHOCK = 300.0
COMMODITY_LOAN_PD_MULTIPLIER_KICKER_PER_UNIT_SHOCK = 0.5

# --- impact-score-driven severity scaling -----------------------------------
DEFAULT_IMPACT_THRESHOLD = 7.0
DEFAULT_SEVERITY_MIN_MULTIPLIER = 0.5
DEFAULT_SEVERITY_MAX_MULTIPLIER = 2.0
DEFAULT_TOP_N_CONTRIBUTORS = 10


class PortfolioAsset(BaseModel):
    """One row of the synthetic portfolio. Fields not relevant to a
    given asset_type are None - see scripts/generate_portfolio.py's
    docstring for exactly which fields apply to which type."""

    asset_id: str
    asset_type: AssetType
    issuer: str
    ticker: Optional[str] = None
    sector: Optional[str] = None
    country: Optional[str] = None
    currency: Optional[str] = None
    notional: Optional[float] = None
    market_value: float
    rating: Optional[str] = None
    duration: Optional[float] = None
    convexity: Optional[float] = None
    pd: Optional[float] = None
    lgd: Optional[float] = None
    beta: Optional[float] = None
    dv01: Optional[float] = None
    fx_sensitivity: Optional[float] = None


class Scenario(BaseModel):
    """One entry of the scenario library - see data/scenarios/scenario_library.json."""

    event_type: str
    breadth: ScenarioBreadth
    equity_return_shock: float = 0.0
    rate_shock_bps: float = 0.0
    credit_spread_shock_bps: float = 0.0
    pd_multiplier: float = 1.0
    lgd_change: float = 0.0
    commodity_return_shock: float = 0.0
    fx_return_shock: float = 0.0
    description: str = ""
    notes: str = ""


class AssetStressResult(BaseModel):
    """Per-asset stress outcome, with an explicit explanation of why
    (or why not) it was affected."""

    asset_id: str
    # Deliberately plain `str`, not the `AssetType` Literal: an
    # unsupported/unrecognized asset_type must still be representable
    # here (skipped=True, skip_reason="unsupported asset_type '...'")
    # for transparency, rather than failing to even report the skip.
    asset_type: str
    issuer: str
    ticker: Optional[str] = None
    market_value_before: float
    market_value_after: float
    stress_loss: float  # positive = loss; see module docstring's sign convention
    matching_rules: List[str] = []
    explanation: str
    skipped: bool = False
    skip_reason: Optional[str] = None


class StressTestResult(BaseModel):
    """Structured, serializable output of one run_stress_test() call."""

    triggered: bool
    trigger_reason: str
    signal_event_id: Optional[str] = None
    signal_entities: List[str] = []
    impact_score: Optional[float] = None
    scenario_name: Optional[str] = None
    scenario_assumptions: Optional[dict] = None
    severity_multiplier: Optional[float] = None
    portfolio_value_before: Optional[float] = None
    portfolio_value_after: Optional[float] = None
    absolute_loss: Optional[float] = None
    pct_loss: Optional[float] = None
    top_contributors: List[AssetStressResult] = []
    attribution_by_asset_class: Dict[str, float] = {}
    affected_assets: List[AssetStressResult] = []
    skipped_assets: List[AssetStressResult] = []


def _parse_optional_float(value: Optional[str]) -> Optional[float]:
    if value is None or value.strip() == "":
        return None
    return float(value)


def load_portfolio(path: Optional[Path] = None) -> List[PortfolioAsset]:
    """
    Load and validate the synthetic portfolio. Raises clearly (via
    pydantic validation, e.g. an unrecognized asset_type) on a
    fundamentally malformed file - this is a one-time static load of a
    single file, not a batch of independent documents, so unlike the
    NLP ingestion pipeline there is no "skip the bad one, keep going"
    story here: a broken portfolio file is a setup error you want to
    know about immediately, not silently work around.
    """
    path = path or DEFAULT_PORTFOLIO_PATH
    assets: List[PortfolioAsset] = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            assets.append(
                PortfolioAsset(
                    asset_id=row["asset_id"],
                    asset_type=row["asset_type"],
                    issuer=row["issuer"],
                    ticker=row["ticker"] or None,
                    sector=row["sector"] or None,
                    country=row["country"] or None,
                    currency=row["currency"] or None,
                    notional=_parse_optional_float(row["notional"]),
                    market_value=float(row["market_value"]),
                    rating=row["rating"] or None,
                    duration=_parse_optional_float(row["duration"]),
                    convexity=_parse_optional_float(row["convexity"]),
                    pd=_parse_optional_float(row["pd"]),
                    lgd=_parse_optional_float(row["lgd"]),
                    beta=_parse_optional_float(row["beta"]),
                    dv01=_parse_optional_float(row["dv01"]),
                    fx_sensitivity=_parse_optional_float(row["fx_sensitivity"]),
                )
            )
    return assets


def load_scenario_library(path: Optional[Path] = None) -> Dict[str, Scenario]:
    """Load the scenario library, keyed by event_type. Skips the
    "_README" documentation entry."""
    path = path or DEFAULT_SCENARIO_LIBRARY_PATH
    with Path(path).open(encoding="utf-8") as f:
        raw = json.load(f)
    return {key: Scenario(**value) for key, value in raw.items() if key != "_README"}


# --- valuation mechanics (pure, individually testable) ----------------------


def bond_return(duration: float, convexity: float, delta_y: float) -> float:
    """return ≈ -duration * Δy + 0.5 * convexity * (Δy ** 2); Δy in decimal yield units (0.01 = 100bps)."""
    return -duration * delta_y + 0.5 * convexity * (delta_y ** 2)


def loan_stress_loss(ead: float, pd_base: float, lgd_base: float, pd_stress: float, lgd_stress: float) -> float:
    """
    Incremental expected-credit-loss stress: EL = EAD * PD * LGD,
    stress_loss = EL_stress - EL_base. `pd_stress`/`lgd_stress` are the
    ALREADY-STRESSED (post-multiplier/post-delta) values, clamped to
    [0, 1] here so a large multiplier/delta can never imply a PD or LGD
    outside its valid probability/fraction range.
    """
    pd_stress = min(1.0, max(0.0, pd_stress))
    lgd_stress = min(1.0, max(0.0, lgd_stress))
    el_base = ead * pd_base * lgd_base
    el_stress = ead * pd_stress * lgd_stress
    return el_stress - el_base


def equity_return(beta: float, shock: float) -> float:
    """Δvalue/market_value ≈ beta * equity_return_shock."""
    return beta * shock


def swap_pv_change(dv01: float, yield_change_bps: float) -> float:
    """
    ΔPV = -DV01 * Δyield_bps. Sign convention: a POSITIVE dv01
    represents a position that LOSES value when yields rise (e.g. a
    receive-fixed swap) - so a positive yield_change_bps with a
    positive dv01 gives a negative ΔPV (a loss), consistent with the
    module's overall sign convention once stress_loss = -ΔPV.
    """
    return -dv01 * yield_change_bps


def fx_value_change(fx_sensitivity: float, fx_return_shock: float) -> float:
    """Δvalue = fx_sensitivity * fx_return_shock (a simple linear delta approximation, not a full FX option/forward pricer)."""
    return fx_sensitivity * fx_return_shock


# --- impact-score severity scaling ------------------------------------------


def compute_severity_multiplier(
    impact_score: float,
    threshold: float = DEFAULT_IMPACT_THRESHOLD,
    min_multiplier: float = DEFAULT_SEVERITY_MIN_MULTIPLIER,
    max_multiplier: float = DEFAULT_SEVERITY_MAX_MULTIPLIER,
) -> float:
    """
    Documented, bounded scaling of scenario shock magnitudes by
    impact_score: `multiplier = clamp(impact_score / threshold, min,
    max)`. At exactly the trigger threshold, the multiplier is 1.0
    (the scenario's base/documented magnitudes apply unscaled); above
    threshold, shocks scale up (capped at `max_multiplier`); this
    function is never called for impact_score <= threshold in the
    normal flow (the scenario doesn't trigger at all), so the
    `min_multiplier` floor mostly guards direct/unit-test calls with
    an out-of-range impact_score rather than normal operation.
    """
    if threshold <= 0:
        raise ValueError("threshold must be > 0")
    raw = impact_score / threshold
    return max(min_multiplier, min(max_multiplier, raw))


def _determine_tier(
    asset: PortfolioAsset,
    direct_tickers: FrozenSet[str],
    direct_sectors: FrozenSet[str],
    direct_countries: FrozenSet[str],
) -> "tuple[float, str]":
    if asset.ticker and asset.ticker in direct_tickers:
        return DIRECT_MULTIPLIER, "direct_issuer_match"
    if asset.sector and asset.sector in direct_sectors:
        return SECTOR_CONTAGION_MULTIPLIER, "sector_contagion"
    if asset.country and asset.country in direct_countries:
        return COUNTRY_CONTAGION_MULTIPLIER, "country_contagion"
    return 0.0, "unaffected"


def stress_asset(
    asset: PortfolioAsset,
    scenario: Scenario,
    severity_multiplier: float,
    direct_tickers: FrozenSet[str],
    direct_sectors: FrozenSet[str],
    direct_countries: FrozenSet[str],
) -> AssetStressResult:
    """Stress one asset under one scenario; never raises - invalid/
    missing inputs or an unsupported asset_type produce a `skipped`
    result with an explicit reason instead."""
    tier_mult, tier_rule = _determine_tier(asset, direct_tickers, direct_sectors, direct_countries)
    is_systemic = scenario.breadth == "systemic"
    credit_mult = tier_mult if tier_mult > 0 else (SYSTEMIC_BASELINE_MULTIPLIER if is_systemic else 0.0)
    credit_rule = tier_rule if tier_mult > 0 else ("systemic_baseline" if is_systemic else "unaffected")

    rules_applied: List[str] = []
    delta_value = 0.0
    skip_reason: Optional[str] = None

    try:
        if asset.asset_type == "CORPORATE_BOND":
            if asset.duration is None or asset.convexity is None:
                skip_reason = "missing duration/convexity"
            else:
                rate_bps = 0.0
                if is_systemic and scenario.rate_shock_bps:
                    rate_bps = scenario.rate_shock_bps * severity_multiplier
                    rules_applied.append("systemic_rate")
                spread_bps = scenario.credit_spread_shock_bps * severity_multiplier * credit_mult
                if credit_mult > 0 and scenario.credit_spread_shock_bps:
                    rules_applied.append(credit_rule)
                commodity_bps = 0.0
                if is_systemic and scenario.commodity_return_shock and asset.sector in COMMODITY_SECTORS:
                    commodity_bps = abs(scenario.commodity_return_shock) * severity_multiplier * COMMODITY_BOND_SPREAD_BPS_PER_UNIT_SHOCK
                    rules_applied.append("systemic_commodity")
                delta_y = (rate_bps + spread_bps + commodity_bps) / 10000.0
                delta_value = asset.market_value * bond_return(asset.duration, asset.convexity, delta_y)

        elif asset.asset_type == "CORPORATE_LOAN":
            if asset.pd is None or asset.lgd is None or asset.notional is None:
                skip_reason = "missing pd/lgd/notional"
            else:
                commodity_kicker = 0.0
                if is_systemic and scenario.commodity_return_shock and asset.sector in COMMODITY_SECTORS:
                    commodity_kicker = abs(scenario.commodity_return_shock) * severity_multiplier * COMMODITY_LOAN_PD_MULTIPLIER_KICKER_PER_UNIT_SHOCK
                    rules_applied.append("systemic_commodity")
                effective_pd_multiplier = 1.0 + (scenario.pd_multiplier - 1.0) * severity_multiplier * credit_mult + commodity_kicker
                effective_lgd_change = scenario.lgd_change * severity_multiplier * credit_mult
                if credit_mult > 0 and (scenario.pd_multiplier != 1.0 or scenario.lgd_change != 0.0):
                    rules_applied.append(credit_rule)
                stress_loss_amount = loan_stress_loss(
                    asset.notional, asset.pd, asset.lgd, asset.pd * effective_pd_multiplier, asset.lgd + effective_lgd_change
                )
                delta_value = -stress_loss_amount

        elif asset.asset_type == "EQUITY":
            if asset.beta is None:
                skip_reason = "missing beta"
            else:
                equity_shock = scenario.equity_return_shock * severity_multiplier * credit_mult
                if credit_mult > 0 and scenario.equity_return_shock:
                    rules_applied.append(credit_rule)
                commodity_component = 0.0
                if is_systemic and scenario.commodity_return_shock and asset.sector in COMMODITY_SECTORS:
                    commodity_component = scenario.commodity_return_shock * severity_multiplier
                    rules_applied.append("systemic_commodity")
                delta_value = asset.market_value * equity_return(asset.beta, equity_shock + commodity_component)

        elif asset.asset_type == "INTEREST_RATE_SWAP":
            if asset.dv01 is None:
                skip_reason = "missing dv01"
            elif is_systemic and scenario.rate_shock_bps:
                delta_value = swap_pv_change(asset.dv01, scenario.rate_shock_bps * severity_multiplier)
                rules_applied.append("systemic_rate")

        elif asset.asset_type == "FX_FORWARD":
            if asset.fx_sensitivity is None:
                skip_reason = "missing fx_sensitivity"
            elif is_systemic and scenario.fx_return_shock:
                delta_value = fx_value_change(asset.fx_sensitivity, scenario.fx_return_shock * severity_multiplier)
                rules_applied.append("systemic_fx")

        else:
            skip_reason = f"unsupported asset_type '{asset.asset_type}'"

    except (TypeError, ValueError) as exc:
        skip_reason = f"invalid input: {exc}"

    if skip_reason:
        return AssetStressResult(
            asset_id=asset.asset_id,
            asset_type=asset.asset_type,
            issuer=asset.issuer,
            ticker=asset.ticker,
            market_value_before=asset.market_value,
            market_value_after=asset.market_value,
            stress_loss=0.0,
            matching_rules=[],
            explanation=f"Skipped: {skip_reason}.",
            skipped=True,
            skip_reason=skip_reason,
        )

    market_value_after = asset.market_value + delta_value
    floored = False
    # Cash instruments can't sensibly have negative value in this simple
    # model; derivative MTM (swaps/FX forwards) legitimately can, so it
    # is never floored.
    if asset.asset_type in ("CORPORATE_BOND", "CORPORATE_LOAN", "EQUITY") and market_value_after < 0:
        market_value_after = 0.0
        floored = True

    stress_loss = asset.market_value - market_value_after  # positive = loss

    if not rules_applied:
        explanation = "Unaffected: no direct, sector, country, or systemic relevance for this event type."
    else:
        explanation = f"Stressed via: {', '.join(rules_applied)} under the '{scenario.event_type}' scenario."
        if floored:
            explanation += " Stressed value floored at 0."

    return AssetStressResult(
        asset_id=asset.asset_id,
        asset_type=asset.asset_type,
        issuer=asset.issuer,
        ticker=asset.ticker,
        market_value_before=asset.market_value,
        market_value_after=round(market_value_after, 2),
        stress_loss=round(stress_loss, 2),
        matching_rules=rules_applied,
        explanation=explanation,
        skipped=False,
    )


def run_stress_test(
    signal,  # src.impact.signal_builder.Signal - not type-hinted directly to avoid a hard import-time dependency loop risk; see module docstring
    portfolio: List[PortfolioAsset],
    scenario_library: Dict[str, Scenario],
    threshold: float = DEFAULT_IMPACT_THRESHOLD,
    top_n: int = DEFAULT_TOP_N_CONTRIBUTORS,
) -> StressTestResult:
    """
    The full chain: trigger check -> scenario selection -> exposure
    matching -> per-asset valuation -> portfolio loss -> attribution.
    Always returns a structured result, whether or not the scenario
    actually triggered (so callers - including the dashboard - never
    need to special-case "no result").
    """
    entity_tickers = [e.ticker for e in signal.entities]

    if signal.impact_score is None or signal.impact_score <= threshold:
        reason = (
            "impact_score is None (not yet computed for this signal)"
            if signal.impact_score is None
            else f"impact_score {signal.impact_score} does not exceed threshold {threshold}"
        )
        return StressTestResult(
            triggered=False,
            trigger_reason=reason,
            signal_event_id=signal.event_id,
            signal_entities=entity_tickers,
            impact_score=signal.impact_score,
        )

    scenario = scenario_library.get(signal.event.event_type)
    if scenario is None:
        return StressTestResult(
            triggered=False,
            trigger_reason=f"no scenario mapped for event_type '{signal.event.event_type}'",
            signal_event_id=signal.event_id,
            signal_entities=entity_tickers,
            impact_score=signal.impact_score,
        )

    severity_multiplier = compute_severity_multiplier(signal.impact_score, threshold)

    direct_tickers = frozenset(t for t in entity_tickers if t)
    direct_sectors = frozenset(a.sector for a in portfolio if a.ticker in direct_tickers and a.sector)
    direct_countries = frozenset(a.country for a in portfolio if a.ticker in direct_tickers and a.country)

    results = [
        stress_asset(asset, scenario, severity_multiplier, direct_tickers, direct_sectors, direct_countries)
        for asset in portfolio
    ]
    valid_results = [r for r in results if not r.skipped]
    skipped_results = [r for r in results if r.skipped]
    affected_results = [r for r in valid_results if r.matching_rules]

    portfolio_value_before = sum(r.market_value_before for r in valid_results)
    portfolio_value_after = sum(r.market_value_after for r in valid_results)
    absolute_loss = portfolio_value_before - portfolio_value_after
    pct_loss = (absolute_loss / portfolio_value_before) if portfolio_value_before else 0.0

    top_contributors = sorted(affected_results, key=lambda r: r.stress_loss, reverse=True)[:top_n]

    attribution: Dict[str, float] = {}
    for r in valid_results:
        attribution[r.asset_type] = attribution.get(r.asset_type, 0.0) + r.stress_loss
    attribution = {k: round(v, 2) for k, v in attribution.items()}

    return StressTestResult(
        triggered=True,
        trigger_reason=f"impact_score {signal.impact_score} exceeds threshold {threshold}",
        signal_event_id=signal.event_id,
        signal_entities=entity_tickers,
        impact_score=signal.impact_score,
        scenario_name=scenario.event_type,
        scenario_assumptions=scenario.model_dump(),
        severity_multiplier=round(severity_multiplier, 4),
        portfolio_value_before=round(portfolio_value_before, 2),
        portfolio_value_after=round(portfolio_value_after, 2),
        absolute_loss=round(absolute_loss, 2),
        pct_loss=round(pct_loss, 6),
        top_contributors=top_contributors,
        attribution_by_asset_class=attribution,
        affected_assets=affected_results,
        skipped_assets=skipped_results,
    )

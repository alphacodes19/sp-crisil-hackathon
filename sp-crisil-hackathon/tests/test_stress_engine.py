import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from src.stress.stress_engine import (
    AssetStressResult,
    DEFAULT_IMPACT_THRESHOLD,
    PortfolioAsset,
    Scenario,
    bond_return,
    compute_severity_multiplier,
    equity_return,
    fx_value_change,
    load_portfolio,
    load_scenario_library,
    loan_stress_loss,
    run_stress_test,
    stress_asset,
    swap_pv_change,
)


def _asset(**overrides):
    base = dict(
        asset_id="X0001", asset_type="EQUITY", issuer="Test Co", ticker="TST",
        sector="Technology", country="US", currency="USD", notional=None,
        market_value=1_000_000.0, rating=None, duration=None, convexity=None,
        pd=None, lgd=None, beta=1.0, dv01=None, fx_sensitivity=None,
    )
    base.update(overrides)
    return PortfolioAsset(**base)


def _scenario(**overrides):
    base = dict(
        event_type="CREDIT", breadth="idiosyncratic", equity_return_shock=-0.1,
        rate_shock_bps=0, credit_spread_shock_bps=50, pd_multiplier=1.5, lgd_change=0.05,
        commodity_return_shock=0.0, fx_return_shock=0.0,
    )
    base.update(overrides)
    return Scenario(**base)


# --- portfolio loading --------------------------------------------------


def test_load_portfolio_returns_expected_count_and_types():
    assets = load_portfolio()
    assert 40 <= len(assets) <= 60
    types = {a.asset_type for a in assets}
    assert types == {"CORPORATE_BOND", "CORPORATE_LOAN", "EQUITY", "INTEREST_RATE_SWAP", "FX_FORWARD"}


def test_load_portfolio_includes_direct_zzdemo_exposure():
    assets = load_portfolio()
    zzdemo_assets = [a for a in assets if a.ticker == "ZZDEMO"]
    assert len(zzdemo_assets) >= 2
    assert {a.asset_type for a in zzdemo_assets} >= {"CORPORATE_BOND"}
    for a in zzdemo_assets:
        assert "SYNTHETIC" in a.issuer.upper()


def test_load_portfolio_type_specific_fields_are_none_when_not_applicable():
    assets = load_portfolio()
    equities = [a for a in assets if a.asset_type == "EQUITY"]
    assert all(a.beta is not None for a in equities)
    assert all(a.duration is None and a.pd is None and a.dv01 is None for a in equities)

    swaps = [a for a in assets if a.asset_type == "INTEREST_RATE_SWAP"]
    assert all(a.dv01 is not None for a in swaps)
    assert all(a.beta is None and a.pd is None for a in swaps)


def test_load_scenario_library_covers_required_event_types():
    scenarios = load_scenario_library()
    required = {
        "REGULATORY_LEGAL", "CREDIT", "RATING_ACTION", "MACRO_CENTRAL_BANK",
        "GEOPOLITICAL", "SUPPLY_CHAIN", "COMMODITY", "CYBER_OPERATIONAL",
    }
    assert required <= set(scenarios.keys())
    assert "_README" not in scenarios


# --- valuation mechanics: hand-computed examples ----------------------------


def test_bond_return_hand_computed():
    # duration=5, convexity=0.5, delta_y=0.01 (100bps)
    # -5*0.01 + 0.5*0.5*(0.01**2) = -0.05 + 0.000025 = -0.049975
    result = bond_return(duration=5.0, convexity=0.5, delta_y=0.01)
    assert result == pytest.approx(-0.049975)


def test_bond_return_zero_shock_is_zero():
    assert bond_return(duration=7.0, convexity=0.3, delta_y=0.0) == 0.0


def test_loan_stress_loss_hand_computed():
    # EAD=10,000,000 PD_base=0.02 LGD_base=0.4 -> EL_base=80,000
    # PD_stress=0.03 LGD_stress=0.45 -> EL_stress=10,000,000*0.03*0.45=135,000
    # stress_loss = 135,000 - 80,000 = 55,000
    result = loan_stress_loss(ead=10_000_000, pd_base=0.02, lgd_base=0.4, pd_stress=0.03, lgd_stress=0.45)
    assert result == pytest.approx(55_000.0)


def test_loan_stress_loss_clamps_pd_and_lgd_to_valid_range():
    # pd_stress/lgd_stress far outside [0,1] must be clamped, not produce nonsensical EL.
    result = loan_stress_loss(ead=1_000_000, pd_base=0.01, lgd_base=0.3, pd_stress=5.0, lgd_stress=5.0)
    el_base = 1_000_000 * 0.01 * 0.3
    el_stress_clamped = 1_000_000 * 1.0 * 1.0
    assert result == pytest.approx(el_stress_clamped - el_base)


def test_equity_return_hand_computed():
    assert equity_return(beta=1.5, shock=-0.1) == pytest.approx(-0.15)


def test_swap_pv_change_sign_convention():
    # positive dv01 (loses value when yields rise) + positive yield_change_bps -> loss (negative PV change)
    assert swap_pv_change(dv01=10_000, yield_change_bps=50) == pytest.approx(-500_000)
    # negative dv01 (gains when yields rise) + positive yield_change_bps -> gain
    assert swap_pv_change(dv01=-10_000, yield_change_bps=50) == pytest.approx(500_000)


def test_fx_value_change_hand_computed():
    assert fx_value_change(fx_sensitivity=1_000_000, fx_return_shock=-0.05) == pytest.approx(-50_000)


# --- severity scaling --------------------------------------------------------


def test_severity_multiplier_at_threshold_is_one():
    assert compute_severity_multiplier(7.0, threshold=7.0) == pytest.approx(1.0)


def test_severity_multiplier_is_bounded():
    assert compute_severity_multiplier(100.0, threshold=7.0, max_multiplier=2.0) == 2.0
    assert compute_severity_multiplier(0.1, threshold=7.0, min_multiplier=0.5) == 0.5


def test_severity_multiplier_rejects_non_positive_threshold():
    with pytest.raises(ValueError):
        compute_severity_multiplier(8.0, threshold=0.0)


# --- exposure targeting / stress_asset ---------------------------------------


def test_direct_issuer_match_gets_full_shock():
    asset = _asset(ticker="TST", sector="Technology", country="US", beta=2.0)
    scenario = _scenario(equity_return_shock=-0.1, credit_spread_shock_bps=0, pd_multiplier=1.0, lgd_change=0.0)
    result = stress_asset(asset, scenario, severity_multiplier=1.0, direct_tickers=frozenset({"TST"}), direct_sectors=frozenset(), direct_countries=frozenset())
    assert result.matching_rules == ["direct_issuer_match"]
    assert result.stress_loss == pytest.approx(1_000_000 * 2.0 * 0.1)  # loss = -delta_value = -(mv*beta*shock)


def test_sector_contagion_gets_reduced_shock_vs_direct():
    direct = _asset(ticker="TST", sector="Technology", beta=2.0)
    indirect = _asset(asset_id="X0002", ticker="OTHER", sector="Technology", beta=2.0)
    scenario = _scenario(equity_return_shock=-0.1)
    direct_result = stress_asset(direct, scenario, 1.0, frozenset({"TST"}), frozenset(), frozenset())
    sector_result = stress_asset(indirect, scenario, 1.0, frozenset({"TST"}), frozenset({"Technology"}), frozenset())
    assert sector_result.matching_rules == ["sector_contagion"]
    assert 0 < sector_result.stress_loss < direct_result.stress_loss


def test_country_contagion_is_smaller_than_sector_contagion():
    sector_asset = _asset(asset_id="X0002", ticker="OTHER1", sector="Technology", country="DE", beta=2.0)
    country_asset = _asset(asset_id="X0003", ticker="OTHER2", sector="Healthcare", country="US", beta=2.0)
    scenario = _scenario(equity_return_shock=-0.1)
    sector_result = stress_asset(sector_asset, scenario, 1.0, frozenset({"TST"}), frozenset({"Technology"}), frozenset({"US"}))
    country_result = stress_asset(country_asset, scenario, 1.0, frozenset({"TST"}), frozenset({"Technology"}), frozenset({"US"}))
    assert country_result.matching_rules == ["country_contagion"]
    assert 0 < country_result.stress_loss < sector_result.stress_loss


def test_unrelated_asset_gets_no_arbitrary_shock_for_idiosyncratic_scenario():
    unrelated = _asset(asset_id="X0004", ticker="UNRELATED", sector="Energy", country="JP", beta=2.0)
    scenario = _scenario(breadth="idiosyncratic", equity_return_shock=-0.5)  # even a huge shock
    result = stress_asset(unrelated, scenario, 1.0, frozenset({"TST"}), frozenset({"Technology"}), frozenset({"US"}))
    assert result.matching_rules == []
    assert result.stress_loss == 0.0
    assert "Unaffected" in result.explanation


def test_systemic_scenario_gives_unrelated_asset_a_small_baseline_not_zero():
    unrelated = _asset(asset_id="X0004", ticker="UNRELATED", sector="Energy", country="JP", beta=2.0)
    scenario = _scenario(breadth="systemic", equity_return_shock=-0.1)
    result = stress_asset(unrelated, scenario, 1.0, frozenset({"TST"}), frozenset({"Technology"}), frozenset({"US"}))
    assert result.matching_rules == ["systemic_baseline"]
    direct_equivalent_loss = 1_000_000 * 2.0 * 0.1
    assert 0 < result.stress_loss < direct_equivalent_loss


def test_systemic_rate_shock_hits_bonds_and_swaps_regardless_of_tier():
    bond = _asset(asset_id="BND01", asset_type="CORPORATE_BOND", ticker="UNRELATED", sector="Energy", country="JP",
                   beta=None, duration=5.0, convexity=0.3)
    swap = _asset(asset_id="SWP01", asset_type="INTEREST_RATE_SWAP", ticker=None, sector="Rates", country="JP",
                   beta=None, dv01=10_000.0)
    scenario = _scenario(breadth="systemic", rate_shock_bps=50, equity_return_shock=0.0, credit_spread_shock_bps=0.0)
    bond_result = stress_asset(bond, scenario, 1.0, frozenset({"TST"}), frozenset(), frozenset())
    swap_result = stress_asset(swap, scenario, 1.0, frozenset({"TST"}), frozenset(), frozenset())
    assert "systemic_rate" in bond_result.matching_rules
    assert "systemic_rate" in swap_result.matching_rules
    assert bond_result.stress_loss != 0.0
    assert swap_result.stress_loss != 0.0


def test_fx_forward_only_stressed_by_systemic_fx_shock():
    fx = _asset(asset_id="FX01", asset_type="FX_FORWARD", ticker=None, sector="FX", country="GB", beta=None, fx_sensitivity=2_000_000.0)
    idiosyncratic_scenario = _scenario(breadth="idiosyncratic", fx_return_shock=0.1)
    systemic_scenario = _scenario(breadth="systemic", fx_return_shock=0.1)
    idio_result = stress_asset(fx, idiosyncratic_scenario, 1.0, frozenset(), frozenset(), frozenset())
    sys_result = stress_asset(fx, systemic_scenario, 1.0, frozenset(), frozenset(), frozenset())
    assert idio_result.matching_rules == []
    assert idio_result.stress_loss == 0.0
    assert "systemic_fx" in sys_result.matching_rules
    assert sys_result.stress_loss == pytest.approx(-2_000_000.0 * 0.1)  # gain -> negative loss


# --- missing / invalid / unsupported handling -------------------------------


def test_missing_required_field_is_skipped_not_crashed():
    bond_missing_duration = _asset(asset_id="BND02", asset_type="CORPORATE_BOND", ticker="TST", duration=None, convexity=0.3)
    scenario = _scenario()
    result = stress_asset(bond_missing_duration, scenario, 1.0, frozenset({"TST"}), frozenset(), frozenset())
    assert result.skipped is True
    assert "missing" in result.skip_reason
    assert result.stress_loss == 0.0


def test_unsupported_asset_type_is_skipped():
    # Construct a PortfolioAsset bypassing the Literal validator isn't
    # possible via normal construction, so we exercise this path via
    # model_construct (used only for this one test to prove stress_asset's
    # defensive branch, since real CSV loading already guarantees a valid
    # asset_type via pydantic validation).
    weird_asset = PortfolioAsset.model_construct(
        asset_id="X9999", asset_type="SOMETHING_UNSUPPORTED", issuer="Weird Co", ticker=None,
        sector=None, country=None, currency=None, notional=None, market_value=1_000_000.0,
        rating=None, duration=None, convexity=None, pd=None, lgd=None, beta=None, dv01=None, fx_sensitivity=None,
    )
    result = stress_asset(weird_asset, _scenario(), 1.0, frozenset(), frozenset(), frozenset())
    assert result.skipped is True
    assert "unsupported" in result.skip_reason


def test_negative_stressed_value_floored_for_cash_instruments_not_derivatives():
    tiny_equity = _asset(asset_id="X0005", ticker="TST", market_value=100.0, beta=50.0)  # huge beta forces negative value
    scenario = _scenario(equity_return_shock=-0.9)
    result = stress_asset(tiny_equity, scenario, 1.0, frozenset({"TST"}), frozenset(), frozenset())
    assert result.market_value_after == 0.0
    assert "floored" in result.explanation.lower()

    swap_asset = _asset(asset_id="SWP02", asset_type="INTEREST_RATE_SWAP", ticker=None, market_value=-500_000.0, dv01=100_000.0, beta=None)
    swap_scenario = _scenario(breadth="systemic", rate_shock_bps=500)
    swap_result = stress_asset(swap_asset, swap_scenario, 1.0, frozenset(), frozenset(), frozenset())
    assert swap_result.market_value_after < 0  # derivative MTM may legitimately stay negative


# --- full run_stress_test orchestration --------------------------------------


class _FakeEntity:
    def __init__(self, ticker):
        self.ticker = ticker


class _FakeEvent:
    def __init__(self, event_type):
        self.event_type = event_type


class _FakeSignal:
    def __init__(self, event_id, entities, event_type, impact_score):
        self.event_id = event_id
        self.entities = [_FakeEntity(t) for t in entities]
        self.event = _FakeEvent(event_type)
        self.impact_score = impact_score


def _tiny_portfolio():
    return [
        _asset(asset_id="E1", ticker="TST", sector="Technology", country="US", beta=1.5),
        _asset(asset_id="E2", ticker="OTHER", sector="Technology", country="DE", beta=1.0),
        _asset(asset_id="E3", asset_type="CORPORATE_BOND", ticker="TST", sector="Technology", country="US", beta=None, duration=5.0, convexity=0.3, market_value=2_000_000),
    ]


def test_run_stress_test_not_triggered_below_threshold():
    signal = _FakeSignal("evt-1", ["TST"], "CREDIT", 5.0)
    result = run_stress_test(signal, _tiny_portfolio(), load_scenario_library())
    assert result.triggered is False
    assert "does not exceed" in result.trigger_reason
    assert result.top_contributors == []


def test_run_stress_test_not_triggered_when_impact_score_is_none():
    signal = _FakeSignal("evt-1", ["TST"], "CREDIT", None)
    result = run_stress_test(signal, _tiny_portfolio(), load_scenario_library())
    assert result.triggered is False
    assert "None" in result.trigger_reason


def test_run_stress_test_not_triggered_when_no_scenario_mapped():
    signal = _FakeSignal("evt-1", ["TST"], "EARNINGS", 9.0)  # EARNINGS is not in the required scenario list
    result = run_stress_test(signal, _tiny_portfolio(), load_scenario_library())
    assert result.triggered is False
    assert "no scenario mapped" in result.trigger_reason


def test_run_stress_test_triggered_computes_full_result():
    signal = _FakeSignal("evt-1", ["TST"], "CREDIT", 9.0)
    result = run_stress_test(signal, _tiny_portfolio(), load_scenario_library())
    assert result.triggered is True
    assert result.scenario_name == "CREDIT"
    assert result.portfolio_value_before is not None
    assert result.portfolio_value_after is not None
    assert result.absolute_loss == pytest.approx(result.portfolio_value_before - result.portfolio_value_after)
    assert result.pct_loss == pytest.approx(result.absolute_loss / result.portfolio_value_before, rel=1e-4)
    assert len(result.affected_assets) >= 1
    assert set(result.attribution_by_asset_class.keys()) <= {"EQUITY", "CORPORATE_BOND"}
    assert result.top_contributors == sorted(result.affected_assets, key=lambda r: r.stress_loss, reverse=True)[: len(result.top_contributors)]


def test_run_stress_test_explanations_identify_cause_for_each_affected_asset():
    signal = _FakeSignal("evt-1", ["TST"], "CREDIT", 9.0)
    result = run_stress_test(signal, _tiny_portfolio(), load_scenario_library())
    for asset_result in result.affected_assets:
        assert asset_result.explanation
        assert "CREDIT" in asset_result.explanation
        assert asset_result.matching_rules


def test_run_stress_test_severity_scales_with_impact_score():
    low = run_stress_test(_FakeSignal("e1", ["TST"], "CREDIT", 7.5), _tiny_portfolio(), load_scenario_library())
    high = run_stress_test(_FakeSignal("e2", ["TST"], "CREDIT", 14.0), _tiny_portfolio(), load_scenario_library())
    assert high.severity_multiplier > low.severity_multiplier
    assert high.absolute_loss > low.absolute_loss


def test_run_stress_test_full_real_portfolio_runs_without_error():
    portfolio = load_portfolio()
    scenarios = load_scenario_library()
    signal = _FakeSignal("evt-real", ["ZZDEMO"], "REGULATORY_LEGAL", 9.46)
    result = run_stress_test(signal, portfolio, scenarios)
    assert result.triggered is True
    assert result.skipped_assets == []  # the real generated portfolio has no missing required fields
    assert len(result.top_contributors) <= 10
    zzdemo_contributors = [c for c in result.top_contributors if c.ticker == "ZZDEMO"]
    assert len(zzdemo_contributors) >= 1  # direct issuer exposure should dominate the top contributors

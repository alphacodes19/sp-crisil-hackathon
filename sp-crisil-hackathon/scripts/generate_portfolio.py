"""
Deterministic generator for the synthetic wholesale-banking portfolio
(data/portfolio/portfolio.csv). Run once to (re)produce the file:

    python scripts/generate_portfolio.py

ALL data here is synthetic. Issuers are real, well-known public
companies used only as realistic labels (per the master context:
"synthetic/public data only") - notional amounts, market values,
ratings, durations, PDs/LGDs, betas, DV01s and FX sensitivities are
NOT real positions and must never be treated as such. One issuer,
"Fictional Demo Holdings Inc. (SYNTHETIC DEMO COMPANY - NOT REAL)"
(ticker ZZDEMO), is entirely fictional (see src/ingestion/synthetic_demo.py)
and is deliberately given direct exposure across multiple asset classes
so the synthetic high-impact fixture can demonstrate issuer-specific
stress end-to-end.

Columns (one CSV for all five asset types; a column left blank means
"not applicable to this asset type" - see ASSET_TYPE_FIELDS below for
exactly which fields apply to which type):

    asset_id, asset_type, issuer, ticker, sector, country, currency,
    notional, market_value, rating, duration, convexity, pd, lgd,
    beta, dv01, fx_sensitivity

Per-asset-type field usage (documented assumption - deliberately
minimal, not a full securities master):
- CORPORATE_BOND:    issuer/ticker/sector/country/currency, notional,
                      market_value, rating, duration, convexity.
                      No pd/lgd (credit loss is modeled for loans only -
                      see src/stress/stress_engine.py's double-counting
                      note), no beta/dv01/fx_sensitivity.
- CORPORATE_LOAN:     issuer/ticker/sector/country/currency, notional
                      (used as EAD), market_value (carried ~at par),
                      rating, pd, lgd. No duration/convexity/beta/
                      dv01/fx_sensitivity (loans are not revalued via
                      a yield-duration approximation here).
- EQUITY:             issuer/ticker/sector/country/currency,
                      market_value, beta. No notional (not a
                      meaningful concept for a cash equity position
                      here), no rating/duration/convexity/pd/lgd/dv01/
                      fx_sensitivity.
- INTEREST_RATE_SWAP: issuer (counterparty)/sector("Rates")/country/
                      currency, notional, market_value (current MTM,
                      can be negative), dv01. No ticker (not an
                      issuer-linked instrument), no rating/duration/
                      convexity/pd/lgd/beta/fx_sensitivity.
- FX_FORWARD:         issuer (counterparty)/sector("FX")/country,
                      currency (the non-USD leg), notional,
                      market_value (current MTM), fx_sensitivity.
                      No ticker/rating/duration/convexity/pd/lgd/
                      beta/dv01.

Numeric values are generated with a FIXED random seed for full
reproducibility - re-running this script always produces the same
file.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

SEED = 20261010
random.seed(SEED)

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "portfolio" / "portfolio.csv"

FIELDNAMES = [
    "asset_id", "asset_type", "issuer", "ticker", "sector", "country", "currency",
    "notional", "market_value", "rating", "duration", "convexity", "pd", "lgd",
    "beta", "dv01", "fx_sensitivity",
]

# (issuer, ticker, sector, country) - real public companies used only as
# realistic labels, per the master context's "synthetic/public data
# only" rule. The last entry is the fictional synthetic demo issuer.
ISSUERS = [
    ("Apple Inc.", "AAPL", "Technology", "US"),
    ("Microsoft Corporation", "MSFT", "Technology", "US"),
    ("NVIDIA Corporation", "NVDA", "Technology", "US"),
    ("Tesla Inc.", "TSLA", "Consumer Discretionary", "US"),
    ("JPMorgan Chase & Co.", "JPM", "Financials", "US"),
    ("Goldman Sachs Group Inc.", "GS", "Financials", "US"),
    ("Bank of America Corporation", "BAC", "Financials", "US"),
    ("Citigroup Inc.", "C", "Financials", "US"),
    ("Wells Fargo & Company", "WFC", "Financials", "US"),
    ("Exxon Mobil Corporation", "XOM", "Energy", "US"),
    ("Chevron Corporation", "CVX", "Energy", "US"),
    ("Johnson & Johnson", "JNJ", "Healthcare", "US"),
    ("Pfizer Inc.", "PFE", "Healthcare", "US"),
    ("UnitedHealth Group Incorporated", "UNH", "Healthcare", "US"),
    ("Visa Inc.", "V", "Financials", "US"),
    ("Mastercard Incorporated", "MA", "Financials", "US"),
    ("Procter & Gamble Company", "PG", "Consumer Staples", "US"),
    ("The Coca-Cola Company", "KO", "Consumer Staples", "US"),
    ("Walmart Inc.", "WMT", "Consumer Staples", "US"),
    ("The Boeing Company", "BA", "Industrials", "US"),
    ("Intel Corporation", "INTC", "Technology", "US"),
    ("Advanced Micro Devices Inc.", "AMD", "Technology", "US"),
    ("Taiwan Semiconductor Manufacturing Company Limited", "TSM", "Technology", "TW"),
    ("Sempra", "SRE", "Utilities", "US"),
    ("Gevo Inc.", "GEVO", "Energy", "US"),
    ("The Walt Disney Company", "DIS", "Communication Services", "US"),
]
ZZDEMO = ("Fictional Demo Holdings Inc. (SYNTHETIC DEMO COMPANY - NOT REAL)", "ZZDEMO", "Financials", "US")

RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B"]
SYNTHETIC_COUNTERPARTIES = [
    ("Synthetic Street Bank A (SYNTHETIC COUNTERPARTY)", "Rates", "US"),
    ("Synthetic Street Bank B (SYNTHETIC COUNTERPARTY)", "Rates", "GB"),
    ("Synthetic Global Markets C (SYNTHETIC COUNTERPARTY)", "Rates", "DE"),
]
FX_COUNTERPARTIES_AND_CURRENCIES = [
    ("Synthetic FX Desk A (SYNTHETIC COUNTERPARTY)", "GB", "GBP"),
    ("Synthetic FX Desk B (SYNTHETIC COUNTERPARTY)", "DE", "EUR"),
    ("Synthetic FX Desk C (SYNTHETIC COUNTERPARTY)", "JP", "JPY"),
    ("Synthetic FX Desk D (SYNTHETIC COUNTERPARTY)", "CH", "CHF"),
    ("Synthetic FX Desk E (SYNTHETIC COUNTERPARTY)", "CA", "CAD"),
    ("Synthetic FX Desk F (SYNTHETIC COUNTERPARTY)", "AU", "AUD"),
]


def _rating_for(risk_tier: str) -> str:
    if risk_tier == "high_grade":
        return random.choice(["AAA", "AA", "A"])
    if risk_tier == "mid_grade":
        return random.choice(["A", "BBB"])
    return random.choice(["BB", "B"])  # speculative_grade


def _pd_lgd_for(risk_tier: str):
    if risk_tier == "high_grade":
        return round(random.uniform(0.003, 0.012), 4), round(random.uniform(0.30, 0.40), 3)
    if risk_tier == "mid_grade":
        return round(random.uniform(0.012, 0.030), 4), round(random.uniform(0.35, 0.45), 3)
    return round(random.uniform(0.035, 0.08), 4), round(random.uniform(0.45, 0.60), 3)  # speculative_grade


def build_rows():
    rows = []
    counter = {"BOND": 0, "LOAN": 0, "EQTY": 0, "SWAP": 0, "FXFW": 0}

    def next_id(prefix):
        counter[prefix] += 1
        return f"{prefix}{counter[prefix]:04d}"

    # --- Corporate bonds: 13 single-issuer bonds + 2 ZZDEMO bonds = 15 ---
    bond_issuers = ISSUERS[:13]
    for issuer, ticker, sector, country in bond_issuers:
        risk_tier = "high_grade" if sector in ("Technology", "Healthcare", "Consumer Staples") else "mid_grade"
        rows.append({
            "asset_id": next_id("BOND"), "asset_type": "CORPORATE_BOND",
            "issuer": issuer, "ticker": ticker, "sector": sector, "country": country, "currency": "USD",
            "notional": round(random.uniform(10_000_000, 40_000_000), 2),
            "market_value": None,  # filled in below, close to notional
            "rating": _rating_for(risk_tier),
            "duration": round(random.uniform(3.0, 10.0), 2),
            "convexity": round(random.uniform(0.1, 0.7), 3),
            "pd": "", "lgd": "", "beta": "", "dv01": "", "fx_sensitivity": "",
        })
    for i in range(2):
        rows.append({
            "asset_id": next_id("BOND"), "asset_type": "CORPORATE_BOND",
            "issuer": ZZDEMO[0], "ticker": ZZDEMO[1], "sector": ZZDEMO[2], "country": ZZDEMO[3], "currency": "USD",
            "notional": round(random.uniform(15_000_000, 25_000_000), 2),
            "market_value": None,
            "rating": "BB",
            "duration": round(random.uniform(4.0, 8.0), 2),
            "convexity": round(random.uniform(0.2, 0.5), 3),
            "pd": "", "lgd": "", "beta": "", "dv01": "", "fx_sensitivity": "",
        })
    for row in rows:
        if row["asset_type"] == "CORPORATE_BOND":
            row["market_value"] = round(row["notional"] * random.uniform(0.95, 1.03), 2)

    # --- Corporate loans: 11 single-issuer loans + 1 ZZDEMO loan = 12 ---
    loan_issuers = [ISSUERS[4], ISSUERS[5], ISSUERS[6], ISSUERS[7], ISSUERS[8],  # financials
                    ISSUERS[9], ISSUERS[10], ISSUERS[19], ISSUERS[20], ISSUERS[21], ISSUERS[24]]  # energy/industrials/tech/gevo
    for issuer, ticker, sector, country in loan_issuers:
        risk_tier = "high_grade" if sector == "Financials" else "mid_grade"
        pd, lgd = _pd_lgd_for(risk_tier)
        notional = round(random.uniform(5_000_000, 30_000_000), 2)
        rows.append({
            "asset_id": next_id("LOAN"), "asset_type": "CORPORATE_LOAN",
            "issuer": issuer, "ticker": ticker, "sector": sector, "country": country, "currency": "USD",
            "notional": notional,
            "market_value": round(notional * random.uniform(0.97, 1.0), 2),
            "rating": _rating_for(risk_tier),
            "duration": "", "convexity": "",
            "pd": pd, "lgd": lgd,
            "beta": "", "dv01": "", "fx_sensitivity": "",
        })
    pd, lgd = _pd_lgd_for("speculative_grade")
    zzdemo_loan_notional = round(random.uniform(8_000_000, 15_000_000), 2)
    rows.append({
        "asset_id": next_id("LOAN"), "asset_type": "CORPORATE_LOAN",
        "issuer": ZZDEMO[0], "ticker": ZZDEMO[1], "sector": ZZDEMO[2], "country": ZZDEMO[3], "currency": "USD",
        "notional": zzdemo_loan_notional,
        "market_value": round(zzdemo_loan_notional * 0.98, 2),
        "rating": "B",
        "duration": "", "convexity": "",
        "pd": pd, "lgd": lgd,
        "beta": "", "dv01": "", "fx_sensitivity": "",
    })

    # --- Equities: 11 single-issuer equities + 1 ZZDEMO equity = 12 ---
    equity_issuers = [ISSUERS[0], ISSUERS[1], ISSUERS[2], ISSUERS[3], ISSUERS[11],
                      ISSUERS[12], ISSUERS[13], ISSUERS[14], ISSUERS[15], ISSUERS[17], ISSUERS[25]]
    for issuer, ticker, sector, country in equity_issuers:
        beta = round(random.uniform(1.1, 1.8), 2) if sector == "Technology" else round(random.uniform(0.5, 1.2), 2)
        rows.append({
            "asset_id": next_id("EQTY"), "asset_type": "EQUITY",
            "issuer": issuer, "ticker": ticker, "sector": sector, "country": country, "currency": "USD",
            "notional": "",
            "market_value": round(random.uniform(2_000_000, 25_000_000), 2),
            "rating": "", "duration": "", "convexity": "", "pd": "", "lgd": "",
            "beta": beta, "dv01": "", "fx_sensitivity": "",
        })
    rows.append({
        "asset_id": next_id("EQTY"), "asset_type": "EQUITY",
        "issuer": ZZDEMO[0], "ticker": ZZDEMO[1], "sector": ZZDEMO[2], "country": ZZDEMO[3], "currency": "USD",
        "notional": "",
        "market_value": round(random.uniform(3_000_000, 8_000_000), 2),
        "rating": "", "duration": "", "convexity": "", "pd": "", "lgd": "",
        "beta": round(random.uniform(1.2, 1.6), 2), "dv01": "", "fx_sensitivity": "",
    })

    # --- Interest-rate swaps: 6, synthetic counterparties, mixed payer/receiver ---
    for i in range(6):
        issuer, sector, country = SYNTHETIC_COUNTERPARTIES[i % len(SYNTHETIC_COUNTERPARTIES)]
        notional = round(random.uniform(20_000_000, 100_000_000), 2)
        sign = 1 if i % 2 == 0 else -1  # alternate receive-fixed / pay-fixed exposure
        rows.append({
            "asset_id": next_id("SWAP"), "asset_type": "INTEREST_RATE_SWAP",
            "issuer": issuer, "ticker": "", "sector": sector, "country": country, "currency": "USD",
            "notional": notional,
            "market_value": round(random.uniform(-3_000_000, 3_000_000), 2),
            "rating": "", "duration": "", "convexity": "", "pd": "", "lgd": "", "beta": "",
            "dv01": round(sign * notional * random.uniform(0.00035, 0.00055), 2),
            "fx_sensitivity": "",
        })

    # --- FX forwards: 6, synthetic counterparties, six different currencies ---
    for issuer, country, currency in FX_COUNTERPARTIES_AND_CURRENCIES:
        notional = round(random.uniform(5_000_000, 30_000_000), 2)
        rows.append({
            "asset_id": next_id("FXFW"), "asset_type": "FX_FORWARD",
            "issuer": issuer, "ticker": "", "sector": "FX", "country": country, "currency": currency,
            "notional": notional,
            "market_value": round(random.uniform(-1_000_000, 1_000_000), 2),
            "rating": "", "duration": "", "convexity": "", "pd": "", "lgd": "", "beta": "", "dv01": "",
            "fx_sensitivity": notional,
        })

    return rows


def main() -> None:
    rows = build_rows()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} synthetic portfolio rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

"""Calibrate the lineup/form news layer on the 2018 + 2022 World Cups — for BOTH base
models: Dixon-Coles and the hybrid random forest.

For every team-match, base expected goals come from the pre-tournament model (DC fit on
the 8y window; RF trained only on tournaments strictly before the calibration year),
features from official lineups/goals of *earlier* matches only, target = goals actually
scored. Coefficients are fit separately per base model (they act on that model's
residual structure) and saved to paper/news_coefficients.json.

    python scripts/calibrate_news_layer.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wc_trader.data.lineups import fetch_lineups, load_lineups          # noqa: E402
from wc_trader.data.results import load_results                        # noqa: E402
from wc_trader.experiment import (                                      # noqa: E402
    TRAIN_YEARS,
    build_frame,
    build_snapshots,
    covariate_lookups,
    wc_matches,
)
from wc_trader.model.groll_rf import HybridRF, rf_lambda_table          # noqa: E402
from wc_trader.model.lineup_adjust import (                             # noqa: E402
    LambdaGridAdapter,
    NewsFeatureBuilder,
    fit_coefficients,
)

CAL_YEARS = (2018, 2022)


def samples_for(base, recs) -> list[dict]:
    builder = NewsFeatureBuilder(base, recs)
    gv = np.arange(11)
    out = []
    # per_record pairs each match with the features computed from ITS OWN past —
    # immune to repeated-pairing collisions.
    for rec, feats in builder.per_record:
        h, a = rec["home"]["team"], rec["away"]["team"]
        try:
            g = base.score_grid(h, a, neutral=True)
        except KeyError:
            continue
        out.append({"lam_h": float((g.sum(axis=1) * gv).sum()),
                    "lam_a": float((g.sum(axis=0) * gv).sum()),
                    "news_h": feats[h], "news_a": feats[a],
                    "goals_h": len(rec["home"]["goals"]),
                    "goals_a": len(rec["away"]["goals"])})
    return out


def main() -> None:
    df = load_results()
    ranks, gdp, pop = covariate_lookups(df)
    missing: dict = {}

    frames, dcs = {}, {}
    for y in TRAIN_YEARS:
        frames[y], dcs[y] = build_frame(df, y, ranks, gdp, pop, missing)

    dc_samples, rf_samples = [], []
    for year in CAL_YEARS:
        fetch_lineups(year)
        recs = load_lineups(year)

        dc_samples.extend(samples_for(dcs[year], recs))

        pre = [y for y in TRAIN_YEARS if y < year]
        rf = HybridRF().fit(pd.concat([frames[y] for y in pre], ignore_index=True))
        teams = sorted(set(wc_matches(df, year).home_team) | set(wc_matches(df, year).away_team))
        snaps = build_snapshots(teams, year, dcs[year], ranks, gdp, pop, missing)
        adapter = LambdaGridAdapter(rf_lambda_table(rf, snaps))
        rf_samples.extend(samples_for(adapter, recs))
        print(f"{year}: DC samples so far {len(dc_samples)}, RF {len(rf_samples)} "
              f"(RF trained on {pre})")

    out = {}
    for name, samples in (("dc", dc_samples), ("rf", rf_samples)):
        c = fit_coefficients(samples)
        out[name] = {"beta_own": c.beta_own, "beta_opp": c.beta_opp, "gamma": c.gamma,
                     "n_matches": len(samples)}
        print(f"{name.upper()}: beta_own {c.beta_own:+.4f}  beta_opp {c.beta_opp:+.4f}  "
              f"gamma {c.gamma:+.4f}")

    out["calibrated_on"] = list(CAL_YEARS)
    Path("paper").mkdir(exist_ok=True)
    Path("paper/news_coefficients.json").write_text(json.dumps(out, indent=1))
    print("saved -> paper/news_coefficients.json")


if __name__ == "__main__":
    main()

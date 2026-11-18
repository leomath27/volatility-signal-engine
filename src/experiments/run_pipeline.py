from __future__ import annotations

import argparse
import json
import logging

from src import config
from src.backtest.vol_backtest import run_backtest
from src.models.train_xgb import DEFAULT_HORIZONS, train_many


def main() -> None:
    ap = argparse.ArgumentParser(description="Train 1/3/5-day vol models, evaluate walk-forward, run proxy backtest")
    ap.add_argument("--symbols", nargs="+", default=list(config.DEFAULT_SYMBOLS))
    ap.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    ap.add_argument("--no-refresh", action="store_true", help="reuse cached market data")
    ap.add_argument("--iv-markup", type=float, default=1.0, help="scale trailing vol to mimic IV > RV in the backtest")
    ap.add_argument("--cost", type=float, default=0.02, help="round-trip cost as a fraction of premium")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    summaries, backtests = {}, {}
    for sym in args.symbols:
        try:
            summaries[sym] = train_many(sym, args.horizons, refresh_data=not args.no_refresh)
            for h in args.horizons:
                backtests[f"{sym}_h{h}"] = run_backtest(sym, h, args.iv_markup, args.cost)
        except Exception as exc:
            logging.error("Skipping %s: %s", sym, exc)

    print("\nWalk-forward out-of-sample forecast quality (QLIKE: lower is better; gain = % better than baseline)")
    print(f"{'symbol':8}{'h':>3}{'xgb':>9}{'har':>9}{'naive':>9}{'vs naive':>10}{'vs har':>9}{'edge_ic':>9}")
    for sym, s in summaries.items():
        for r in s["results"]:
            m = r["metrics"]
            print(f"{sym:8}{r['horizon_days']:>3}{m['xgb_qlike']:9.4f}{m['har_qlike']:9.4f}{m['naive_qlike']:9.4f}"
                  f"{m['qlike_gain_vs_naive_pct']:+9.1f}%{m['qlike_gain_vs_har_pct']:+8.1f}%{m['xgb_edge_ic']:9.3f}")

    print(f"\nProxy options backtest (iv_markup={args.iv_markup}, cost={args.cost:.0%} of premium)")
    print(f"{'run':14}{'trades':>7}{'hit':>7}{'avg %spot':>11}{'sharpe':>8}{'worst %':>9}   vs always-long-straddle avg")
    for k, b in backtests.items():
        if not b.get("trade_count"):
            print(f"{k:14}{0:>7}")
            continue
        base = b["benchmarks"]["always_long_straddle"].get("avg_pnl_pct_spot", float("nan"))
        sh = b["sharpe_ann"] if b["sharpe_ann"] is not None else float("nan")
        print(f"{k:14}{b['trade_count']:>7}{b['hit_rate']:7.1%}{b['avg_pnl_pct_spot']:11.3f}{sh:8.2f}"
              f"{b['worst_trade_pct_spot']:9.2f}   {base:+.3f}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (config.MODELS_DIR / "pipeline_summary.json").write_text(
        json.dumps({"training": summaries, "backtest": backtests}, indent=2, default=str))


if __name__ == "__main__":
    main()

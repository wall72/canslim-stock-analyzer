#!/usr/bin/env python3
# CANSLIM-style early-stage growth screener template
# Data source: yfinance (free, convenient, but not institution-grade)

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# =========================
# User settings
# =========================
TICKERS_US = [
    "NVDA", "AMD", "SMCI", "ARM", "PLTR", "SNOW", "CRWD", "DDOG", "NET", "SHOP",
    "APP", "CELH", "DUOL", "ELF", "ANET", "ONON", "TSM", "AVGO", "UBER", "MNDY"
]

TICKERS_KR_EXAMPLE = [
    # Yahoo Finance suffix examples for Korea
    "005930.KS",  # Samsung Electronics
    "000660.KS",  # SK hynix
    "035420.KS",  # NAVER
    "035720.KQ",  # Kakao (check suffix in your environment if needed)
]

USE_TICKERS = TICKERS_US
BENCHMARK = "^IXIC"  # Nasdaq Composite; for Korea you can try ^KS11
LOOKBACK_DAYS = 260

RULES = {
    "min_qtr_revenue_yoy": 20.0,
    "min_annual_revenue_growth": 15.0,
    "min_price_vs_52w_high": -15.0,   # within -15% of 52w high
    "min_avg_dollar_volume_usd": 20_000_000,
    "min_relative_strength_vs_benchmark": 1.0,  # >1 means better than benchmark over 6m
    "max_market_cap_usd": 50_000_000_000,       # early-stage bias
}

# =========================
# Helpers
# =========================
def safe_get(d: dict, *keys, default=np.nan):
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def pct_change(new, old):
    try:
        if pd.isna(new) or pd.isna(old) or old == 0:
            return np.nan
        return (new / old - 1.0) * 100.0
    except Exception:
        return np.nan


def get_history(ticker: str, period="1y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    return df.dropna(how="all")


def compute_price_metrics(px: pd.DataFrame, benchmark_px: pd.DataFrame) -> Dict[str, float]:
    if px.empty:
        return {}
    close = px["Close"].dropna()
    vol = px["Volume"].dropna()
    latest = float(close.iloc[-1])
    high_52w = float(close.tail(252).max()) if len(close) >= 20 else float(close.max())
    pct_from_high = (latest / high_52w - 1.0) * 100.0 if high_52w else np.nan
    avg_dollar_volume = float((px["Close"] * px["Volume"]).tail(50).mean())

    rs_6m = np.nan
    if len(close) >= 126 and not benchmark_px.empty:
        bclose = benchmark_px["Close"].dropna()
        aligned = pd.concat([close, bclose], axis=1, join="inner").dropna()
        if len(aligned) >= 126:
            stock_ret = aligned.iloc[-1, 0] / aligned.iloc[-126, 0]
            bench_ret = aligned.iloc[-1, 1] / aligned.iloc[-126, 1]
            rs_6m = stock_ret / bench_ret if bench_ret else np.nan

    return {
        "price": latest,
        "high_52w": high_52w,
        "pct_from_52w_high": pct_from_high,
        "avg_dollar_volume_50d": avg_dollar_volume,
        "rs_vs_benchmark_6m": rs_6m,
    }


def compute_financial_metrics(t: yf.Ticker) -> Dict[str, float]:
    info = {}
    try:
        info = t.get_info()
    except Exception:
        try:
            info = t.info
        except Exception:
            info = {}

    market_cap = safe_get(info, "marketCap")
    sector = safe_get(info, "sector", default=None)
    industry = safe_get(info, "industry", default=None)
    name = safe_get(info, "shortName", "longName", default=None)

    q_rev_yoy = np.nan
    annual_rev_growth = np.nan

    try:
        qf = t.quarterly_financials
        if qf is not None and not qf.empty:
            idx = [str(x).lower() for x in qf.index]
            revenue_row = None
            for candidate in ["total revenue", "operating revenue", "revenue"]:
                if candidate in idx:
                    revenue_row = qf.iloc[idx.index(candidate)]
                    break
            if revenue_row is not None and len(revenue_row) >= 5:
                vals = revenue_row.sort_index(ascending=True).astype(float)
                q_rev_yoy = pct_change(vals.iloc[-1], vals.iloc[-5])
    except Exception:
        pass

    try:
        fin = t.financials
        if fin is not None and not fin.empty:
            idx = [str(x).lower() for x in fin.index]
            revenue_row = None
            for candidate in ["total revenue", "operating revenue", "revenue"]:
                if candidate in idx:
                    revenue_row = fin.iloc[idx.index(candidate)]
                    break
            if revenue_row is not None and len(revenue_row) >= 2:
                vals = revenue_row.sort_index(ascending=True).astype(float)
                annual_rev_growth = pct_change(vals.iloc[-1], vals.iloc[-2])
    except Exception:
        pass

    return {
        "name": name,
        "sector": sector,
        "industry": industry,
        "market_cap": market_cap,
        "qtr_revenue_yoy": q_rev_yoy,
        "annual_revenue_growth": annual_rev_growth,
    }


def pass_rules(row: pd.Series, rules: dict) -> Dict[str, bool]:
    return {
        "qtr_growth_ok": pd.notna(row["qtr_revenue_yoy"]) and row["qtr_revenue_yoy"] >= rules["min_qtr_revenue_yoy"],
        "annual_growth_ok": pd.notna(row["annual_revenue_growth"]) and row["annual_revenue_growth"] >= rules["min_annual_revenue_growth"],
        "near_high_ok": pd.notna(row["pct_from_52w_high"]) and row["pct_from_52w_high"] >= rules["min_price_vs_52w_high"],
        "liquidity_ok": pd.notna(row["avg_dollar_volume_50d"]) and row["avg_dollar_volume_50d"] >= rules["min_avg_dollar_volume_usd"],
        "rs_ok": pd.notna(row["rs_vs_benchmark_6m"]) and row["rs_vs_benchmark_6m"] >= rules["min_relative_strength_vs_benchmark"],
        "size_ok": pd.notna(row["market_cap"]) and row["market_cap"] <= rules["max_market_cap_usd"],
    }


def score_row(row: pd.Series) -> float:
    score = 0.0
    checks = pass_rules(row, RULES)
    score += sum(checks.values()) * 15
    if pd.notna(row["qtr_revenue_yoy"]):
        score += min(max(row["qtr_revenue_yoy"], 0), 100) * 0.15
    if pd.notna(row["annual_revenue_growth"]):
        score += min(max(row["annual_revenue_growth"], 0), 80) * 0.12
    if pd.notna(row["pct_from_52w_high"]):
        score += max(0, 20 + row["pct_from_52w_high"]) * 0.8
    if pd.notna(row["rs_vs_benchmark_6m"]):
        score += min(row["rs_vs_benchmark_6m"], 2.0) * 10
    return round(score, 2)


def run_screen(tickers: List[str]) -> pd.DataFrame:
    benchmark_px = get_history(BENCHMARK, period="1y")
    rows = []
    for tk in tickers:
        try:
            t = yf.Ticker(tk)
            px = get_history(tk, period="1y")
            if px.empty:
                continue
            pm = compute_price_metrics(px, benchmark_px)
            fm = compute_financial_metrics(t)
            row = {"ticker": tk, **fm, **pm}
            checks = pass_rules(pd.Series(row), RULES)
            row.update(checks)
            row["passed_rules"] = int(sum(checks.values()))
            row["score"] = score_row(pd.Series(row))
            rows.append(row)
        except Exception as e:
            rows.append({"ticker": tk, "error": str(e)})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    ordered = [
        "ticker", "name", "sector", "industry", "market_cap",
        "qtr_revenue_yoy", "annual_revenue_growth",
        "price", "high_52w", "pct_from_52w_high",
        "avg_dollar_volume_50d", "rs_vs_benchmark_6m",
        "qtr_growth_ok", "annual_growth_ok", "near_high_ok",
        "liquidity_ok", "rs_ok", "size_ok", "passed_rules", "score"
    ]
    existing = [c for c in ordered if c in df.columns]
    others = [c for c in df.columns if c not in existing]
    df = df[existing + others]
    df = df.sort_values(["passed_rules", "score"], ascending=[False, False])
    return df


def main():
    df = run_screen(USE_TICKERS)
    if df.empty:
        print("No data returned. Check ticker list and internet access.")
        return

    numeric_cols = [
        "market_cap", "qtr_revenue_yoy", "annual_revenue_growth", "price",
        "high_52w", "pct_from_52w_high", "avg_dollar_volume_50d",
        "rs_vs_benchmark_6m", "score"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    out_csv = "canslim_screen_results.csv"
    df.to_csv(out_csv, index=False)

    winners = df[df["passed_rules"] >= 5].copy() if "passed_rules" in df.columns else df.head(10)
    print("Top candidates:")
    print(winners[[c for c in ["ticker", "passed_rules", "score", "qtr_revenue_yoy", "annual_revenue_growth", "pct_from_52w_high", "rs_vs_benchmark_6m"] if c in winners.columns]].head(15).to_string(index=False))
    print(f"\nSaved full results to {out_csv}")


if __name__ == "__main__":
    main()

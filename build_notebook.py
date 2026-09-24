"""Build the AlgoChowk NIFTY event-study notebook."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": [],
}


def md(source: str) -> None:
    nb["cells"].append(
        {
            "id": uuid.uuid4().hex[:8],
            "cell_type": "markdown",
            "metadata": {},
            "source": source.strip() + "\n",
        }
    )


def code(source: str) -> None:
    nb["cells"].append(
        {
            "id": uuid.uuid4().hex[:8],
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": source.strip() + "\n",
        }
    )


md(
    """
# NIFTY 50 Event Study

**Hypothesis:** After a *significant* one-day close-to-close fall in NIFTY 50, the index tends to recover over the next few trading days.

This notebook is the research engine for that hypothesis. It does not hunt for the prettiest parameter combination. Primary rules are fixed first, then robustness, out-of-sample, and a simple cost-aware backtest are used to try to break the result.

**Data used:** the NSE daily OHLC file in this folder (`NIFTY 50-23-09-2025-to-23-09-2026.csv`). Coverage is about one year, so sample size is a first-class limitation and is treated as such throughout.
"""
)

md(
    """
## 1. Research question

| Item | Choice |
|---|---|
| Question | After NIFTY 50 falls by at least 2% close-to-close, are next-day-open to subsequent-close returns positive on average? |
| Null | Mean forward return after events is ≤ 0 |
| Alternative | Mean forward return after events is > 0 |
| Economic companion | Do event-window returns differ from ordinary NIFTY holding-period returns? |

**Failure conditions (pre-committed):** the hypothesis is treated as unsupported if (a) forward returns are not consistently positive across the primary holding periods, (b) event-period returns do not materially differ from the non-event baseline, or (c) the relationship disappears out of sample after parameters are frozen.
"""
)

md("## 2. Configuration")

code(
    r"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("display.max_rows", 40)
pd.set_option("display.float_format", lambda x: f"{x:,.6f}")
plt.rcParams["figure.figsize"] = (9, 4)
plt.rcParams["axes.grid"] = True
np.random.seed(42)

DATA_PATH = next(Path(".").glob("NIFTY 50*.csv"))

# Primary research rules — chosen on rationale, then frozen for out-of-sample.
EVENT_THRESHOLD = -0.02          # significant fall: daily close-to-close return <= -2%
HOLDING_DAYS = 5                 # primary horizon
FORWARD_HORIZONS = (1, 3, 5)     # days from entry
COOLDOWN_DAYS = 3                # ignore new events for N trading days after an event
OOS_START = pd.Timestamp("2026-04-01")  # development: before this date

# Backtest frictions (round-trip applied at exit as a simple cost model).
TRANSACTION_COST = 0.0005        # 5 bps round-trip
SLIPPAGE = 0.0005                # 5 bps round-trip (cannot assume fill at the printed open/close)

BOOTSTRAP_REPS = 10_000
ALPHA = 0.05

print("Data file:", DATA_PATH.resolve())
print("Primary threshold:", EVENT_THRESHOLD)
print("Primary holding days:", HOLDING_DAYS)
print("Cooldown:", COOLDOWN_DAYS)
print("OOS start:", OOS_START.date())
"""
)

md(
    """
## 3. Load data

NSE files in this export are newest-first, with trailing spaces in column names. Dates are parsed after stripping. Empty trailing rows are dropped.
"""
)

code(
    r"""
raw = pd.read_csv(DATA_PATH)
print("Raw columns:", list(raw.columns))
print("Raw shape:", raw.shape)
raw.head()
"""
)

code(
    r"""
df = raw.copy()
df.columns = df.columns.str.strip()

ohlc_cols = ["Open", "High", "Low", "Close"]
keep = ["Date"] + ohlc_cols
df = df[keep].copy()
df = df.dropna(how="all")

for c in ohlc_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["Date_raw"] = df["Date"].astype(str).str.strip()
df["Date"] = pd.to_datetime(df["Date_raw"], format="%d-%b-%y", errors="coerce")

print("Parsed rows:", len(df))
print("Unparseable dates:", int(df["Date"].isna().sum()))
df.head()
"""
)

md(
    """
## 4. Data validation

Weekends are not treated as missing data. Gaps of 3 calendar days over a weekend, or longer around exchange holidays, are expected. Extreme *internally consistent* returns are retained because they are economically meaningful and may be the events themselves.

Two records in this file are not internally consistent / not parseable. They are investigated, not silently deleted.
"""
)

code(
    r"""
print("=== Missing Date after parse ===")
print(int(df["Date"].isna().sum()))
print(df.loc[df["Date"].isna(), ["Date_raw", *ohlc_cols]])

print("\n=== Duplicate Date (among parsed) ===")
parsed = df.dropna(subset=["Date"])
print(int(parsed["Date"].duplicated().sum()))

print("\n=== Missing OHLC ===")
print(df[ohlc_cols].isna().sum())
"""
)

code(
    r"""
# Logical OHLC relationships
ohlc_ok = (
    (df["High"] >= df["Open"])
    & (df["High"] >= df["Close"])
    & (df["Low"] <= df["Open"])
    & (df["Low"] <= df["Close"])
    & (df["High"] >= df["Low"])
    & df[ohlc_cols].notna().all(axis=1)
)
print("Invalid OHLC rows:", int((~ohlc_ok).sum()))
df.loc[~ohlc_ok, ["Date_raw", "Date", *ohlc_cols]]
"""
)

md(
    """
### Repair log

| Issue | Evidence | Action |
|---|---|---|
| `Close = 6` on 11-Sep-2026 | High/Low around 23,200–23,400; Close cannot be 6. Fails High ≥ Close and Low ≤ Close. | Drop. This is corruption, not a crash. |
| Date `B38-1` | Excel-style artefact. Neighbours are 04-Aug-2026 and 31-Jul-2026. 03-Aug-2026 is the missing Monday. OHLC is internally consistent. | Assign 03-Aug-2026. |
| Blank trailing row | All fields empty | Drop. |
"""
)

code(
    r"""
repair_notes = []

# Infer the unparseable date from the surrounding (file-order) neighbours.
bad_date_idx = df.index[df["Date"].isna() & df[ohlc_cols].notna().all(axis=1)]
for i in bad_date_idx:
    prev_parsed = df.loc[: i - 1, "Date"].dropna()
    next_parsed = df.loc[i + 1 :, "Date"].dropna()
    # File is newest-first: previous row is a later calendar date, next row is an earlier one.
    later = prev_parsed.iloc[-1] if len(prev_parsed) else pd.NaT
    earlier = next_parsed.iloc[0] if len(next_parsed) else pd.NaT
    inferred = pd.Timestamp("2026-08-03")
    df.loc[i, "Date"] = inferred
    repair_notes.append(
        f"Row {i}: Date_raw={df.loc[i, 'Date_raw']!r} inferred as {inferred.date()} "
        f"(neighbours later={None if pd.isna(later) else later.date()}, "
        f"earlier={None if pd.isna(earlier) else earlier.date()})"
    )

invalid_mask = ~ohlc_ok
n_drop_ohlc = int(invalid_mask.sum())
if n_drop_ohlc:
    repair_notes.append(f"Dropped {n_drop_ohlc} row(s) failing OHLC inequalities (corrupted Close).")
    df = df.loc[~invalid_mask].copy()

df = df.dropna(subset=["Date", *ohlc_cols]).copy()
df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="first")
df = df.reset_index(drop=True)

print("Repair notes:")
for note in repair_notes:
    print("-", note)

print("\nMonotonic dates:", bool(df["Date"].is_monotonic_increasing))
print("Duplicate dates:", int(df["Date"].duplicated().sum()))
print("Start:", df["Date"].min().date())
print("End:", df["Date"].max().date())
print("Trading observations:", len(df))
"""
)

code(
    r"""
# Calendar gaps: weekdays with no row. Weekends are expected; weekday gaps are holidays or missing data.
full_bdays = pd.bdate_range(df["Date"].min(), df["Date"].max(), freq="C")
missing_weekdays = full_bdays.difference(df["Date"])
print("Weekday dates with no observation:", len(missing_weekdays))
print(missing_weekdays[:30].strftime("%Y-%m-%d").tolist())
if len(missing_weekdays) > 30:
    print("...")

gaps = df["Date"].diff().dt.days
print("\nLargest calendar gaps (days between consecutive rows):")
display(df.assign(gap_days=gaps).nlargest(8, "gap_days")[["Date", "gap_days", "Close"]])
"""
)

md("## 5. Calculate returns")

code(
    r"""
df["daily_return"] = df["Close"].pct_change()

print(df["daily_return"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))
print("\nLargest down days:")
display(df.nsmallest(10, "daily_return")[["Date", "Open", "High", "Low", "Close", "daily_return"]])
print("Largest up days:")
display(df.nlargest(8, "daily_return")[["Date", "Close", "daily_return"]])
"""
)

md(
    """
Extreme but internally consistent observations are **retained**. A −3% NIFTY day is unusual; it is not automatically an error, and it is part of the event definition.
"""
)

md(
    r"""
## 6–8. Event engine: detection, de-clustering, forward returns

**Look-ahead control.** A -2% close-to-close return is only known after that day's close. Entry is therefore the **next trading day's open**, not that day's close.

For an event on day t:

`R_h = Close[t+h] / Open[t+1] - 1`

**De-clustering.** Consecutive crash days are not independent events. After an event is kept, the next `COOLDOWN_DAYS` trading days cannot start a new event.
"""
)

code(
    r"""
def decluster_events(event_positions: np.ndarray, cooldown: int) -> np.ndarray:
    kept = []
    last = -10**9
    for pos in event_positions:
        if pos - last > cooldown:
            kept.append(pos)
            last = pos
    return np.asarray(kept, dtype=int)


def build_event_table(
    prices: pd.DataFrame,
    threshold: float,
    cooldown: int,
    horizons: tuple[int, ...] = FORWARD_HORIZONS,
) -> pd.DataFrame:
    # Map daily OHLC into an event-study table. Positions are integer index locations.
    ret = prices["Close"].pct_change()
    raw_pos = np.flatnonzero((ret <= threshold).to_numpy())
    # Need t+max(horizons) to exist, and entry at t+1.
    max_h = max(horizons)
    raw_pos = raw_pos[(raw_pos + max_h) < len(prices)]
    kept_pos = decluster_events(raw_pos, cooldown)

    rows = []
    for t in kept_pos:
        entry = prices.loc[t + 1, "Open"]
        row = {
            "event_date": prices.loc[t, "Date"],
            "event_return": ret.iloc[t],
            "entry_date": prices.loc[t + 1, "Date"],
            "entry": entry,
        }
        for h in horizons:
            row[f"return_{h}d"] = prices.loc[t + h, "Close"] / entry - 1
            row[f"exit_date_{h}d"] = prices.loc[t + h, "Date"]
            row[f"exit_{h}d"] = prices.loc[t + h, "Close"]
        rows.append(row)
    events = pd.DataFrame(rows)
    events.attrs["raw_event_count"] = int(len(raw_pos))
    events.attrs["threshold"] = threshold
    events.attrs["cooldown"] = cooldown
    return events


def summarise_returns(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if len(s) == 0:
        return pd.Series(dtype=float)
    return pd.Series(
        {
            "n": int(len(s)),
            "mean": s.mean(),
            "median": s.median(),
            "win_rate": (s > 0).mean(),
            "std": s.std(ddof=1),
            "min": s.min(),
            "max": s.max(),
        }
    )


events = build_event_table(df, EVENT_THRESHOLD, COOLDOWN_DAYS)
print("Raw qualifying days (with enough forward data):", events.attrs["raw_event_count"])
print("After cooldown de-clustering:", len(events))
events
"""
)

md("## 9. Descriptive statistics")

code(
    r"""
stat_rows = []
for h in FORWARD_HORIZONS:
    row = summarise_returns(events[f"return_{h}d"])
    row.name = f"{h}D"
    stat_rows.append(row)
event_stats = pd.DataFrame(stat_rows)
event_stats
"""
)

md(
    """
## 10. Distribution

A histogram of primary-horizon event returns shows whether the mean is a typical outcome or is being pulled by a few large winners.
"""
)

code(
    r"""
col = f"return_{HOLDING_DAYS}d"
fig, ax = plt.subplots()
ax.hist(events[col].dropna() * 100, bins=min(20, max(5, len(events))), edgecolor="black")
ax.axvline(0, color="black", linewidth=1)
ax.axvline(events[col].mean() * 100, color="tab:red", linestyle="--", label="mean")
ax.axvline(events[col].median() * 100, color="tab:orange", linestyle=":", label="median")
ax.set_xlabel(f"{HOLDING_DAYS}-day forward return (%)")
ax.set_ylabel("Events")
ax.set_title("Distribution of event forward returns")
ax.legend()
plt.tight_layout()
plt.show()
"""
)

md(
    """
## 11. Statistical evidence

Primary: t-based 95% confidence interval for the mean, plus a one-sided t-test of H0: mean <= 0.

Supporting: percentile bootstrap CI (returns need not be normal).

Statistical significance is not treated as trading significance. A tiny mean with a small p-value can still be useless after costs.
"""
)

code(
    r"""
def mean_ci_t(x: pd.Series, alpha: float = ALPHA):
    x = x.dropna().to_numpy()
    n = len(x)
    if n < 2:
        return {"n": n, "mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_one_sided": np.nan}
    mean = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    tstat = mean / se
    p = 1 - stats.t.cdf(tstat, df=n - 1)
    return {
        "n": n,
        "mean": mean,
        "ci_low": mean - tcrit * se,
        "ci_high": mean + tcrit * se,
        "t_stat": tstat,
        "p_one_sided": p,
    }


def bootstrap_mean_ci(x: pd.Series, reps: int = BOOTSTRAP_REPS, alpha: float = ALPHA, seed: int = 42):
    x = x.dropna().to_numpy()
    n = len(x)
    if n < 2:
        return {"boot_ci_low": np.nan, "boot_ci_high": np.nan}
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(reps, n), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"boot_mean": means.mean(), "boot_ci_low": lo, "boot_ci_high": hi}


evidence_rows = []
for h in FORWARD_HORIZONS:
    t_res = mean_ci_t(events[f"return_{h}d"])
    b_res = bootstrap_mean_ci(events[f"return_{h}d"])
    evidence_rows.append({"horizon": f"{h}D", **t_res, **b_res})

evidence = pd.DataFrame(evidence_rows).set_index("horizon")
evidence
"""
)

code(
    r"""
print("Interpretation keys")
print("- If the 95% CI lies entirely above 0, the sample mean is inconsistently explained by a zero-or-negative true mean under the t model.")
print("- Bootstrap CI is preferred if the histogram is skewed or n is small.")
print("- One-sided p-value tests H0: mean <= 0 vs H1: mean > 0.")
print()
primary = evidence.loc[f"{HOLDING_DAYS}D"]
print(f"Primary {HOLDING_DAYS}D mean: {primary['mean']:.4%}")
print(f"t 95% CI: [{primary['ci_low']:.4%}, {primary['ci_high']:.4%}]")
print(f"bootstrap 95% CI: [{primary['boot_ci_low']:.4%}, {primary['boot_ci_high']:.4%}]")
print(f"one-sided p-value: {primary['p_one_sided']:.4f}")
"""
)

md(
    """
## 12. Baseline comparison

A positive event-window return is not enough. NIFTY can have a positive drift on ordinary days too. The baseline is the same holding-period return, entered at the next day's open, on **non-event** days (days that were not kept as events, still requiring enough future bars).
"""
)

code(
    r"""
def baseline_table(prices: pd.DataFrame, event_dates: pd.Series, horizons=FORWARD_HORIZONS) -> pd.DataFrame:
    event_set = set(pd.to_datetime(event_dates))
    max_h = max(horizons)
    rows = []
    ret = prices["Close"].pct_change()
    for t in range(1, len(prices) - max_h):
        if prices.loc[t, "Date"] in event_set:
            continue
        if pd.isna(ret.iloc[t]):
            continue
        entry = prices.loc[t + 1, "Open"]
        row = {"date": prices.loc[t, "Date"]}
        for h in horizons:
            row[f"return_{h}d"] = prices.loc[t + h, "Close"] / entry - 1
        rows.append(row)
    return pd.DataFrame(rows)


base = baseline_table(df, events["event_date"] if len(events) else pd.Series(dtype="datetime64[ns]"))

cmp_rows = []
for h in FORWARD_HORIZONS:
    e = summarise_returns(events[f"return_{h}d"] if len(events) else pd.Series(dtype=float))
    b = summarise_returns(base[f"return_{h}d"] if len(base) else pd.Series(dtype=float))
    cmp_rows.append(
        {
            "horizon": f"{h}D",
            "event_n": e.get("n", 0),
            "event_mean": e.get("mean", np.nan),
            "event_median": e.get("median", np.nan),
            "event_win_rate": e.get("win_rate", np.nan),
            "base_n": b.get("n", 0),
            "base_mean": b.get("mean", np.nan),
            "base_median": b.get("median", np.nan),
            "base_win_rate": b.get("win_rate", np.nan),
            "mean_gap": e.get("mean", np.nan) - b.get("mean", np.nan),
        }
    )
comparison = pd.DataFrame(cmp_rows).set_index("horizon")
comparison
"""
)

code(
    r"""
# Welch-style comparison of means at the primary horizon (unequal variance).
h = HOLDING_DAYS
if len(events) >= 2 and len(base) >= 2:
    tw = stats.ttest_ind(
        events[f"return_{h}d"].dropna(),
        base[f"return_{h}d"].dropna(),
        equal_var=False,
        alternative="greater",
    )
    print(f"Welch t-test, event mean > baseline mean, {h}D: t={tw.statistic:.3f}, p={tw.pvalue:.4f}")
else:
    print("Not enough observations for a Welch test.")
"""
)

md(
    """
## 13. Robustness

The question is not “which cell is largest?” It is whether the **sign and qualitative gap vs baseline** survive nearby, reasonable choices.

Thresholds: −1.5%, −2.0%, −2.5%, −3.0%.  
Holdings: 1, 3, 5, 10 days.  
Cooldown: 0, 3, 5 days (primary threshold and 5-day hold).
"""
)

code(
    r"""
thresholds = (-0.015, -0.02, -0.025, -0.03)
holdings = (1, 3, 5, 10)


def mean_at(prices, threshold, cooldown, horizon):
    tbl = build_event_table(prices, threshold, cooldown, horizons=(horizon,))
    if len(tbl) == 0:
        return np.nan, 0
    return tbl[f"return_{horizon}d"].mean(), len(tbl)


grid = []
for thr in thresholds:
    row = {"threshold": thr}
    for h in holdings:
        mu, n = mean_at(df, thr, COOLDOWN_DAYS, h)
        row[f"{h}D_mean"] = mu
        row[f"{h}D_n"] = n
    grid.append(row)
robust_grid = pd.DataFrame(grid)
robust_grid
"""
)

code(
    r"""
cooldown_rows = []
for cd in (0, 3, 5):
    tbl = build_event_table(df, EVENT_THRESHOLD, cd, horizons=(HOLDING_DAYS,))
    s = summarise_returns(tbl[f"return_{HOLDING_DAYS}d"] if len(tbl) else pd.Series(dtype=float))
    cooldown_rows.append({"cooldown": cd, "raw_qualifying": tbl.attrs.get("raw_event_count", 0), **s})
cooldown_table = pd.DataFrame(cooldown_rows)
cooldown_table
"""
)

md(
    """
## 14. Out-of-sample

Development window: first observation through 31 March 2026.  
Out-of-sample: 1 April 2026 onward.

Parameters are **not** re-tuned on the later window. Same -2% / 5-day / 3-day cooldown rule.

Events are split by **signal date** on the full sample, so a late-March crash is not discarded merely because the 5-day holding window crosses into April.
"""
)

code(
    r"""
# Split by signal date on the full event table so a late-March event is not
# dropped just because its 5-day holding window crosses 1 Apr 2026.
dev_mask = df["Date"] < OOS_START
oos_mask = df["Date"] >= OOS_START
print("Development bars:", int(dev_mask.sum()), df.loc[dev_mask, "Date"].min().date(), "→", df.loc[dev_mask, "Date"].max().date())
print("Out-of-sample bars:", int(oos_mask.sum()), df.loc[oos_mask, "Date"].min().date(), "→", df.loc[oos_mask, "Date"].max().date())

if len(events):
    dev_events = events[events["event_date"] < OOS_START].copy()
    oos_events = events[events["event_date"] >= OOS_START].copy()
else:
    dev_events = events
    oos_events = events

oos_cmp = pd.DataFrame(
    {
        "development": summarise_returns(dev_events[f"return_{HOLDING_DAYS}d"] if len(dev_events) else pd.Series(dtype=float)),
        "out_of_sample": summarise_returns(oos_events[f"return_{HOLDING_DAYS}d"] if len(oos_events) else pd.Series(dtype=float)),
    }
).T
oos_cmp
"""
)

code(
    r"""
print("Development events")
display(dev_events if len(dev_events) else "None")
print("Out-of-sample events")
display(oos_events if len(oos_events) else "None")
"""
)

md(
    """
## 15. Challenge the result

Reasons the result can be wrong, even if the tables look tidy:

1. **Look-ahead.** Entry is next open, which is the intended fix. Anyone entering at the event close would be cheating.
2. **Print vs fill.** NIFTY cash index is not directly tradable. Futures/ETF fills will differ from the official open.
3. **Costs and slippage.** A 5 bp + 5 bp round-trip haircut is a lower bound, not a live execution model.
4. **Small sample.** One year of daily data produces few −2% days. Means and p-values are fragile.
5. **Overlapping selloffs.** Cooldown helps; it does not create independence.
6. **Regimes.** If almost all events sit inside one March–April drawdown, this is a single-episode study, not a general NIFTY law.
7. **Data quality.** One Close=6 row would have manufactured a −99% “event” if it had been kept.
8. **Data snooping.** The robustness grid is for stability, not for picking a winner after the fact.

**What would reject the hypothesis under this methodology:** event means not consistently positive; event vs baseline gap not material; or the frozen rule failing out of sample.
"""
)

code(
    r"""
print("Event dates (full sample, de-clustered)")
if len(events):
    display(events[["event_date", "event_return", "entry", f"return_{HOLDING_DAYS}d"]])
    span = (events["event_date"].max() - events["event_date"].min()).days
    print(f"Event date span: {span} calendar days")
    print("Share of events in Mar-Apr 2026:",
          ((events["event_date"] >= "2026-03-01") & (events["event_date"] <= "2026-04-30")).mean())
else:
    print("No events under the primary rule.")
"""
)

md(
    """
## 16. Event-driven backtest

Rule (after all research choices are fixed):

1. Significant fall (≤ −2% close-to-close), after cooldown filter.  
2. Buy at the next session open.  
3. Hold `HOLDING_DAYS` trading days.  
4. Sell at that day's close.  
5. Net return = gross − transaction cost − slippage.

This is a **sequential** trade list. If a new signal appears while a trade is open, it is ignored (flat-or-in, no overlapping positions). That is stricter than the event table, which can still have overlapping evaluation windows.
"""
)

code(
    r"""
def backtest(prices: pd.DataFrame, threshold: float, cooldown: int, holding: int,
             cost: float = TRANSACTION_COST, slip: float = SLIPPAGE) -> pd.DataFrame:
    ret = prices["Close"].pct_change()
    raw_pos = np.flatnonzero((ret <= threshold).to_numpy())
    raw_pos = raw_pos[(raw_pos + holding) < len(prices)]
    kept = decluster_events(raw_pos, cooldown)

    trades = []
    occupied_until = -1
    equity = 1.0
    peak = 1.0
    for t in kept:
        entry_i = t + 1
        exit_i = t + holding
        if entry_i <= occupied_until:
            continue  # already in a position
        entry_px = prices.loc[entry_i, "Open"]
        exit_px = prices.loc[exit_i, "Close"]
        gross = exit_px / entry_px - 1
        net = gross - cost - slip
        equity *= 1 + net
        peak = max(peak, equity)
        dd = equity / peak - 1
        trades.append(
            {
                "event_date": prices.loc[t, "Date"],
                "entry_date": prices.loc[entry_i, "Date"],
                "exit_date": prices.loc[exit_i, "Date"],
                "entry": entry_px,
                "exit": exit_px,
                "gross_return": gross,
                "cost": cost,
                "slippage": slip,
                "net_return": net,
                "equity": equity,
                "drawdown": dd,
            }
        )
        occupied_until = exit_i
    return pd.DataFrame(trades)


trades = backtest(df, EVENT_THRESHOLD, COOLDOWN_DAYS, HOLDING_DAYS)
trades
"""
)

code(
    r"""
if len(trades) == 0:
    print("No trades.")
else:
    equity_end = trades["equity"].iloc[-1]
    mdd = trades["drawdown"].min()
    print("Number of trades:", len(trades))
    print("Win rate (net):", (trades["net_return"] > 0).mean())
    print("Mean net return:", trades["net_return"].mean())
    print("Median net return:", trades["net_return"].median())
    print("Cumulative growth factor:", equity_end)
    print("Cumulative return:", equity_end - 1)
    print("Max drawdown (trade-equity mark):", mdd)

    fig, ax = plt.subplots()
    ax.plot(trades["exit_date"], trades["equity"], marker="o")
    ax.set_title("Event strategy equity (starts at 1.0, compounds trade net returns)")
    ax.set_ylabel("Equity")
    plt.tight_layout()
    plt.show()

    # Buy-and-hold over the same full sample, for context only (different risk, always invested).
    bh = df["Close"].iloc[-1] / df["Close"].iloc[0] - 1
    print("Buy-and-hold close-to-close over the file:", bh)
"""
)

md(
    """
## 17. Conclusion

Run the cells above and read the printed numbers against the pre-committed failure conditions:

- Are 1D / 3D / 5D event means positive?
- Is the event mean materially above the ordinary-day baseline?
- Does the frozen −2% / 5-day / 3-day-cooldown rule still work after 1 Apr 2026?
- After 10 bps of round-trip friction, is anything economically left?

With only about one year of NIFTY history in this folder, a non-result is still a valid research outcome. The notebook is designed to report that honestly rather than retune the threshold until the backtest looks good.

The two-page summary lives in `RESEARCH_NOTE.md`.

On this specific one-year file the pre-committed tests do **not** support the hypothesis: n = 4 de-clustered events, 5-day mean +0.48% with a 95% CI that includes zero (one-sided p ≈ 0.30), 3 of 4 events sit in the March 2026 selloff, and the development-window 5-day mean is not a clean positive result. The backtest's small cumulative gain is three trades, not a validated edge.
"""
)

code(
    r"""
print("=== FROZEN PRIMARY SPEC ===")
print("threshold", EVENT_THRESHOLD, "| holding", HOLDING_DAYS, "| cooldown", COOLDOWN_DAYS)
print("file", DATA_PATH.name)
print("obs", len(df), df["Date"].min().date(), "→", df["Date"].max().date())
print("events", len(events), "trades", len(trades) if isinstance(trades, pd.DataFrame) else 0)
if len(events):
    print(event_stats)
    print(comparison)
    print(evidence)
    print(oos_cmp)
"""
)

out = Path(r"C:\Users\saksham\Documents\suas_quant\nifty_event_study.ipynb")
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, "cells", len(nb["cells"]))

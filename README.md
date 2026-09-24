# NIFTY 50 Event Study

This repository contains a small event-study research project on the NIFTY 50 index. The objective is to test whether a large daily fall is followed by a positive recovery over the next few trading days, while keeping the method disciplined and transparent.

## Research question

The research tests the following idea:

> After a NIFTY 50 close-to-close fall of at least 2%, are subsequent forward returns positive on average?

The notebook structure follows a strict workflow:

1. Define a pre-committed rule.
2. Clean and validate the data.
3. Detect events.
4. De-cluster overlapping signals.
5. Measure forward returns.
6. Compare with a baseline of non-event days.
7. Check robustness and out-of-sample performance.
8. Report the result honestly.

## Data and scope

- Source file: `NIFTY 50-23-09-2025-to-23-09-2026.csv`
- Period covered: approximately 2025-09-23 to 2026-09-23
- Data type: daily NSE OHLC data
- Main rule: event day is a daily close-to-close return of at most -2%
- Entry: next trading day open
- Holding period: 5 trading days
- Cooldown: 3 trading days

## Methodology

The event definition is intentionally simple and conservative:

- A crash signal is based on daily close-to-close return, not intraday movement.
- Entry is the next open, not the same-day close, to avoid look-ahead bias.
- Overlapping events are de-clustered using a cooldown rule.
- Event-window returns are compared with ordinary non-event days.
- The analysis checks robustness across thresholds and holding periods.
- The result is evaluated out of sample after a fixed rule is selected.

The actual notebook generator is in `build_notebook.py`, and it produces the final notebook file `nifty_event_study.ipynb`.

## Main findings

The project does not claim a strong trading edge. The evidence is small-sample and fragile.

Key observations from the generated analysis:

- The dataset required cleaning: one corrupted close value and one malformed date were identified and repaired.
- After cleaning, the sample is about one year of daily data, which is limited for a strategy based on rare crash days.
- Only a small number of qualifying events remain after de-clustering.
- The 5-day event return is positive on average, but the sample is too small for strong statistical confidence.
- The result is not strong enough to justify treating it as a validated strategy without more data.
- The frozen rule does not show a consistently decisive edge in the out-of-sample period.

This is a valid outcome in research: a non-result is still a result if the data and process are disciplined.

## Assumptions and limitations

- This is a single-index daily study, not a live-trading framework.
- NIFTY index data is a proxy for an investable product; real execution would include futures or ETF frictions.
- Transaction costs and slippage are included only as a simple model.
- The dataset covers only one year, making the event sample sparse.
- The research is designed to test the hypothesis, not to optimize for attractive backtest statistics.

## Setup and reproduction

Requirements are listed in `requirements.txt`.

To regenerate the notebook:

```bash
py -3.12 -m pip install -r requirements.txt
py -3.12 build_notebook.py
```

This writes the notebook to `nifty_event_study.ipynb`.

## Repository structure

- `build_notebook.py` — notebook generator
- `nifty_event_study.ipynb` — generated research notebook
- `NIFTY 50-23-09-2025-to-23-09-2026.csv` — input data
- `requirements.txt` — dependency list
- `RESEARCH_NOTE.md` — short summary of findings
- `AI_USAGE.md` — AI usage note

## AI Usage Note

This project used AI assistance to support development, but the decisions and final interpretation were made by the researcher.

The AI was used to:

- help structure the event-study workflow,
- sanity-check the notebook logic,
- help write documentation,
- diagnose the environment issue caused by NumPy compatibility,
- improve clarity and formatting of the written summary.

The original notebook logic was not accepted blindly. The project intentionally rejected any suggestion that would optimize the backtest by tuning parameters after seeing the result.

A wrong AI suggestion was the initial assumption that the notebook bug was a logic error in the event code itself. The real problem was environmental: the installed scientific stack was mixed across NumPy versions. That was fixed by aligning the environment to a compatible NumPy 1.x version.

The key lesson is that AI can accelerate analysis and writing, but it cannot replace statistical judgment, data validation, and critical review.

---

## Final interpretation

The method was designed to test a disciplined hypothesis rather than produce a marketable strategy. The evidence suggests a small positive effect in the limited sample, but not enough to call it a robust or generally valid trading edge. The notebook and documentation therefore present the result honestly and do not overstate the conclusion.

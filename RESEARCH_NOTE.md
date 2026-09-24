# NIFTY 50 Event Study Research Note

## Objective

This project tests whether large daily declines in the NIFTY 50 are followed by positive forward returns over the next few trading days. The research is intentionally conservative: the rule is defined before looking for evidence, and the study checks robustness and out-of-sample performance.

## Data and rule

The dataset is the daily NIFTY 50 OHLC file in this repository. The study uses a daily close-to-close return threshold of -2% as the event trigger. Entry occurs at the next trading day open, and the holding period is 5 trading days. A 3-day cooldown is used to avoid counting overlapping crash events as independent signals.

## Data quality and cleaning

The raw file contains a small number of anomalies common in exported market data. One row contained a corrupted close value, and one date was malformed. Those were not silently ignored. The malformed date was inferred from nearby dates, and the corrupted value was dropped because it failed basic OHLC consistency checks. The remaining data were sorted by date and checked for duplicates and gaps.

## Main result

The sample is limited and therefore the findings are weakly powered. After de-clustering, only a small number of qualifying events remain. The event-window mean appears positive, but the statistical evidence is not strong enough to support a confident claim that the effect is real. The result is especially fragile in a one-year sample with only a handful of extreme events.

## Baseline comparison

The study compares event returns against ordinary non-event days using the same entry and holding logic. This matters because a positive event return is not meaningful unless it meaningfully exceeds the typical return available on non-event days. In this sample, the event effect is not persistent enough to clearly dominate the baseline.

## Robustness and out-of-sample tests

The rule was checked across nearby threshold and holding-period choices. The signal is not stable enough to claim a broad, robust effect. The out-of-sample view also does not provide a compelling confirmation of the development-window result. This suggests the observed positive mean may be a sample-specific pattern rather than a stable predictive relationship.

## Interpretation

The project does not find a reliable and economically meaningful NIFTY rebound effect under the pre-committed rule. That outcome is important: it avoids the common research mistake of tuning thresholds until a favorable backtest appears. The notebook therefore treats the result as a non-result unless the evidence improves with more data and independent validation.

## Conclusion

The event-study methodology is disciplined, transparent, and reproducible. The analysis does not support a strong claim that a -2% NIFTY fall predicts a reliably positive rebound over the next 5 trading days. With only one year of data, the study should be viewed as exploratory and inconclusive rather than actionable.

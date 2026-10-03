# BTC · ETH daily signals v2

Upbit KRW-BTC and KRW-ETH, confirmed daily candles only; daily close at 09:00 KST. Five-minute cached data with manual refresh.

## Indicators
- Wilder RSI(14) and DMI(14), SMA seeded; RSI signal line SMA(9), ADX for context.
- Slow Stochastic(30,10,10): raw %K = 100*(close-lowest low over 30)/(highest high over 30-lowest low over 30). Slow %K is SMA(10) of raw %K; Slow %D is SMA(10) of Slow %K. Zero price range uses neutral 50. Warm-up is preserved.

## Trade rules
- Default divergence window: 14 completed daily candles, adjustable 5–60. Linear regression slopes on closing price and RSI must point in opposite directions. Fitted price change >=1% and fitted RSI change >=3 points (adjustable). Window endpoint changes must agree with the regression directions.
- Rising close trend + falling RSI trend: sell divergence; falling close trend + rising RSI trend: buy divergence. Signal fires on onset of the condition, not repeatedly while the condition persists. Current divergence state remains visible. No pivot confirmation delay or future data.
- Buy also on DMI up-cross when Slow %K>%D, OR on Slow %K/%D up-cross when +DI>-DI.
- Sell also on the opposite confirmed cross conditions. Simultaneous buy/sell conditions produce conflict/wait.
- Standalone RSI 30/70 crosses are no longer trade triggers. RSI and stochastic reference levels are displayed.

Price, RSI, Slow Stochastic and DMI panels; divergent intervals are connected on price/RSI charts, signals marked with blue/purple arrows. History contains up to 100 events with exact reasons and indicator values. No automatic orders or background push notifications.

## Backtest
Cash-only start over the selected chart window, historical indicator warm-up, next available open after confirmed signal, long-only spot, minimum 24h holding. Per side: fee 0.05%, slippage 0.03%. Pending final signals stay unfilled. Equity includes open-position value net of liquidation costs. Historical simulation is not a forecast or optimized strategy.

```bash
pip install -r requirements.txt
streamlit run app.py
python -m unittest test_strategy -q
```

## v3 display and signal sequencing
All emitted buy/sell signals alternate across the entire loaded history, independent of trigger type. The first directional event can be either buy or sell; after buy, further buys are suppressed until sell, and vice versa. Conflicts do not change the last directional state. Raw candidates remain internal. Changing chart period does not reset the signal sequence.

Main display: one large daily candlestick chart with buy/sell arrows only, default three-month view with longer periods selectable. RSI, stochastic and DMI remain in the calculations but their graphs and numeric values are hidden. History only shows alternating directional events. Signal state refers to this app's event history, not actual account holdings.

## v5 frozen evaluated parameters (2026-10-03)
Default active_signals(indicator_frame, market) applies BTC: divergence window30, existing thresholds1%/3 RSI points, immediate divergence, no ADX buy filter, early exit on DMI down-cross. ETH retains the previous window14 baseline because the proposed window21/Slow %K direction/ADX15 candidate worsened the final evaluation. Stochastic remains30/10/10 for both.

32 candidates per market ranked on first50% training (after60-row warm-up) and next25% validation. Last25% was excluded from ranking, but was used as a deployment gate: selected return and drawdown must both be no worse than the preexisting baseline. Thus final evaluation is not wholly untouched by the deployment decision. Fee0.05%/side, slippage0.03%/side, next-open fills, >=24h holding, long-only.

Final evaluation: 2026-01-26 through2026-10-02. BTC baseline+0.6368%, selected/applied+30.0189% (5 completed trades); baseline MDD-16.0627%, selected MDD-8.3138%; buy-and-hold-10.6101%. ETH baseline/applied-0.5176%; candidate-7.5621% and worse drawdown, rejected; buy-and-hold-13.2666%. Full historical charts using selected conditions are retrospective and may include tuning data. Frozen conditions are not automatically retrained. A manual mode remains available. Results are historical simulation, not a forecast or guaranteed maximum.

Run python optimize_offline.py to reproduce from archived daily CSV snapshots. If a snapshot is absent, the script downloads daily data. optimization_results.json records candidate settings and selection/evaluation dates. The four-hour alert task uses active_signals with the same default market-specific parameters.

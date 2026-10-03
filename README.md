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

## v6 one additional indicator: EMA (2026-10-03)
Only EMA is added as a new indicator. Fixed v5 BTC/ETH base parameters are compared with 16 EMA variants: period20/50/100/200, price-above or rising-slope entry gate, optional downward price-cross exit. EMA uses close.ewm(span=period,adjust=False,min_periods=period).mean(); no future observations. Existing entries must pass the EMA gate; EMA does not generate an independent entry. Existing exits remain, with optional extra exit. Alternation is applied after filtering.

Rank using next25% validation return plus0.1*drawdown and0.01*training return, minimum2 completed trades in each fold, training warmup200. Dates and archived snapshots match v5. Recent comparison 2026-01-26..2026-10-02 also gates adoption alongside validation return. These snapshots and the recent comparison have already been examined in v5, so this is reused retrospective research, NOT a new untouched out-of-sample test.

ETH applies EMA50 price-above filter and downward-cross exit: previous return-0.5176%/MDD-16.5655%; new+8.3300%/MDD-5.4372%, only2 completed trades. BTC's ranked EMA200 candidate return+4.1036% versus existing+30.0189%, so BTC keeps v5. Manual mode offers EMA50. Chart stays daily price and arrows only, stochastic stays30/10/10, four-hour monitoring and notifications use active_signals. Results include fee0.05% and slip0.03%/side with ideal next-open execution; real monitoring/execution latency may differ. Future maximum profit is not established or guaranteed.

Run python optimize_ema.py from the archived daily CSV snapshots. ema_results.json preserves all16 candidates per coin and decisions. Ten strategy/UI tests pass via python -m unittest test_strategy -q.

## v7 daily-close 10% protective stop
active_signals and manual UI apply stop_pct=10.0 for both markets. Price at the next candle open after a buy signal, plus0.03% simulated slippage, anchors the stop. If a completed daily close is <=90% of this simulated entry, emit sell with explicit stop reason, overriding an opposing/conflicting candidate. Intraday lows alone do not trigger. Existing exits are retained. Stop exits reset the alternation state so the next valid buy is allowed; repeated sells remain suppressed. Stop sale executes at next open in backtests, so realized loss can exceed10%. No actual accounts or automatic orders are connected; the anchor is not an actual purchase price.

sequence_signals handles entry anchors, stop_price and stop_trigger. Generic signals keeps stop_pct=0 by default for legacy research reproduction; production must use active_signals, which explicitly enables10%. Current archived BTC/ETH histories trigger zero stops; recent simulated returns remain BTC+30.0189%, ETH+8.3300%. Eleven tests pass, including equality boundary, stop priority, next-open gap, close-only trigger, no-lookahead and re-entry.

## v8 compact responsive display
Main page shows a compact title, coin/period/refresh row, three summary cards and a410px daily chart. Detailed stop rules and monitoring explanations are folded into expanders, including the current simulated stop threshold when holding. Light gray background, white cards, dark navy text, larger main labels, red rising/blue falling candles and blue buy/purple sell arrows. Price axis uses KRW units of 억 or 만 to avoid long numbers. Narrow screens keep control and summary rows side by side with smaller typography. Indicators stay hidden. All11 strategy/UI checks pass; no strategy or alert changes from v7.

## v9 strength display
signal_label displays 매수, 강력매수, 매도, 강력매도, 손절 매도 or wait. Underlying sig remains the alternating base direction; backtest decisions and profit are unchanged. Strong labels require all three same-direction families: RSI divergence-state OR RSI above/below its9-day signal with corresponding RSI daily change; DMI direction plus ADX>=20; SlowK above/below SlowD with corresponding SlowK daily change. Opposite RSI divergence blocks the strong label. Protective10% stop label takes priority. ADX is part of the already computed DMI family, not a new trade filter. No independent new buy/sell is generated on a strength upgrade.

Chart annotations, metrics, history CSV and four-hour alerts display signal_label; event detection/deduplication continue to use sig. Strong means indicator agreement, not established higher accuracy or a profit guarantee. Twelve tests pass. Current colors: distinct light-blue background, navy text/buttons, coral rising candles, blue falling candles, teal buy markers, violet sell markers.

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

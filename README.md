# BTC · ETH 일봉 매매 신호

Streamlit app for Upbit KRW-BTC and KRW-ETH, using completed daily candles only. Upbit daily candles close at 09:00 KST. Data is cached for five minutes; refresh clears the cache.

## Rules
- Wilder DMI and RSI, default period 14, SMA initialization.
- Buy: +DI crosses above -DI with RSI below 70; RSI crosses above 30; or bullish close/RSI divergence.
- Sell: +DI crosses below -DI with RSI above 30; RSI crosses below 70; or bearish close/RSI divergence.
- Conditions are alternatives (OR). Opposite conditions on one candle yield conflict/wait.
- Divergence compares consecutive strict local closing-price lows/highs and RSI at those points, no more than 60 days apart. Each pivot needs two candles on both sides by default. Signal is emitted on the confirmation candle, never backdated to the pivot. Daily close-to-close opposite direction is not possible for standard RSI; swing comparison is used.
- ADX is displayed for context, without filtering signals.

## Display and backtest
Price, RSI and DMI panels, separated arrow colors, divergence lines, latest confirmed signal, and up to 100 recent events with reasons. Event history is independent of portfolio holdings; there is no automatic order placement or background push notification.

Backtest starts in cash over the selected chart period. Indicators use earlier history. Orders execute at the next available candle open after the signal; spot long-only, minimum 24-hour holding, fee 0.05% and slippage 0.03% per side. Final pending signals remain unfilled. Equity includes unrealized holdings valued net of liquidation costs. No parameter optimization or claimed out-of-sample validation.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
python -m unittest test_strategy -q
```

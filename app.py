import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

VERSION = '일봉 매매 신호 · 검증 최적화 · EMA 추가 검증 · 일봉 종가 10% 손절 · 4시간 모니터링 / v11'


OPTIMIZATION_REPORT = {'KRW-BTC': {'params': {'window': 30, 'min_price': 1.0, 'min_rsi': 3.0, 'div_confirm': 'none', 'min_adx': 0, 'exit_mode': 'dmi_early'}, 'train': {'return': 59.504, 'dd': -9.5508, 'trades': 12}, 'validation': {'return': -5.4235, 'dd': -13.1394, 'trades': 6}, 'holdout': {'return': 30.0189, 'dd': -8.3138, 'trades': 5}, 'baseline_holdout': {'return': 0.6368, 'dd': -16.0627, 'trades': 7}, 'full': {'return': 91.7306, 'dd': -13.4732, 'trades': 24}, 'baseline_full': {'return': -3.957, 'dd': -25.7951, 'trades': 27}, 'candidates': 32, 'train_start': '2024-03-08 09:00:00+09:00', 'train_end': '2025-05-20 09:00:00+09:00', 'validation_start': '2025-05-21 09:00:00+09:00', 'validation_end': '2026-01-25 09:00:00+09:00', 'holdout_start': '2026-01-26 09:00:00+09:00', 'holdout_end': '2026-10-02 09:00:00+09:00', 'fit_end': '2026-01-25 09:00:00+09:00', 'baseline_validation': {'return': np.float64(-6.1246), 'dd': np.float64(-14.1909)}, 'buyhold_holdout': np.float64(-10.6101), 'applied_params': {'window': 30, 'min_price': 1.0, 'min_rsi': 3.0, 'div_confirm': 'none', 'min_adx': 0, 'exit_mode': 'dmi_early'}, 'applied_holdout': {'return': 30.0189, 'dd': -8.3138, 'trades': 5}, 'decision': 'BTC: 비교 후 적용'}, 'KRW-ETH': {'params': {'window': 21, 'min_price': 1.0, 'min_rsi': 3.0, 'div_confirm': 'stoch_direction', 'min_adx': 15, 'exit_mode': 'confirmed'}, 'train': {'return': 80.6914, 'dd': -27.0236, 'trades': 8}, 'validation': {'return': 21.3897, 'dd': -24.613, 'trades': 6}, 'holdout': {'return': -7.5621, 'dd': -24.1031, 'trades': 5}, 'baseline_holdout': {'return': -0.5176, 'dd': -16.5655, 'trades': 6}, 'full': {'return': 104.1785, 'dd': -36.407, 'trades': 20}, 'baseline_full': {'return': 18.7774, 'dd': -42.651, 'trades': 24}, 'candidates': 32, 'train_start': '2024-03-08 09:00:00+09:00', 'train_end': '2025-05-20 09:00:00+09:00', 'validation_start': '2025-05-21 09:00:00+09:00', 'validation_end': '2026-01-25 09:00:00+09:00', 'holdout_start': '2026-01-26 09:00:00+09:00', 'holdout_end': '2026-10-02 09:00:00+09:00', 'fit_end': '2026-01-25 09:00:00+09:00', 'baseline_validation': {'return': np.float64(6.6622), 'dd': np.float64(-29.9458)}, 'buyhold_holdout': np.float64(-13.2666), 'applied_params': {'window': 14, 'min_price': 1.0, 'min_rsi': 3.0, 'div_confirm': 'none', 'min_adx': 0, 'exit_mode': 'confirmed'}, 'applied_holdout': {'return': -0.5176, 'dd': -16.5655, 'trades': 6}, 'decision': 'ETH: 별도 평가에서 악화되어 기존 조건 유지'}}

EMA_REPORT = {'KRW-BTC': {'params': {'window': 30, 'min_price': 1.0, 'min_rsi': 3.0, 'div_confirm': 'none', 'min_adx': 0, 'exit_mode': 'dmi_early', 'ema_period': 200, 'ema_mode': 'price', 'ema_exit': True}, 'train': {'ret': 55.3473, 'dd': -12.5999, 'trades': 7}, 'validation': {'ret': 5.3165, 'dd': -6.0457, 'trades': 4}, 'score': 5.265403, 'recent': {'ret': 4.1036, 'dd': -1.2577, 'trades': 1}, 'previous': {'ret': 30.0189, 'dd': -8.3138, 'trades': 5}, 'previous_validation': {'ret': -5.4235, 'dd': -13.1394, 'trades': 6}, 'previous_params': {'window': 30, 'min_price': 1.0, 'min_rsi': 3.0, 'div_confirm': 'none', 'min_adx': 0, 'exit_mode': 'dmi_early'}, 'applied_params': {'window': 30, 'min_price': 1.0, 'min_rsi': 3.0, 'div_confirm': 'none', 'min_adx': 0, 'exit_mode': 'dmi_early'}, 'applied': {'ret': 30.0189, 'dd': -8.3138, 'trades': 5}, 'decision': '개선 검증 미충족 · 기존 유지', 'candidates': 16, 'comparison_start': '2026-01-26 09:00:00+09:00', 'comparison_end': '2026-10-02 09:00:00+09:00', 'selection_end': '2026-01-25 09:00:00+09:00'}, 'KRW-ETH': {'params': {'window': 14, 'min_price': 1.0, 'min_rsi': 3.0, 'div_confirm': 'none', 'min_adx': 0, 'exit_mode': 'confirmed', 'ema_period': 50, 'ema_mode': 'price', 'ema_exit': True}, 'train': {'ret': 22.3504, 'dd': -8.381, 'trades': 2}, 'validation': {'ret': 9.5783, 'dd': -28.0445, 'trades': 8}, 'score': 6.9973540000000005, 'recent': {'ret': 8.33, 'dd': -5.4372, 'trades': 2}, 'previous': {'ret': -0.5176, 'dd': -16.5655, 'trades': 6}, 'previous_validation': {'ret': 6.6622, 'dd': -29.9458, 'trades': 7}, 'previous_params': {'window': 14, 'min_price': 1.0, 'min_rsi': 3.0, 'div_confirm': 'none', 'min_adx': 0, 'exit_mode': 'confirmed'}, 'applied_params': {'window': 14, 'min_price': 1.0, 'min_rsi': 3.0, 'div_confirm': 'none', 'min_adx': 0, 'exit_mode': 'confirmed', 'ema_period': 50, 'ema_mode': 'price', 'ema_exit': True}, 'applied': {'ret': 8.33, 'dd': -5.4372, 'trades': 2}, 'decision': 'EMA 추가 적용', 'candidates': 16, 'comparison_start': '2026-01-26 09:00:00+09:00', 'comparison_end': '2026-10-02 09:00:00+09:00', 'selection_end': '2026-01-25 09:00:00+09:00'}}
for _market, _result in EMA_REPORT.items():
    OPTIMIZATION_REPORT[_market]['applied_params'] = _result['applied_params']
    OPTIMIZATION_REPORT[_market]['applied_holdout'] = {'return': _result['applied']['ret'], 'dd': _result['applied']['dd'], 'trades': _result['applied']['trades']}

EXTENDED_REPORT = {'KRW-BTC': {'candidate': {'profile': {'volume': 1.0, 'macd': 'positive'}, 'train': {'return': 77.6946, 'dd': -7.8245, 'trades': 7}, 'validation': {'return': -0.6561, 'dd': -8.4932, 'trades': 4}, 'recent': {'return': 3.3522, 'dd': -10.9594, 'trades': 4}, 'year': {'return': -0.0169, 'dd': -13.1452, 'trades': 5}, 'score': 25.527771749999996}, 'baseline': {'profile': {}, 'train': {'return': 59.504, 'dd': -9.5508, 'trades': 12}, 'validation': {'return': -5.4235, 'dd': -13.1394, 'trades': 6}, 'recent': {'return': 30.0189, 'dd': -8.3138, 'trades': 5}, 'year': {'return': 16.7598, 'dd': -10.5429, 'trades': 7}, 'score': 15.518616499999998}, 'applied': {'profile': {}, 'train': {'return': 59.504, 'dd': -9.5508, 'trades': 12}, 'validation': {'return': -5.4235, 'dd': -13.1394, 'trades': 6}, 'recent': {'return': 30.0189, 'dd': -8.3138, 'trades': 5}, 'year': {'return': 16.7598, 'dd': -10.5429, 'trades': 7}, 'score': 15.518616499999998}, 'decision': '개선 기준 미충족 또는 기존 우수 · 기존 유지', 'count': 84, 'train_start': '2024-03-08 09:00:00+09:00', 'train_end': '2025-05-20 09:00:00+09:00', 'validation_start': '2025-05-21 09:00:00+09:00', 'validation_end': '2026-01-25 09:00:00+09:00', 'recent_start': '2026-01-26 09:00:00+09:00', 'recent_end': '2026-10-02 09:00:00+09:00', 'year_start': '2025-10-03 09:00:00+09:00', 'year_end': '2026-10-02 09:00:00+09:00', 'quarters': [{'start': '2025-10-03 09:00:00+09:00', 'end': '2025-12-31 09:00:00+09:00', 'baseline': {'return': -7.1718, 'dd': -8.6809, 'trades': 1}, 'applied': {'return': -7.1718, 'dd': -8.6809, 'trades': 1}}, {'start': '2026-01-01 09:00:00+09:00', 'end': '2026-03-31 09:00:00+09:00', 'baseline': {'return': 2.5151, 'dd': -7.6095, 'trades': 2}, 'applied': {'return': 2.5151, 'dd': -7.6095, 'trades': 2}}, {'start': '2026-04-01 09:00:00+09:00', 'end': '2026-06-29 09:00:00+09:00', 'baseline': {'return': 8.9628, 'dd': -4.2482, 'trades': 1}, 'applied': {'return': 8.9628, 'dd': -4.2482, 'trades': 1}}, {'start': '2026-06-30 09:00:00+09:00', 'end': '2026-10-02 09:00:00+09:00', 'baseline': {'return': 12.2721, 'dd': -7.2205, 'trades': 2}, 'applied': {'return': 12.2721, 'dd': -7.2205, 'trades': 2}}], 'baseline_double_cost': {'return': 15.4594, 'dd': -10.9712, 'trades': 7}, 'applied_double_cost': {'return': 15.4594, 'dd': -10.9712, 'trades': 7}}, 'KRW-ETH': {'candidate': {'profile': {'macd': 'positive', 'trail': 5.0}, 'train': {'return': 22.1743, 'dd': -8.3386, 'trades': 6}, 'validation': {'return': 31.9211, 'dd': -20.5991, 'trades': 6}, 'recent': {'return': 5.333, 'dd': -5.4372, 'trades': 2}, 'year': {'return': 1.0263, 'dd': -6.7355, 'trades': 4}, 'score': 26.06353125}, 'baseline': {'profile': {}, 'train': {'return': 22.2409, 'dd': -8.8651, 'trades': 5}, 'validation': {'return': 9.5783, 'dd': -28.0445, 'trades': 8}, 'recent': {'return': 8.33, 'dd': -5.4372, 'trades': 2}, 'year': {'return': 1.8292, 'dd': -8.5951, 'trades': 5}, 'score': 10.810453500000001}, 'applied': {'profile': {'macd': 'positive'}, 'train': {'return': 22.2409, 'dd': -8.8651, 'trades': 5}, 'validation': {'return': 21.983, 'dd': -26.5807, 'trades': 6}, 'recent': {'return': 8.33, 'dd': -5.4372, 'trades': 2}, 'year': {'return': 3.9008, 'dd': -6.7355, 'trades': 4}, 'score': 19.016229}, 'decision': '추가 조건 적용', 'count': 84, 'train_start': '2024-03-08 09:00:00+09:00', 'train_end': '2025-05-20 09:00:00+09:00', 'validation_start': '2025-05-21 09:00:00+09:00', 'validation_end': '2026-01-25 09:00:00+09:00', 'recent_start': '2026-01-26 09:00:00+09:00', 'recent_end': '2026-10-02 09:00:00+09:00', 'year_start': '2025-10-03 09:00:00+09:00', 'year_end': '2026-10-02 09:00:00+09:00', 'quarters': [{'start': '2025-10-03 09:00:00+09:00', 'end': '2025-12-31 09:00:00+09:00', 'baseline': {'return': -4.6468, 'dd': -4.6468, 'trades': 2}, 'applied': {'return': -2.7069, 'dd': -2.7069, 'trades': 1}}, {'start': '2026-01-01 09:00:00+09:00', 'end': '2026-03-31 09:00:00+09:00', 'baseline': {'return': -1.4202, 'dd': -3.8165, 'trades': 1}, 'applied': {'return': -1.4202, 'dd': -3.8165, 'trades': 1}}, {'start': '2026-04-01 09:00:00+09:00', 'end': '2026-06-29 09:00:00+09:00', 'baseline': {'return': 3.6951, 'dd': -5.4372, 'trades': 1}, 'applied': {'return': 3.6951, 'dd': -5.4372, 'trades': 1}}, {'start': '2026-06-30 09:00:00+09:00', 'end': '2026-10-02 09:00:00+09:00', 'baseline': {'return': 4.4698, 'dd': -3.9897, 'trades': 1}, 'applied': {'return': 4.4698, 'dd': -3.9897, 'trades': 1}}], 'baseline_double_cost': {'return': 0.8563, 'dd': -9.1782, 'trades': 5}, 'applied_double_cost': {'return': 3.0729, 'dd': -7.1821, 'trades': 4}}}

def active_signals(x, market):
    profile = EXTENDED_REPORT[market]["applied"]["profile"]
    if profile:
        return research_signals(x, market, profile)
    return signals(x, stop_pct=10.0, **OPTIMIZATION_REPORT[market]["applied_params"])


def wilder(series, period):
    """SMA seed followed by Wilder recursive smoothing, retaining warm-up NaNs."""
    out = pd.Series(np.nan, index=series.index, dtype=float)
    seed = series.rolling(period, min_periods=period).mean().first_valid_index()
    if seed is None:
        return out
    start = series.index.get_loc(seed)
    out.iloc[start] = series.iloc[start-period+1:start+1].mean()
    for i in range(start+1, len(series)):
        out.iloc[i] = (out.iloc[i-1]*(period-1)+series.iloc[i])/period
    return out


def indicators(raw, period=14, stoch_period=30, smooth_k=10, smooth_d=10):
    x = raw.reset_index(drop=True).copy()
    delta = x.close.diff()
    gain, loss = wilder(delta.clip(lower=0), period), wilder((-delta).clip(lower=0), period)
    x['RSI'] = 100-100/(1+gain/loss.replace(0, np.nan))
    x.loc[(loss == 0) & (gain > 0), 'RSI'] = 100
    x.loc[(gain == 0) & (loss > 0), 'RSI'] = 0
    x.loc[(gain == 0) & (loss == 0), 'RSI'] = 50
    up, down = x.high.diff(), -x.low.diff()
    plus = up.where((up > down) & (up > 0), 0.0)
    minus = down.where((down > up) & (down > 0), 0.0)
    plus.iloc[0] = minus.iloc[0] = np.nan
    tr = pd.concat([x.high-x.low, (x.high-x.close.shift()).abs(), (x.low-x.close.shift()).abs()], axis=1).max(axis=1)
    tr.iloc[0] = np.nan
    atr = wilder(tr, period)
    x['PDI'] = (100*wilder(plus, period)/atr.replace(0, np.nan)).where(atr != 0, 0)
    x['MDI'] = (100*wilder(minus, period)/atr.replace(0, np.nan)).where(atr != 0, 0)
    total = x.PDI+x.MDI
    dx = (100*(x.PDI-x.MDI).abs()/total.replace(0, np.nan)).where(total != 0, 0)
    x['ADX'] = wilder(dx, period)
    x['ATR'] = atr
    x['MACD'] = x.close.ewm(span=12,adjust=False,min_periods=12).mean()-x.close.ewm(span=26,adjust=False,min_periods=26).mean()
    x['MACDsignal'] = x.MACD.ewm(span=9,adjust=False,min_periods=9).mean()
    x['MACDhist'] = x.MACD-x.MACDsignal
    x['VolumeRatio'] = x.volume/x.volume.shift().rolling(20,min_periods=20).mean().replace(0,np.nan)
    x['BBmid'] = x.close.rolling(20,min_periods=20).mean()
    bbdev=x.close.rolling(20,min_periods=20).std(ddof=0)
    x['BBlower']=x.BBmid-2*bbdev
    x['BBupper']=x.BBmid+2*bbdev
    x['RSIsignal'] = x.RSI.rolling(9, min_periods=9).mean()
    lowest = x.low.rolling(stoch_period, min_periods=stoch_period).min()
    highest = x.high.rolling(stoch_period, min_periods=stoch_period).max()
    spread = highest-lowest
    fast = (100*(x.close-lowest)/spread.replace(0, np.nan)).where(spread != 0, 50)
    x['SlowK'] = fast.rolling(smooth_k, min_periods=smooth_k).mean()
    x['SlowD'] = x.SlowK.rolling(smooth_d, min_periods=smooth_d).mean()
    return x


def signals(x, window=14, min_price=1.0, min_rsi=3.0, div_confirm='none', min_adx=0, exit_mode='confirmed', ema_period=0, ema_mode='price', ema_exit=False, stop_pct=0.0):
    """Trailing regression divergence; no future pivots or retroactive signals."""
    z = x.copy()
    if ema_period:
        z['EMA'] = z.close.ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
        z['EMAready'] = z.EMA.notna() & z.EMA.shift().notna()
        z['EMAentry'] = z.EMAready & ((z.close > z.EMA) if ema_mode == 'price' else (z.EMA > z.EMA.shift()))
        z['EMAdown'] = z.EMAready & (z.close < z.EMA) & (z.close.shift() >= z.EMA.shift())
    axis = np.arange(window, dtype=float)
    centered = axis-axis.mean()
    def slope(values):
        return float(np.dot(centered, values)/np.dot(centered, centered))
    z['price_slope'] = z.close.rolling(window, min_periods=window).apply(slope, raw=True)
    z['rsi_slope'] = z.RSI.rolling(window, min_periods=window).apply(slope, raw=True)
    price_start = z.close.shift(window-1)
    price_fit = z.price_slope*(window-1)/price_start*100
    rsi_fit = z.rsi_slope*(window-1)
    price_change = (z.close/price_start-1)*100
    rsi_change = z.RSI-z.RSI.shift(window-1)
    z['bull_state'] = (price_fit < -min_price) & (rsi_fit > min_rsi) & (price_change < 0) & (rsi_change > 0)
    z['bear_state'] = (price_fit > min_price) & (rsi_fit < -min_rsi) & (price_change > 0) & (rsi_change < 0)
    if div_confirm == 'stoch_direction':
        bull_ready = z.bull_state & (z.SlowK.diff() > 0)
        bear_ready = z.bear_state & (z.SlowK.diff() < 0)
    else:
        bull_ready, bear_ready = z.bull_state, z.bear_state
    z['bull_div'] = bull_ready & ~bull_ready.shift(1, fill_value=False)
    z['bear_div'] = bear_ready & ~bear_ready.shift(1, fill_value=False)
    z['div_from'] = np.where(z.bull_div | z.bear_div, z.index-window+1, -1)
    z['div_to'] = np.where(z.bull_div | z.bear_div, z.index, -1)
    z['DMIup'] = (z.PDI > z.MDI) & (z.PDI.shift() <= z.MDI.shift())
    z['DMIdown'] = (z.PDI < z.MDI) & (z.PDI.shift() >= z.MDI.shift())
    z['StochUp'] = (z.SlowK > z.SlowD) & (z.SlowK.shift() <= z.SlowD.shift())
    z['StochDown'] = (z.SlowK < z.SlowD) & (z.SlowK.shift() >= z.SlowD.shift())
    z['buy_reason'] = ''; z['sell_reason'] = ''
    for i in range(1, len(z)):
        r = z.iloc[i]; buys, sells = [], []
        if r.bull_div:
            buys.append(f'상승 다이버전스 · {window}일 종가 하락 / RSI 상승'+(' · Slow %K 상승 확인' if div_confirm != 'none' else ''))
        if r.bear_div:
            sells.append(f'하락 다이버전스 · {window}일 종가 상승 / RSI 하락'+(' · Slow %K 하락 확인' if div_confirm != 'none' else ''))
        if r.DMIup and r.SlowK > r.SlowD and (min_adx <= 0 or r.ADX >= min_adx):
            buys.append('DMI 상향교차 + Slow %K > %D')
        if r.DMIdown and (r.SlowK < r.SlowD or exit_mode == 'dmi_early'):
            sells.append('DMI 하향교차 · 조기청산' if exit_mode == 'dmi_early' and r.SlowK >= r.SlowD else 'DMI 하향교차 + Slow %K < %D')
        if r.StochUp and r.PDI > r.MDI and (min_adx <= 0 or r.ADX >= min_adx):
            buys.append('Slow %K/%D 상향교차 + DMI 상승 방향')
        if r.StochDown and r.PDI < r.MDI:
            sells.append('Slow %K/%D 하향교차 + DMI 하락 방향')
        if ema_period:
            if not r.EMAentry:
                buys = []
            elif buys:
                buys.append(f'EMA({ema_period}) '+('상회 확인' if ema_mode == 'price' else '상승 확인'))
            if ema_exit and r.EMAdown:
                sells.append(f'EMA({ema_period}) 종가 하향돌파')
        z.loc[i, 'buy_reason'] = ' / '.join(buys)
        z.loc[i, 'sell_reason'] = ' / '.join(sells)
    buy, sell = z.buy_reason.ne(''), z.sell_reason.ne('')
    z['sig'] = np.select([buy & sell, buy, sell], ['관망(충돌)', '매수', '매도'], default='관망')
    z['candidate_sig'] = z.sig.copy()
    z['reason'] = (z.buy_reason+' / '+z.sell_reason).str.strip(' /')
    z = sequence_signals(z, stop_pct=stop_pct)
    z['confirmed_at'] = z.time+pd.Timedelta(days=1)
    z['div_state'] = np.select([z.bull_state,z.bear_state],['상승 다이버전스(매수 방향)','하락 다이버전스(매도 방향)'],default='없음')
    return label_strength(z)


def sequence_signals(z, stop_pct=0.0, slip=0.0003, atr_mult=0.0, trailing_pct=0.0):
    """Alternate raw events; daily-close stop anchored to next-open simulated entry."""
    z = z.reset_index(drop=True).copy()
    last_signal = None; entry_price = None; peak_close = None; trail_line = None
    states = []; emitted = []; anchors = []; stops = []; triggers = []
    for i in range(len(z)):
        row = z.iloc[i]
        # Yesterday's confirmed signal fills at today's open in the simulation.
        if i > 0:
            if emitted[-1] == '매수':
                entry_price = float(row.open)*(1+slip)
                peak_close = entry_price; trail_line = None
            elif emitted[-1] == '매도':
                entry_price = None; peak_close = None; trail_line = None
        stop_line = entry_price*(1-stop_pct/100) if entry_price is not None and stop_pct > 0 else np.nan
        trail_hit = False
        if entry_price is not None and (atr_mult > 0 or trailing_pct > 0):
            peak_close = max(peak_close, float(row.close))
            candidate_line = peak_close*(1-trailing_pct/100) if trailing_pct > 0 else (peak_close-atr_mult*float(row.ATR) if pd.notna(row.ATR) else np.nan)
            if pd.notna(candidate_line):
                trail_line = max(trail_line,candidate_line) if trail_line is not None else candidate_line
            trail_hit = trail_line is not None and float(row.close) <= trail_line
        hit = entry_price is not None and stop_pct > 0 and float(row.close) <= stop_line
        signal = '매도' if hit or trail_hit else row.candidate_sig
        if hit:
            z.loc[i, 'reason'] = f'일봉 종가 {stop_pct:g}% 손절 · 모의 매수가 ₩{entry_price:,.0f} · 손절 기준 ₩{stop_line:,.0f}'
        if trail_hit and not hit:
            z.loc[i, 'reason'] = f'일봉 종가 추적청산 · '+(f'ATR(14) × {atr_mult:g}' if atr_mult else f'고점 대비 {trailing_pct:g}%')+f' · 보호선 ₩{trail_line:,.0f}'
        if signal in ('매수', '매도'):
            if signal == last_signal:
                signal = '관망'
                z.loc[i, 'reason'] = f'동일 {last_signal} 조건 반복 · 새 신호 없음'
            else:
                last_signal = signal
        z.loc[i, 'sig'] = signal
        emitted.append(signal)
        states.append('매수 이후' if last_signal == '매수' else '매도 이후' if last_signal == '매도' else '신호 대기')
        anchors.append(entry_price if entry_price is not None else np.nan)
        stops.append(max(stop_line,trail_line) if pd.notna(stop_line) and trail_line is not None else trail_line if trail_line is not None else stop_line); triggers.append(bool(hit and signal == '매도'))
    z['signal_state'] = states
    z['sim_entry_price'] = anchors; z['stop_price'] = stops; z['stop_trigger'] = triggers
    return z


def research_signals(x, market, profile=None, base_frame=None):
    """Add tested entry confirmation / profit protection to the frozen v10 base."""
    profile = profile or {}
    z = (base_frame if base_frame is not None else signals(x,**OPTIMIZATION_REPORT[market]['applied_params'])).copy()
    # Rebuild from unsuppressed indicator reasons before applying alternation and risk.
    buy=z.buy_reason.ne(''); sell=z.sell_reason.ne('')
    gate=pd.Series(True,index=z.index)
    volume=profile.get('volume',0)
    macd=profile.get('macd','none')
    bb=profile.get('bb','none')
    if volume:gate &= z.VolumeRatio >= volume
    if macd=='positive':gate &= z.MACDhist > 0
    elif macd=='rising':gate &= z.MACDhist.diff() > 0
    if bb=='mid':gate &= z.close > z.BBmid
    elif bb=='rebound':gate &= (z.close > z.BBlower) & (z.close.diff() > 0)
    buy &= gate
    z['candidate_sig']=np.select([buy & sell,buy,sell],['관망(충돌)','매수','매도'],default='관망')
    z['sig']=z.candidate_sig.copy()
    z['reason']=np.where(buy & sell,z.buy_reason+' / '+z.sell_reason,np.where(buy,z.buy_reason,np.where(sell,z.sell_reason,'')))
    checks=[]
    if volume:checks.append(f'거래량 ≥ 직전20일 평균 × {volume:g}')
    if macd!='none':checks.append('MACD 히스토그램 '+('양수' if macd=='positive' else '상승'))
    if bb!='none':checks.append('볼린저 '+('중심선 상회' if bb=='mid' else '하단 위 종가 상승'))
    if checks:z.loc[buy,'reason']=z.loc[buy,'reason']+' · '+' / '.join(checks)
    z=sequence_signals(z,stop_pct=10,atr_mult=profile.get('atr',0),trailing_pct=profile.get('trail',0))
    return label_strength(z)


def label_strength(z):
    """Display strength only: preserve base events, trades and alternation."""
    z = z.copy()
    missing = pd.Series(np.nan, index=z.index)
    neutral = pd.Series(False, index=z.index)
    rsi_signal = z.get('RSIsignal', missing)
    bull = z.get('bull_state', neutral); bear = z.get('bear_state', neutral)
    rsi_up = bull | ((z.RSI > rsi_signal) & (z.RSI.diff() > 0))
    rsi_down = bear | ((z.RSI < rsi_signal) & (z.RSI.diff() < 0))
    adx = z.get('ADX', missing)
    dmi_up = (z.PDI > z.MDI) & (adx >= 20)
    dmi_down = (z.PDI < z.MDI) & (adx >= 20)
    stoch_up = (z.SlowK > z.SlowD) & (z.SlowK.diff() > 0)
    stoch_down = (z.SlowK < z.SlowD) & (z.SlowK.diff() < 0)
    buy = z.sig.eq('매수'); sell = z.sig.eq('매도')
    up_count = rsi_up.astype(int)+dmi_up.astype(int)+stoch_up.astype(int)
    down_count = rsi_down.astype(int)+dmi_down.astype(int)+stoch_down.astype(int)
    z['strength_count'] = np.where(buy, up_count, np.where(sell, down_count, 0))
    z['signal_label'] = z.sig.copy()
    z.loc[buy & (up_count == 3) & ~bear, 'signal_label'] = '강력매수'
    z.loc[sell & (down_count == 3) & ~bull, 'signal_label'] = '강력매도'
    z.loc[sell & z.get('stop_trigger',neutral), 'signal_label'] = '손절 매도'
    z['upgrade_warning'] = False
    z['alert_event'] = buy | sell
    z['event_kind'] = np.where(buy | sell, '매매 신호', '')
    sell_phase = False; warned = False
    for i in range(len(z)):
        direction = z.sig.iloc[i]
        if direction == '매수':
            sell_phase = False; warned = False
        elif direction == '매도':
            sell_phase = True
            warned = z.signal_label.iloc[i] == '강력매도'
        elif sell_phase and not warned and direction == '관망' and down_count.iloc[i] == 3 and not bull.iloc[i]:
            z.loc[z.index[i], 'signal_label'] = '강력매도 · 추가 경고'
            z.loc[z.index[i], 'upgrade_warning'] = True
            z.loc[z.index[i], 'alert_event'] = True
            z.loc[z.index[i], 'event_kind'] = '추가 매도 경고'
            z.loc[z.index[i], 'strength_count'] = 3
            z.loc[z.index[i], 'reason'] = '기존 매도 이후 하락 조건 강화 · RSI·DMI(ADX≥20)·스토캐스틱 3/3 일치 · 매도 미실행 시 확인용 경고'
            warned = True
    strong = z.signal_label.isin(['강력매수','강력매도'])
    z.loc[strong,'reason'] = z.loc[strong,'reason']+' · 강도 확인: RSI·DMI(ADX≥20)·스토캐스틱 3/3 일치'
    return z


def backtest(z, fee=0.0005, slip=0.0003):
    """Signal at closed candle -> next available open, cash/spot only, >=24h hold."""
    cash = 1.0; qty = 0.0; entry = None; records = []; equity = []
    for i in range(len(z)):
        row = z.iloc[i]
        if i > 0:
            signal = z.sig.iloc[i-1]
            if signal == '매수' and qty == 0:
                price = row.open*(1+slip)
                qty = cash/(price*(1+fee)); cash = 0.0
                entry = (row.time, price, qty*price*(1+fee))
            elif signal == '매도' and qty > 0 and row.time-entry[0] >= pd.Timedelta(days=1):
                price = row.open*(1-slip); cash = qty*price*(1-fee)
                records.append({'매수시각': entry[0], '매도시각': row.time, '매수가': entry[1], '매도가': price, '수익률(%)': (cash/entry[2]-1)*100})
                qty = 0.0; entry = None
        equity.append(cash+qty*row.close*(1-slip)*(1-fee))
    curve = pd.Series(equity, index=z.index)
    dd = (curve/curve.cummax()-1).min()*100 if len(curve) else 0.0
    return pd.DataFrame(records), curve, dd, qty > 0


def closed_candles(raw, now=None):
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    return raw[raw.time.dt.tz_convert('UTC')+pd.Timedelta(days=1) <= now].reset_index(drop=True)


@st.cache_data(ttl=300, show_spinner=False)
def daily_candles(market, count=1000):
    rows = []; cursor = None
    for _ in range((count+199)//200):
        params = {'market': market, 'count': min(200, count-len(rows))}
        if cursor:
            params['to'] = cursor
        for attempt in range(3):
            response = requests.get('https://api.upbit.com/v1/candles/days', params=params, timeout=15)
            if response.status_code != 429:
                break
            time.sleep(0.4*(attempt+1))
        response.raise_for_status(); batch = response.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1]['candle_date_time_utc']+'Z'
        if len(batch) < params['count']:
            break
        time.sleep(0.15)
    if not rows:
        raise ValueError('일봉 데이터가 없습니다.')
    d = pd.DataFrame(rows).drop_duplicates('candle_date_time_utc')
    d['time'] = pd.to_datetime(d.candle_date_time_utc, utc=True).dt.tz_convert('Asia/Seoul')
    d = d.rename(columns={'opening_price':'open', 'high_price':'high', 'low_price':'low', 'trade_price':'close', 'candle_acc_trade_volume':'volume'})
    return d[['time', 'open', 'high', 'low', 'close', 'volume']].sort_values('time').reset_index(drop=True)


def chart(z, days):
    view = z[z.time >= z.time.max()-pd.Timedelta(days=days)].copy()
    dates = view.time.dt.tz_localize(None)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=dates, open=view.open, high=view.high, low=view.low, close=view.close,
        name='일봉', increasing_line_color='#e45f5f', increasing_fillcolor='#e45f5f',
        decreasing_line_color='#397dcc', decreasing_fillcolor='#397dcc', line_width=1.3))
    for _, r in view[view.get('alert_event',view.sig.isin(['매수','매도']))].iterrows():
        buy = r.sig == '매수'; offset = max(r.high-r.low, r.close*.015)
        fig.add_annotation(x=r.time.tz_localize(None), y=r.low-offset*.3 if buy else r.high+offset*.3,
            text='강력매도<br>추가 경고' if r.get('upgrade_warning',False) else r.get('signal_label',r.sig), showarrow=True, arrowhead=2, arrowwidth=4 if str(r.get('signal_label','')).startswith('강력') else 3,
            arrowcolor='#087f72' if buy else '#7841ad', font=dict(color='#087f72' if buy else '#7841ad',size=13 if str(r.get('signal_label','')).startswith('강력') else 12),
            ax=46 if r.get('upgrade_warning',False) else 0, ay=-72 if r.get('upgrade_warning',False) else (42 if buy else -42))
    low, high = view.low.min(), view.high.max()
    padding = max((high-low)*.14, high*.025)
    fig.update_layout(height=410,template='plotly_white',xaxis_rangeslider_visible=False,
        hovermode='x unified',dragmode='pan',showlegend=False,
        margin=dict(l=4,r=8,t=20,b=12),font=dict(size=12,color='#344762'),paper_bgcolor='#ffffff',plot_bgcolor='#ffffff',yaxis=dict(range=[low-padding,high+padding],side='right',tickformat=',.0f',gridcolor='#eaf0f7',nticks=5))
    unit=100000000 if high >= 10000000 else 10000
    ticks=np.linspace(low,high,5)
    fig.update_yaxes(tickmode='array',tickvals=ticks,ticktext=[f'{v/unit:.2f}억' if unit==100000000 else f'{v/unit:,.0f}만' for v in ticks])
    fig.update_xaxes(nticks=5,tickformat='%y-%m-%d',showgrid=False,range=[dates.iloc[0]-pd.Timedelta(days=2),dates.iloc[-1]+pd.Timedelta(days=2)])
    return fig


@st.fragment(run_every="4h")
def main():
    st.set_page_config(page_title='BTC · ETH 일봉 매매 신호',page_icon='₿',layout='wide')
    st.markdown("""<style>
    .stApp{background:#dce9f8;color:#20324d;font-family:system-ui,-apple-system,'Malgun Gothic',sans-serif}
    .block-container{max-width:1180px;padding:4.25rem 1.2rem 2rem}
    [data-testid="stVerticalBlock"]{gap:.65rem}
    h1{font-size:1.65rem!important;line-height:1.25!important;padding:.2rem 0!important;font-weight:750!important;color:#16345a}
    p,label{font-size:15px!important;line-height:1.5!important}
    [data-testid="stCaptionContainer"] p{font-size:13px!important;color:#526681!important}
    [data-testid="stMetric"]{background:#fff;border:1px solid #b6cbe3;border-top:3px solid #4b78ae;padding:10px 14px;border-radius:10px}
    [data-testid="stMetricLabel"] p{font-size:13px!important;color:#526681}
    [data-testid="stMetricValue"],[data-testid="stMetricValue"] *{font-size:1.4rem!important;font-weight:700}
    [data-testid="stExpander"]{background:#fff;border-color:#b6cbe3;border-radius:9px}
    [data-testid="stPlotlyChart"]{background:#fff;border:1px solid #b6cbe3;border-radius:12px;overflow:hidden}
    .stButton button{min-height:40px;border-color:#245b92;border-radius:8px;font-weight:600;background:#245b92;color:#fff}
    .stButton button:hover{background:#194a7a;color:#fff;border-color:#194a7a}
    [data-baseweb="select"]>div{background:#fff;border-color:#c6d5e7;color:#20324d}
    [data-baseweb="tab"][aria-selected="true"]{color:#245b92!important}
    [data-baseweb="tab-highlight"]{background:#245b92!important}
    [data-testid="stMetric"]{box-shadow:0 3px 10px rgba(27,54,88,.09)}
    [data-testid="stExpander"] summary{color:#315375}

    [data-baseweb="tab"]{font-size:15px;font-weight:650}
    @media(max-width:640px){
      .block-container{padding:4rem .65rem 1.5rem}
      h1{font-size:1.3rem!important}
      [data-testid="stHorizontalBlock"]{flex-wrap:nowrap!important;gap:.45rem!important}
      [data-testid="stHorizontalBlock"]>[data-testid="stColumn"]{min-width:0!important;flex:1 1 0!important}
      [data-testid="stMetric"]{padding:8px 6px}
      [data-testid="stMetricValue"],[data-testid="stMetricValue"] *{font-size:1rem!important}
      [data-testid="stMetricLabel"] p{font-size:11px!important}
      p,label{font-size:14px!important}
    }
    </style>""",unsafe_allow_html=True)
    st.title('BTC · ETH 일봉 매매')
    st.caption('확정 일봉 · 4시간 확인 · 종가 10% 손절 · 강력 신호 · MACD 추가 비교 · v11')
    a,b,c=st.columns([2,2,1])
    coin=a.selectbox('코인',['Bitcoin (BTC)','Ethereum (ETH)'])
    period=b.selectbox('차트 기간',['1개월','3개월','6개월','1년','2년'],index=1)
    if c.button('새로고침',use_container_width=True):
        daily_candles.clear()
        st.rerun()
    market='KRW-BTC' if 'BTC' in coin else 'KRW-ETH'
    with st.expander('매매 기준 및 설정',expanded=False):
        mode=st.selectbox('매매 조건',['검증 후 선택한 조건','수동 조건'],index=0)
        selected=OPTIMIZATION_REPORT[market]['applied_params']
        st.caption(f'적용 조건: 다이버전스 {selected["window"]}일 · '+('DMI 하향교차 조기청산' if selected['exit_mode']=='dmi_early' else '지표 방향 확인 청산'))
        st.markdown('**Slow Stochastic: 기간 30 · %K 평활 10 · %D 평활 10(SMA). DMI(14) · RSI(14), RSI 신호선(9).**')
        if selected.get('ema_period',0):
            st.caption(f"추가 지표: EMA({selected['ema_period']}) · 종가가 EMA 위일 때만 기존 매수 허용 · 종가 하향돌파 추가 매도")
        else:
            st.caption('EMA 추가 실험에서 BTC 수익률이 감소하여 기본 조건에서는 적용하지 않습니다. 수동 조건에서 사용할 수 있습니다.')
        st.caption('추가 비교 결과: ETH는 MACD(12,26,9) 히스토그램이 양수일 때만 기존 매수 신호를 허용합니다. BTC는 기존 유지. 거래량·볼린저밴드·ATR·추적청산도 비교했으나 기본 조건에 추가하지 않았습니다. 수동 조건에는 MACD 필터를 적용하지 않습니다.')
        st.caption('강력매수·강력매도: 기존 신호에 RSI 방향, DMI 방향(ADX≥20), Slow %K/%D 방향과 %K 추세가 모두 일치하고 반대 다이버전스가 없을 때 표시합니다. RSI 방향은 다이버전스 또는 RSI가 신호선 위에서 상승/아래에서 하락하는 경우입니다. 강력은 조건 일치도를 뜻하며 적중률을 보장하지 않습니다. 손절은 별도 표시합니다.')
        st.caption('손절: 확정 일봉 종가 ≤ 모의 매수가 × 90%이면 매도. 기존 매도 조건은 유지하며 손절을 우선합니다. 모의 매수가는 매수 신호 다음 일봉 시가에 슬리피지 0.03%를 반영한 가격으로, 실제 계좌 매수가와 다릅니다.')
        st.caption('추가 경고: 일반 매도 이후 RSI·DMI·스토캐스틱이 강력매도 조건을 처음 충족하면 다음 매수 전까지 한 번 알립니다. 최초 매도가 강력매도이면 재경고하지 않습니다. 이미 매도한 경우 추가 거래가 필요하지 않습니다. 매수→강력매수 추가 알림은 차단합니다.')
        use_ema=st.checkbox('수동 조건: EMA(50) 매수 필터 및 하향돌파 매도',value=True)
        window=st.number_input('다이버전스 추세 비교 일봉 수',min_value=5,max_value=60,value=14)
        c1,c2=st.columns(2)
        min_price=c1.number_input('추세 종가 변화 최소(%)',min_value=0.0,max_value=20.0,value=1.0,step=0.5)
        min_rsi=c2.number_input('추세 RSI 변화 최소(포인트)',min_value=0.0,max_value=30.0,value=3.0,step=0.5)
        st.markdown('**다이버전스 매도:** 최근 비교 구간의 종가 추세는 상승, RSI 추세는 하락. **매수:** 종가 추세는 하락, RSI 추세는 상승. 구간 전체의 회귀 추세와 시작·끝 방향을 함께 확인합니다. 매일 같은 방향이어야 하는 조건은 아닙니다.\n\n**교차 매수:** DMI 상향교차 + Slow %K > %D, 또는 Slow %K/%D 상향교차 + +DI > −DI. **교차 매도:** 각각 반대 방향.\n\n다이버전스가 처음 발생하거나 교차 조건이 충족된 확정 일봉에 신호를 냅니다. 매수 이후에는 매도 신호만, 매도 이후에는 매수 신호만 표시합니다. 모든 신호 종류에 동일하게 적용합니다. 매수·매도 조건이 겹치면 관망합니다. RSI 30/70 단독 돌파는 매매 조건에 포함하지 않습니다.')
    market='KRW-BTC' if 'BTC' in coin else 'KRW-ETH'
    try:
        with st.spinner('일봉 데이터를 분석하고 있습니다…'):
            raw=daily_candles(market)
        closed=closed_candles(raw)
        if len(closed) < max(60,int(window)+14):
            st.warning('지표 계산에 필요한 확정 일봉이 부족합니다.'); return
        z=active_signals(indicators(closed),market) if mode=='검증 후 선택한 조건' else signals(indicators(closed),int(window),float(min_price),float(min_rsi),ema_period=50 if use_ema else 0,ema_exit=use_ema,stop_pct=10.0)
        last=z.iloc[-1]; current=raw.iloc[-1]
        checked_at=pd.Timestamp.now(tz='Asia/Seoul')
        alert_key=f'monitor_check_{market}'
        previous_check=st.session_state.get(alert_key,checked_at)
        fresh=z[(z.confirmed_at>previous_check)&z.alert_event]
        for _,event in fresh.iterrows():
            st.toast(f'{market} {event.signal_label} · {event.confirmed_at:%m/%d %H:%M} · {event.reason}',icon='🔔')
        st.session_state[alert_key]=checked_at
        a,b,c=st.columns(3)
        a.metric('조회 가격',f'₩{current.close:,.0f}')
        b.metric('일봉 신호',last.signal_label)
        c.metric('신호 상태',last.signal_state)
        st.caption(f'판단 일봉 {last.time:%m/%d} · 확인 {checked_at:%H:%M} KST · 청록 매수 / 보라 매도')
        days={'1개월':30,'3개월':90,'6개월':180,'1년':365,'2년':730}[period]
        st.plotly_chart(chart(z,days),use_container_width=True)
        with st.expander('판단 근거 · 모니터링 안내',expanded=False):
            st.caption(VERSION)
            st.write(f"신호 확정: {last.confirmed_at:%Y-%m-%d %H:%M} KST · 근거: {last.reason or '새로운 매매 조건이 없습니다.'}")
            if last.signal_state == '매수 이후' and pd.notna(last.stop_price):
                st.write(f'현재 모의 손절 기준 ₩{last.stop_price:,.0f} · 모의 매수가 ₩{last.sim_entry_price:,.0f}')
            st.caption(f'이번 확인 {checked_at:%Y-%m-%d %H:%M} · 다음 앱 확인 {checked_at+pd.Timedelta(hours=4):%m-%d %H:%M} KST')
            st.caption('앱이 열려 있으면 4시간마다 재확인합니다. 닫아둔 경우 ChatGPT 예약 모니터링으로 알립니다. 일봉은 KST 09:00 마감, 진행 중인 일봉은 제외합니다. 자동주문은 없습니다.')
            st.caption('종가 10% 손절은 모의 매수가 기준입니다. 실제 계좌와 연동되지 않으며, 마감·알림·체결 지연으로 실제 손실이 10%를 넘을 수 있습니다. 조회 가격은 5분 캐시 또는 새로고침으로 갱신합니다.')
        tab1,tab2=st.tabs(['매매 신호 이력','백테스트'])
        with tab1:
            events=z[z.alert_event].sort_values('confirmed_at',ascending=False).head(100)
            if events.empty:
                st.info('매매 신호가 없습니다.')
            else:
                table=events[['confirmed_at','signal_label','event_kind','close','reason']].copy()
                table['confirmed_at']=table.confirmed_at.dt.strftime('%Y-%m-%d %H:%M')
                table.columns=['신호 확정시각(KST)','신호','구분','판단 종가(원)','판단 근거']
                st.dataframe(table,use_container_width=True,hide_index=True)
                st.download_button('신호 이력 CSV 다운로드',table.to_csv(index=False).encode('utf-8-sig'),file_name=f'{market}_daily_signals.csv',mime='text/csv')
                st.caption('매매 신호는 매수·매도 순서로 번갈아 표시됩니다. 일반 매도 이후 하락 조건이 강해지면 추가 경고를 한 번 표시합니다. 추가 경고는 모의 거래나 보유 상태를 바꾸지 않습니다.')
        with tab2:
            report=EXTENDED_REPORT[market]
            st.markdown('**추가 지표 168개 조합: 최근 1년 모의 비교**')
            st.caption(f'{report["year_start"][:10]} ~ {report["year_end"][:10]} · 코인별 84개 조건 · 거래량, MACD, 볼린저밴드, ATR, 고점 대비 추적청산 비교')
            comparison=pd.DataFrame([
                {'조건':name,'수익률(%)':row['year']['return'],'최대 낙폭(%)':row['year']['dd'],'완료 거래':row['year']['trades'],'1천만원 최종자산(원)':round(10_000_000*(1+row['year']['return']/100))}
                for name,row in [('수정 전(v10)',report['baseline']),('학습·중간구간 최고 점수 후보',report['candidate']),('실제 적용(v11)',report['applied'])]])
            st.dataframe(comparison,use_container_width=True,hide_index=True)
            st.caption(report['decision']+' · '+('MACD(12,26,9) 양수 매수 필터' if report['applied']['profile'] else '기존 조건 유지'))
            st.caption('앞선 기간·중간 기간·최근 기간 및 1년 성과를 모두 조건 선택에 사용했습니다. 최근 1년은 다른 구간과 겹치며 기존 연구 데이터도 재사용했습니다. 독립적인 미래 검증이 아니고, 모든 가능한 지표 조합을 탐색한 결과도 아닙니다. 과적합 가능성이 있습니다.')
            with st.expander('기간별 성과와 거래비용 민감도',expanded=False):
                rows=[]
                for label,key in [('앞선 기간','train'),('중간 기간','validation'),('최근 기간','recent'),('최근 1년','year')]:
                    rows.append({'기간':label,'수정 전 수익률(%)':report['baseline'][key]['return'],'적용 수익률(%)':report['applied'][key]['return'],'적용 최대 낙폭(%)':report['applied'][key]['dd']})
                st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
                st.caption(f"거래비용 두 배(수수료 0.10%, 슬리피지 0.06%, 각 편도): 최근 1년 수정 전 {report['baseline_double_cost']['return']:+.2f}% → 적용 {report['applied_double_cost']['return']:+.2f}%")
                st.caption('아래 네 구간은 각각 현금 100%로 시작한 별도 모의 계산으로 수익률을 단순 합산할 수 없습니다.')
                st.dataframe(pd.DataFrame([{'기간':q['start'][:10]+' ~ '+q['end'][:10],'수정 전(%)':q['baseline']['return'],'적용(%)':q['applied']['return']} for q in report['quarters']]),hide_index=True,use_container_width=True)
            st.caption('설정한 차트 기간의 시작은 현금 100%로 가정합니다. 지표는 이전 데이터로 계산하고, 확정 신호 다음 일봉 시가에 체결합니다. 최소 보유 24시간 · 거래당 수수료 0.05% · 슬리피지 0.03%.')
            btdata=z[z.time >= z.time.max()-pd.Timedelta(days=days)].reset_index(drop=True)
            trades,equity,dd,holding=backtest(btdata)
            total=(equity.iloc[-1]-1)*100
            buyhold=(btdata.close.iloc[-1]*(1-.0003)*(1-.0005)/(btdata.open.iloc[0]*(1+.0003)*(1+.0005))-1)*100
            a,b,c,d=st.columns(4)
            a.metric('전략 수익률',f'{total:+.2f}%')
            b.metric('단순 보유 수익률',f'{buyhold:+.2f}%')
            c.metric('최대 낙폭',f'{dd:.2f}%')
            d.metric('완료 거래',f'{len(trades)}회')
            st.caption('현재 모의 포지션: '+('보유 중(평가손익 및 예상 청산비용 포함)' if holding else '현금'))
            st.line_chart(pd.DataFrame({'전략 자산(초기=100)':equity.values*100},index=btdata.time.dt.tz_localize(None)))
            if not trades.empty:
                st.dataframe(trades,use_container_width=True,hide_index=True)
            else:
                st.info('선택한 기간에 완료된 매매가 없습니다.')
            st.caption('과거 데이터의 모의 결과이며 미래 수익을 보장하지 않습니다.')
    except (requests.RequestException,ValueError,KeyError) as exc:
        st.error('일봉 데이터를 불러오지 못했습니다. 시세 새로고침을 눌러 다시 시도해 주세요.')
        st.caption(str(exc))


if __name__ == '__main__':
    main()

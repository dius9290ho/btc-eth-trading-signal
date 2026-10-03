import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

VERSION = '일봉 매매 신호 · 4시간 모니터링 / v4'


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
    x['RSIsignal'] = x.RSI.rolling(9, min_periods=9).mean()
    lowest = x.low.rolling(stoch_period, min_periods=stoch_period).min()
    highest = x.high.rolling(stoch_period, min_periods=stoch_period).max()
    spread = highest-lowest
    fast = (100*(x.close-lowest)/spread.replace(0, np.nan)).where(spread != 0, 50)
    x['SlowK'] = fast.rolling(smooth_k, min_periods=smooth_k).mean()
    x['SlowD'] = x.SlowK.rolling(smooth_d, min_periods=smooth_d).mean()
    return x


def signals(x, window=14, min_price=1.0, min_rsi=3.0):
    """Trailing regression divergence; no future pivots or retroactive signals."""
    z = x.copy()
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
    z['bull_div'] = z.bull_state & ~z.bull_state.shift(1, fill_value=False)
    z['bear_div'] = z.bear_state & ~z.bear_state.shift(1, fill_value=False)
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
            buys.append(f'상승 다이버전스 · {window}일 종가 하락 / RSI 상승')
        if r.bear_div:
            sells.append(f'하락 다이버전스 · {window}일 종가 상승 / RSI 하락')
        if r.DMIup and r.SlowK > r.SlowD:
            buys.append('DMI 상향교차 + Slow %K > %D')
        if r.DMIdown and r.SlowK < r.SlowD:
            sells.append('DMI 하향교차 + Slow %K < %D')
        if r.StochUp and r.PDI > r.MDI:
            buys.append('Slow %K/%D 상향교차 + DMI 상승 방향')
        if r.StochDown and r.PDI < r.MDI:
            sells.append('Slow %K/%D 하향교차 + DMI 하락 방향')
        z.loc[i, 'buy_reason'] = ' / '.join(buys)
        z.loc[i, 'sell_reason'] = ' / '.join(sells)
    buy, sell = z.buy_reason.ne(''), z.sell_reason.ne('')
    z['sig'] = np.select([buy & sell, buy, sell], ['관망(충돌)', '매수', '매도'], default='관망')
    z['candidate_sig'] = z.sig.copy()
    z['reason'] = (z.buy_reason+' / '+z.sell_reason).str.strip(' /')
    last_signal = None
    states = []
    for i in range(len(z)):
        signal = z.sig.iloc[i]
        if signal in ('매수', '매도'):
            if signal == last_signal:
                z.loc[z.index[i], 'sig'] = '관망'
                z.loc[z.index[i], 'reason'] = f'동일 {signal} 조건 반복 · 새 신호 없음'
            else:
                last_signal = signal
        states.append('매수 이후' if last_signal == '매수' else '매도 이후' if last_signal == '매도' else '신호 대기')
    z['signal_state'] = states
    z['confirmed_at'] = z.time+pd.Timedelta(days=1)
    z['div_state'] = np.select([z.bull_state,z.bear_state],['상승 다이버전스(매수 방향)','하락 다이버전스(매도 방향)'],default='없음')
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
        name='일봉', increasing_line_color='#de5252', increasing_fillcolor='#de5252',
        decreasing_line_color='#3183c8', decreasing_fillcolor='#3183c8', line_width=1.3))
    for _, r in view[view.sig.isin(['매수','매도'])].iterrows():
        buy = r.sig == '매수'; offset = max(r.high-r.low, r.close*.015)
        fig.add_annotation(x=r.time.tz_localize(None), y=r.low-offset*.3 if buy else r.high+offset*.3,
            text=r.sig, showarrow=True, arrowhead=2, arrowwidth=3,
            arrowcolor='#073da8' if buy else '#7c168e', font=dict(color='#073da8' if buy else '#7c168e',size=14),
            ax=0, ay=42 if buy else -42)
    low, high = view.low.min(), view.high.max()
    padding = max((high-low)*.14, high*.025)
    fig.update_layout(height=650,template='plotly_white',xaxis_rangeslider_visible=False,
        hovermode='x unified',dragmode='pan',showlegend=False,
        margin=dict(l=8,r=12,t=30,b=20),yaxis=dict(range=[low-padding,high+padding],side='right',tickformat=',.0f'))
    fig.update_xaxes(tickformat='%y-%m-%d',showgrid=False,range=[dates.iloc[0]-pd.Timedelta(days=2),dates.iloc[-1]+pd.Timedelta(days=2)])
    return fig


@st.fragment(run_every="4h")
def main():
    st.set_page_config(page_title='BTC · ETH 일봉 매매 신호',page_icon='₿',layout='wide')
    st.markdown('<style>.block-container{max-width:1400px;padding-top:1.2rem}[data-testid="stMetric"]{background:#f4f7fb;padding:12px;border-radius:12px}[data-testid="stMetricValue"]{font-size:1.7rem}</style>',unsafe_allow_html=True)
    st.title('₿ BTC · ETH 일봉 매매 신호')
    st.caption(VERSION+' · 확정 일봉 기준 · 자동주문 없음')
    a,b,c=st.columns([2,2,1])
    coin=a.selectbox('코인',['Bitcoin (BTC)','Ethereum (ETH)'])
    period=b.selectbox('차트 기간',['1개월','3개월','6개월','1년','2년'],index=1)
    if c.button('시세 새로고침',use_container_width=True):
        daily_candles.clear()
        st.rerun()
    with st.expander('매매 기준 및 설정',expanded=False):
        st.markdown('**Slow Stochastic: 기간 30 · %K 평활 10 · %D 평활 10(SMA). DMI(14) · RSI(14), RSI 신호선(9).**')
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
        z=signals(indicators(closed),int(window),float(min_price),float(min_rsi))
        last=z.iloc[-1]; current=raw.iloc[-1]
        checked_at=pd.Timestamp.now(tz='Asia/Seoul')
        st.caption(f'4시간 자동 모니터링 · 이번 확인: {checked_at:%Y-%m-%d %H:%M} KST · 다음 확인: {checked_at+pd.Timedelta(hours=4):%m-%d %H:%M} KST')
        alert_key=f'monitor_check_{market}'
        previous_check=st.session_state.get(alert_key,checked_at)
        fresh=z[(z.confirmed_at>previous_check)&z.sig.isin(['매수','매도'])]
        for _,event in fresh.iterrows():
            st.toast(f'{market} {event.sig} · {event.confirmed_at:%m/%d %H:%M} · {event.reason}',icon='🔔')
        st.session_state[alert_key]=checked_at
        st.caption('앱이 열려 있는 동안 4시간마다 재확인합니다. 앱을 닫아둔 경우의 알림은 ChatGPT 예약 모니터링으로 전달됩니다.')
        a,b,c=st.columns(3)
        a.metric('최근 조회 가격',f'₩{current.close:,.0f}')
        b.metric('확정 일봉 신호',last.sig)
        c.metric('최근 신호 상태',last.signal_state)
        st.info(f"판단 일봉: {last.time:%Y-%m-%d} · 신호 확정: {last.confirmed_at:%Y-%m-%d %H:%M} KST\n\n근거: {last.reason or '새로운 매매 조건이 없습니다.'}")
        st.caption(f'업비트 일봉은 한국시간 09:00에 마감합니다. 진행 중인 일봉은 신호와 백테스트에서 제외됩니다. 최근 조회 캔들 시작: {current.time:%Y-%m-%d %H:%M} KST · 가격은 5분 캐시 또는 새로고침으로 갱신됩니다.')
        days={'1개월':30,'3개월':90,'6개월':180,'1년':365,'2년':730}[period]
        st.plotly_chart(chart(z,days),use_container_width=True)
        st.caption('파란 화살표: 매수 · 보라 화살표: 매도. 같은 방향의 신호는 반복 표시하지 않습니다. 일봉 확대는 차트의 + 버튼을 사용해 주세요.')
        tab1,tab2=st.tabs(['매매 신호 이력','백테스트'])
        with tab1:
            events=z[z.sig.isin(['매수','매도'])].sort_values('confirmed_at',ascending=False).head(100)
            if events.empty:
                st.info('매매 신호가 없습니다.')
            else:
                table=events[['confirmed_at','sig','close','reason']].copy()
                table['confirmed_at']=table.confirmed_at.dt.strftime('%Y-%m-%d %H:%M')
                table.columns=['신호 확정시각(KST)','신호','판단 종가(원)','판단 근거']
                st.dataframe(table,use_container_width=True,hide_index=True)
                st.download_button('신호 이력 CSV 다운로드',table.to_csv(index=False).encode('utf-8-sig'),file_name=f'{market}_daily_signals.csv',mime='text/csv')
                st.caption('신호 이력은 매수·매도 순서로 번갈아 표시됩니다. 최근 신호 상태는 앱의 신호 이력 기준이며 실제 계좌와 연동되지 않습니다.')
        with tab2:
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

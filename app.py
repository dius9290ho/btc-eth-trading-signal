import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

VERSION = '일봉 DMI · RSI · Divergence / 2026-10-03'


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


def indicators(raw, period=14):
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
    return x


def signals(x, oversold=30, overbought=70, pivot=2, max_gap=60):
    """Only past/current rows are used; pivots are reported on confirmation day."""
    z = x.copy()
    z['buy_reason'] = ''; z['sell_reason'] = ''
    z['bull_div'] = False; z['bear_div'] = False
    z['div_from'] = -1; z['div_to'] = -1
    previous_low = previous_high = None
    for i in range(1, len(z)):
        r, prev = z.iloc[i], z.iloc[i-1]
        buys, sells = [], []
        if pd.notna(r.RSI) and pd.notna(prev.RSI):
            if r.PDI > r.MDI and prev.PDI <= prev.MDI and r.RSI < overbought:
                buys.append('DMI 상향교차 · RSI 과매수 제외')
            if r.PDI < r.MDI and prev.PDI >= prev.MDI and r.RSI > oversold:
                sells.append('DMI 하향교차 · RSI 과매도 제외')
            if prev.RSI <= oversold < r.RSI:
                buys.append(f'RSI {oversold} 상향돌파')
            if prev.RSI >= overbought > r.RSI:
                sells.append(f'RSI {overbought} 하향돌파')
        k = i-pivot
        if k >= pivot and pd.notna(z.RSI.iloc[k]):
            value = z.close.iloc[k]
            left = z.close.iloc[k-pivot:k]; right = z.close.iloc[k+1:i+1]
            is_low = value < left.min() and value < right.min()
            is_high = value > left.max() and value > right.max()
            if is_low:
                a = previous_low
                if a is not None and k-a <= max_gap and value < z.close.iloc[a] and z.RSI.iloc[k] > z.RSI.iloc[a]:
                    buys.append('상승 다이버전스 · 종가 저점↓ / RSI 저점↑')
                    z.loc[i, ['bull_div', 'div_from', 'div_to']] = [True, a, k]
                previous_low = k
            if is_high:
                a = previous_high
                if a is not None and k-a <= max_gap and value > z.close.iloc[a] and z.RSI.iloc[k] < z.RSI.iloc[a]:
                    sells.append('하락 다이버전스 · 종가 고점↑ / RSI 고점↓')
                    z.loc[i, ['bear_div', 'div_from', 'div_to']] = [True, a, k]
                previous_high = k
        z.loc[i, 'buy_reason'] = ' / '.join(buys)
        z.loc[i, 'sell_reason'] = ' / '.join(sells)
    buy, sell = z.buy_reason.ne(''), z.sell_reason.ne('')
    z['sig'] = np.select([buy & sell, buy, sell], ['관망(충돌)', '매수', '매도'], default='관망')
    z['reason'] = (z.buy_reason+' / '+z.sell_reason).str.strip(' /')
    z['confirmed_at'] = z.time+pd.Timedelta(days=1)
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
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[.52,.24,.24], vertical_spacing=.045)
    dates = view.time.dt.tz_localize(None)
    fig.add_trace(go.Candlestick(x=dates, open=view.open, high=view.high, low=view.low, close=view.close, name='일봉', increasing_line_color='#de5252', decreasing_line_color='#3183c8'), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=view.RSI, name='RSI', line=dict(color='#7953be',width=2)),row=2,col=1)
    for column, label, color in [('PDI','+DI','#089f80'),('MDI','−DI','#e87538'),('ADX','ADX','#8a95a6')]:
        fig.add_trace(go.Scatter(x=dates,y=view[column],name=label,line=dict(color=color,width=2 if column != 'ADX' else 1)),row=3,col=1)
    for level in [30,70]:
        fig.add_hline(y=level,row=2,col=1,line_dash='dot',line_color='#aab4c2')
    for _, r in view[view.sig.isin(['매수','매도'])].iterrows():
        buy = r.sig == '매수'; offset = max(r.high-r.low, r.close*.03)
        fig.add_annotation(x=r.time.tz_localize(None), y=r.low-offset*.35 if buy else r.high+offset*.35,
            text=r.sig, showarrow=True, arrowhead=2, arrowwidth=3,
            arrowcolor='#073da8' if buy else '#7c168e', font=dict(color='#073da8' if buy else '#7c168e',size=12),
            ax=0, ay=42 if buy else -42,row=1,col=1)
    for _, r in view[view.bull_div | view.bear_div].iterrows():
        a,b = int(r.div_from),int(r.div_to); points=z.iloc[[a,b]]
        color = '#073da8' if r.bull_div else '#7c168e'
        if points.time.iloc[0] < view.time.iloc[0]:
            continue
        for column, panel in [('close',1),('RSI',2)]:
            fig.add_trace(go.Scatter(x=points.time.dt.tz_localize(None),y=points[column],mode='lines+markers',line=dict(color=color,width=2,dash='dash'),showlegend=False,name='다이버전스 비교점'),row=panel,col=1)
    fig.update_layout(height=700,template='plotly_white',xaxis_rangeslider_visible=False,hovermode='x unified',margin=dict(l=8,r=8,t=20,b=10),legend=dict(orientation='h',y=1.06))
    fig.update_yaxes(range=[0,100],row=2,col=1)
    fig.update_xaxes(tickformat='%y-%m-%d',showgrid=False)
    return fig


def main():
    st.set_page_config(page_title='BTC · ETH 일봉 매매 신호',page_icon='₿',layout='wide')
    st.markdown('<style>.block-container{max-width:1400px;padding-top:1.2rem}[data-testid="stMetric"]{background:#f4f7fb;padding:12px;border-radius:12px}[data-testid="stMetricValue"]{font-size:1.7rem}</style>',unsafe_allow_html=True)
    st.title('₿ BTC · ETH 일봉 매매 신호')
    st.caption(VERSION+' · 확정 일봉 기준 · 자동주문 없음')
    a,b,c=st.columns([2,2,1])
    coin=a.selectbox('코인',['Bitcoin (BTC)','Ethereum (ETH)'])
    period=b.selectbox('차트 기간',['1개월','3개월','6개월','1년','2년'],index=3)
    if c.button('시세 새로고침',use_container_width=True):
        daily_candles.clear()
        st.rerun()
    with st.expander('매매 기준 및 설정',expanded=False):
        s1,s2,s3=st.columns(3)
        length=s1.number_input('DMI · RSI 기간',min_value=5,max_value=50,value=14)
        lower=s2.number_input('RSI 과매도 기준',min_value=10,max_value=45,value=30)
        upper=s3.number_input('RSI 과매수 기준',min_value=55,max_value=90,value=70)
        pivot=st.number_input('고점·저점 확인 일봉 수(좌우 각각)',min_value=1,max_value=5,value=2)
        st.markdown('**매수:** +DI의 −DI 상향교차(RSI 과매수 제외), RSI 과매도선 상향돌파, 또는 상승 다이버전스.\n\n**매도:** +DI의 −DI 하향교차(RSI 과매도 제외), RSI 과매수선 하향돌파, 또는 하락 다이버전스.\n\n**다이버전스:** 인접한 두 종가 저점은 낮아지나 해당 RSI는 높아지면 매수, 두 종가 고점은 높아지나 RSI는 낮아지면 매도. 비교점 간격은 최대 60일입니다. 신호는 뒤쪽 일봉으로 고점·저점이 확인된 날에 발생합니다.\n\n어느 한 조건이라도 충족하면 신호를 내며, 매수·매도 조건이 겹치면 관망합니다. ADX는 참고용입니다.')
    market='KRW-BTC' if 'BTC' in coin else 'KRW-ETH'
    try:
        with st.spinner('일봉 데이터를 분석하고 있습니다…'):
            raw=daily_candles(market)
        closed=closed_candles(raw)
        if len(closed) < 2*length+2*pivot+5:
            st.warning('지표 계산에 필요한 확정 일봉이 부족합니다.'); return
        z=signals(indicators(closed,int(length)),int(lower),int(upper),int(pivot))
        last=z.iloc[-1]; current=raw.iloc[-1]
        a,b,c,d=st.columns(4)
        a.metric('최근 조회 가격',f'₩{current.close:,.0f}')
        b.metric('확정 일봉 RSI',f'{last.RSI:.1f}')
        c.metric('+DI / −DI',f'{last.PDI:.1f} / {last.MDI:.1f}')
        d.metric('확정 일봉 신호',last.sig)
        st.info(f"판단 일봉: {last.time:%Y-%m-%d} · 신호 확정: {last.confirmed_at:%Y-%m-%d %H:%M} KST\n\n근거: {last.reason or '새로운 매매 조건이 없습니다.'}")
        st.caption(f'업비트 일봉은 한국시간 09:00에 마감합니다. 진행 중인 일봉은 신호와 백테스트에서 제외됩니다. 최근 조회 캔들 시작: {current.time:%Y-%m-%d %H:%M} KST · 가격은 5분 캐시 또는 새로고침으로 갱신됩니다.')
        days={'1개월':30,'3개월':90,'6개월':180,'1년':365,'2년':730}[period]
        st.plotly_chart(chart(z,days),use_container_width=True)
        st.caption('파란 화살표: 매수 · 보라 화살표: 매도 · 점선: 다이버전스 비교점. 화살표는 신호를 확인한 일봉에 표시됩니다.')
        tab1,tab2=st.tabs(['매매 신호 이력','백테스트'])
        with tab1:
            events=z[z.sig.ne('관망')].sort_values('confirmed_at',ascending=False).head(100)
            if events.empty:
                st.info('매매 신호가 없습니다.')
            else:
                table=events[['confirmed_at','sig','close','RSI','PDI','MDI','reason']].copy()
                table['confirmed_at']=table.confirmed_at.dt.strftime('%Y-%m-%d %H:%M')
                table.columns=['신호 확정시각(KST)','신호','판단 종가(원)','RSI','+DI','−DI','판단 근거']
                st.dataframe(table,use_container_width=True,hide_index=True)
                st.download_button('신호 이력 CSV 다운로드',table.to_csv(index=False).encode('utf-8-sig'),file_name=f'{market}_daily_signals.csv',mime='text/csv')
                st.caption('이력은 보유 여부와 관계없이 발생한 지표 신호입니다. 매도는 보유 코인의 청산 신호입니다.')
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

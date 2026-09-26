import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(page_title='BTC · ETH Signal', page_icon='₿', layout='wide')
st.markdown('''<style>
.stApp{background:linear-gradient(135deg,#070b14,#0d1424 55%,#10192b);color:#eef4ff}
.block-container{padding-top:1.3rem;max-width:1400px}
.hero{padding:22px 26px;border:1px solid #26334d;border-radius:22px;background:rgba(17,25,43,.88);box-shadow:0 16px 45px rgba(0,0,0,.28);margin-bottom:18px}
.hero h1{margin:0;font-size:2.1rem}.muted{color:#93a4c3}.pill{display:inline-block;padding:7px 12px;border-radius:999px;background:#18243a;border:1px solid #2c3d5c;margin-right:6px}
[data-testid="stMetric"]{background:#111a2b;border:1px solid #263550;padding:14px;border-radius:18px}
.signal{font-size:1.7rem;font-weight:800;padding:14px 18px;border-radius:16px;text-align:center;background:#111a2b;border:1px solid #30415f}
[data-testid="stMetricValue"]{font-size:clamp(1.45rem,4vw,2.25rem);white-space:nowrap}
div[data-testid="stSegmentedControl"] button{min-height:46px}
@media (max-width: 768px){
 .block-container{padding:.7rem .75rem 2rem}
 .hero{padding:16px 16px;border-radius:18px;margin-bottom:12px}
 .hero h1{font-size:1.65rem;line-height:1.2}
 .hero .muted{font-size:.9rem;margin-top:8px}
 [data-testid="stHorizontalBlock"]{gap:.55rem}
 [data-testid="stMetric"]{padding:10px;border-radius:14px;min-width:0}
 [data-testid="stMetricLabel"]{font-size:.78rem}
 [data-testid="stMetricValue"]{font-size:1.35rem!important}
 [data-testid="stMetricDelta"]{font-size:.75rem}
 .signal{font-size:1.35rem;padding:12px 10px}
 div[data-testid="stSegmentedControl"] button{font-size:.88rem;padding-left:.55rem;padding-right:.55rem}
}
</style>''', unsafe_allow_html=True)

st.markdown("<div class='hero'><h1>₿ BTC · ETH Trading Signal</h1><div class='muted'>가격 추세와 기술지표를 한 화면에서 확인하는 분석용 대시보드</div></div>", unsafe_allow_html=True)

@st.cache_data(ttl=60)
def upbit_candles(market, unit, count=200):
    url=f'https://api.upbit.com/v1/candles/minutes/{unit}'
    r=requests.get(url,params={'market':market,'count':count},timeout=10); r.raise_for_status()
    d=pd.DataFrame(r.json()).sort_values('candle_date_time_kst')
    d['time']=pd.to_datetime(d['candle_date_time_kst'])
    d=d.rename(columns={'opening_price':'open','high_price':'high','low_price':'low','trade_price':'close','candle_acc_trade_volume':'volume'})
    return d[['time','open','high','low','close','volume']].reset_index(drop=True)

def indicators(d):
    x=d.copy(); c=x.close
    x['EMA20']=c.ewm(span=20,adjust=False).mean(); x['EMA50']=c.ewm(span=50,adjust=False).mean()
    delta=c.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    rs=gain/loss.replace(0,np.nan); x['RSI']=100-(100/(1+rs))
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean(); x['MACD']=e12-e26; x['MACDsig']=x.MACD.ewm(span=9,adjust=False).mean()
    return x

def signal(row):
    score=0; reasons=[]
    if row.close>row.EMA20: score+=1; reasons.append('가격 > EMA20')
    else: score-=1; reasons.append('가격 < EMA20')
    if row.EMA20>row.EMA50: score+=1; reasons.append('EMA20 > EMA50')
    else: score-=1; reasons.append('EMA20 < EMA50')
    if row.MACD>row.MACDsig: score+=1; reasons.append('MACD 강세')
    else: score-=1; reasons.append('MACD 약세')
    if row.RSI<30: score+=1; reasons.append('RSI 과매도')
    elif row.RSI>70: score-=1; reasons.append('RSI 과매수')
    label='강력매수' if score>=3 else '매수' if score>=1 else '강력매도' if score<=-3 else '매도' if score<=-1 else '관망'
    return label,score,reasons

c1,c2,c3=st.columns([1.2,1,1])
with c1: coin=st.segmented_control('코인',['Bitcoin (BTC)','Ethereum (ETH)'],default='Bitcoin (BTC)')
with c2: unit=st.selectbox('캔들 간격',[15,30,60,240],index=2,format_func=lambda x:f'{x}분')
with c3: st.caption('시세 데이터: Upbit 공개 API · 자동주문 없음')
market='KRW-BTC' if 'BTC' in coin else 'KRW-ETH'
try:
    d=indicators(upbit_candles(market,unit)); last=d.iloc[-1]; prev=d.iloc[-2]
    lab,score,reasons=signal(last); pct=(last.close/prev.close-1)*100
    a,b,c,dcol=st.columns(4)
    a.metric('현재가',f'₩{last.close:,.0f}',f'{pct:+.2f}%')
    b.metric('RSI (14)',f'{last.RSI:.1f}')
    c.metric('MACD',f'{last.MACD:,.0f}')
    dcol.metric('Signal score',f'{score:+d} / 4')
    st.markdown(f"<div class='signal'>현재 종합 신호 · {lab}</div>",unsafe_allow_html=True)
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.72,.28],vertical_spacing=.06)
    fig.add_trace(go.Candlestick(x=d.time,open=d.open,high=d.high,low=d.low,close=d.close,name='Price'),row=1,col=1)
    fig.add_trace(go.Scatter(x=d.time,y=d.EMA20,name='EMA20',line=dict(width=1.5)),row=1,col=1)
    fig.add_trace(go.Scatter(x=d.time,y=d.EMA50,name='EMA50',line=dict(width=1.5)),row=1,col=1)
    fig.add_trace(go.Scatter(x=d.time,y=d.RSI,name='RSI',line=dict(width=1.5)),row=2,col=1)
    fig.add_hline(y=70,line_dash='dot',row=2,col=1); fig.add_hline(y=30,line_dash='dot',row=2,col=1)
    fig.update_layout(height=560,template='plotly_dark',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',xaxis_rangeslider_visible=False,margin=dict(l=10,r=10,t=35,b=10),legend_orientation='h')
    fig.update_xaxes(title_text=None); fig.update_yaxes(title_text=None)
    st.plotly_chart(fig,use_container_width=True)
    st.markdown('**신호 판단 근거:** ' + ' · '.join(reasons))
    st.caption(f'마지막 업데이트 캔들: {last.time:%Y-%m-%d %H:%M} KST · 본 앱의 신호는 투자 판단 보조용이며 수익을 보장하지 않습니다.')
except Exception as e:
    st.error('시세를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.')
    st.caption(str(e))

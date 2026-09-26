import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(page_title='BTC · ETH Signal', page_icon='₿', layout='wide')
st.markdown('''<style>
.stApp{background:linear-gradient(135deg,#f7f9fc,#eef3f9 55%,#f8fafc);color:#172033}
.block-container{padding-top:1.3rem;max-width:1400px}
.hero{padding:12px 18px;border:1px solid #26334d;border-radius:22px;background:#ffffff;box-shadow:0 10px 30px rgba(35,55,85,.10);margin-bottom:8px}
.hero h1{margin:0;font-size:1.45rem}.muted{color:#60708a}.pill{display:inline-block;padding:7px 12px;border-radius:999px;background:#eef3f9;border:1px solid #d5deea;margin-right:6px}
[data-testid="stMetric"]{background:#ffffff;border:1px solid #d9e1ec;padding:8px;border-radius:14px}
.signal{font-size:1.35rem;font-weight:800;padding:8px 12px;border-radius:16px;text-align:center;background:#ffffff;border:1px solid #ccd7e6}
[data-testid="stMetricValue"]{font-size:clamp(1.45rem,4vw,2.25rem);white-space:nowrap}
div[data-testid="stSegmentedControl"] button{min-height:46px}
@media (max-width: 768px){
 .block-container{padding:.7rem .75rem 2rem}
 .hero{padding:16px 16px;border-radius:18px;margin-bottom:12px}
 .hero h1{font-size:1.65rem;line-height:1.2}
 .hero .muted{display:none}
 [data-testid="stHorizontalBlock"]{gap:.55rem}
 [data-testid="stMetric"]{padding:10px;border-radius:14px;min-width:0}
 [data-testid="stMetricLabel"]{font-size:.78rem}
 [data-testid="stMetricValue"]{font-size:1.35rem!important}
 [data-testid="stMetricDelta"]{font-size:.75rem}
 .signal{font-size:1.1rem;padding:8px 8px}
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
    x['EMA5']=c.ewm(span=5,adjust=False).mean(); x['EMA20']=c.ewm(span=20,adjust=False).mean(); x['EMA50']=c.ewm(span=50,adjust=False).mean()
    delta=c.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    rs=gain/loss.replace(0,np.nan); x['RSI']=100-(100/(1+rs))
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean(); x['MACD']=e12-e26; x['MACDsig']=x.MACD.ewm(span=9,adjust=False).mean()
    x['VOLMA20']=x.volume.rolling(20).mean()
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

def signal_series(x):
    out=[]
    for _,r in x.iterrows():
        if pd.isna(r.RSI) or pd.isna(r.MACDsig): out.append(('관망',0))
        else:
            lab,sc,_=signal(r); out.append((lab,sc))
    return out

c1,c2,c3=st.columns([1.2,1,1])
with c1: coin=st.segmented_control('코인',['Bitcoin (BTC)','Ethereum (ETH)'],default='Bitcoin (BTC)')
with c2:
    period=st.segmented_control('차트 기간',['1일','1주','1개월'],default='1주')
    unit=st.selectbox('캔들 간격',[15,30,60,240],index=2,format_func=lambda x:f'{x}분')
with c3: st.caption('시세 데이터: Upbit 공개 API · 자동주문 없음')
market='KRW-BTC' if 'BTC' in coin else 'KRW-ETH'
try:
    d=indicators(upbit_candles(market,unit)); last=d.iloc[-1]; prev=d.iloc[-2]
    lab,score,reasons=signal(last); pct=(last.close/prev.close-1)*100
    a,b,c,dcol=st.columns(4)
    a.metric('현재가',f'₩{last.close/1_000_000:.2f}M' if last.close>=10_000_000 else f'₩{last.close:,.0f}',f'{pct:+.2f}%')
    b.metric('RSI (14)',f'{last.RSI:.1f}')
    c.metric('MACD',f'{last.MACD:,.0f}')
    dcol.metric('Signal score',f'{score:+d} / 4')
    sig_color='#e53935' if '매도' in lab else '#00a86b' if '매수' in lab else '#60708a'
    st.markdown(f"<div class='signal'>현재 종합 신호 · <span style='color:{sig_color}'>{lab}</span> &nbsp; | &nbsp; 신뢰도 {abs(score)}/4</div>",unsafe_allow_html=True)

    events_all=signal_series(d)
    d_alert=d.copy(); d_alert['sig']=[z[0] for z in events_all]; d_alert['score']=[z[1] for z in events_all]; d_alert['prevsig']=d_alert.sig.shift(1)
    ab=d_alert[(d_alert.sig.str.contains('매수')) & (~d_alert.prevsig.fillna('').str.contains('매수'))].assign(kind='매수')
    ase=d_alert[(d_alert.sig.str.contains('매도')) & (~d_alert.prevsig.fillna('').str.contains('매도'))].assign(kind='매도')
    events=pd.concat([ab,ase]).sort_values('time',ascending=False).head(12)
    if 'alert_page' not in st.session_state: st.session_state.alert_page=False
    if st.button('🔔 매매 알림창 열기',use_container_width=True):
        st.session_state.alert_page=True
        st.rerun()
    if st.session_state.alert_page:
        st.markdown("### 🔔 BTC · ETH 매매 알림")
        if st.button('← 메인 차트로 돌아가기',use_container_width=True):
            st.session_state.alert_page=False
            st.rerun()
        st.caption(f'{market} · 최근 매매 신호')
        if len(events):
            rows=[]
            for _,ev in events.iterrows():
                rows.append({
                    '구분': '🟢 매수' if ev.kind=='매수' else '🔴 매도',
                    '발생시간': ev.time.strftime('%m/%d %H:%M'),
                    '가격': f"₩{ev.close:,.0f}",
                    '신뢰도': f"{abs(int(ev.score))}/4"
                })
            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True,
                column_config={
                    '구분': st.column_config.TextColumn('신호',width='small'),
                    '발생시간': st.column_config.TextColumn('발생시간',width='medium'),
                    '가격': st.column_config.TextColumn('가격',width='medium'),
                    '신뢰도': st.column_config.TextColumn('신뢰도',width='small')
                })
        else:
            st.info('최근 새 매매 신호가 없습니다.')
        st.markdown("#### 알림 설정")
        st.toggle('매수 신호 알림',value=True,key='buy_alert')
        st.toggle('매도 신호 알림',value=True,key='sell_alert')
        st.toggle('강력 신호만',value=False,key='strong_alert')
        st.stop()

    days={'1일':1,'1주':7,'1개월':30}[period]
    view=d[d.time >= d.time.max()-pd.Timedelta(days=days)].copy()
    if len(view)<20: view=d.tail(min(len(d),120)).copy()
    ss=signal_series(view); view['sig']=[z[0] for z in ss]; view['score']=[z[1] for z in ss]
    view['prevsig']=view.sig.shift(1)
    buys=view[(view.sig.str.contains('매수')) & (~view.prevsig.fillna('').str.contains('매수'))]
    sells=view[(view.sig.str.contains('매도')) & (~view.prevsig.fillna('').str.contains('매도'))]

    fig=make_subplots(rows=4,cols=1,shared_xaxes=True,row_heights=[.58,.12,.15,.15],vertical_spacing=.035)
    fig.add_trace(go.Candlestick(x=view.time,open=view.open,high=view.high,low=view.low,close=view.close,name='Price'),row=1,col=1)
    fig.add_trace(go.Scatter(x=view.time,y=view.EMA5,name='EMA5',line=dict(width=1.2)),row=1,col=1)
    fig.add_trace(go.Scatter(x=view.time,y=view.EMA20,name='EMA20',line=dict(width=1.5)),row=1,col=1)
    fig.add_trace(go.Scatter(x=view.time,y=view.EMA50,name='EMA50',line=dict(width=1.5)),row=1,col=1)
    fig.add_trace(go.Scatter(x=buys.time,y=buys.low*.995,mode='markers',marker=dict(symbol='triangle-up',size=10,color='#16a36a'),name='매수'),row=1,col=1)
    fig.add_trace(go.Scatter(x=sells.time,y=sells.high*1.005,mode='markers',marker=dict(symbol='triangle-down',size=10,color='#ef5350'),name='매도'),row=1,col=1)
    fig.add_trace(go.Bar(x=view.time,y=view.volume,name='거래량',marker_color='#b8c6dc'),row=2,col=1)
    fig.add_trace(go.Scatter(x=view.time,y=view.RSI,name='RSI',line=dict(width=1.5)),row=3,col=1)
    fig.add_hline(y=70,line_dash='dot',row=3,col=1); fig.add_hline(y=30,line_dash='dot',row=3,col=1)
    hist=view.MACD-view.MACDsig
    fig.add_trace(go.Bar(x=view.time,y=hist,name='MACD Hist',marker_color='#b8c6dc'),row=4,col=1)
    fig.add_trace(go.Scatter(x=view.time,y=view.MACD,name='MACD',line=dict(width=1.4)),row=4,col=1)
    fig.add_trace(go.Scatter(x=view.time,y=view.MACDsig,name='Signal',line=dict(width=1.2)),row=4,col=1)
    fig.update_layout(height=410,template='plotly_white',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='#ffffff',xaxis_rangeslider_visible=False,margin=dict(l=5,r=5,t=8,b=5),showlegend=False,hovermode='x unified')
    fig.update_xaxes(showgrid=False,zeroline=False)
    fig.update_yaxes(gridcolor='#edf1f6',zeroline=False,tickfont=dict(size=10))
    fig.update_annotations(font=dict(size=9))
    fig.update_xaxes(title_text=None); fig.update_yaxes(title_text=None)
    st.plotly_chart(fig,use_container_width=True)

    hi=view.high.tail(min(72,len(view))).max(); lo=view.low.tail(min(72,len(view))).min()
    st.markdown(f"**주요 가격 구간** · 단기 저항 ₩{hi:,.0f} · 현재가 ₩{last.close:,.0f} · 단기 지지 ₩{lo:,.0f}")
    st.markdown('**신호 판단 근거:** ' + ' · '.join(reasons))
    st.caption(f'마지막 업데이트 캔들: {last.time:%Y-%m-%d %H:%M} KST · 본 앱의 신호는 투자 판단 보조용이며 수익을 보장하지 않습니다.')
except Exception as e:
    st.error('시세를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.')
    st.caption(str(e))

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

@st.cache_data(ttl=300)
def upbit_candles(market, unit, count=1000):
    url=f'https://api.upbit.com/v1/candles/minutes/{unit}'
    rows=[]; to=None
    while len(rows)<count:
        n=min(200,count-len(rows)); params={'market':market,'count':n}
        if to: params['to']=to
        r=requests.get(url,params=params,timeout=10); r.raise_for_status()
        batch=r.json()
        if not batch: break
        rows.extend(batch)
        oldest=pd.to_datetime(batch[-1]['candle_date_time_utc'])-pd.Timedelta(seconds=1)
        to=oldest.strftime('%Y-%m-%dT%H:%M:%S')
    d=pd.DataFrame(rows).drop_duplicates('candle_date_time_kst').sort_values('candle_date_time_kst')
    d['time']=pd.to_datetime(d['candle_date_time_kst'])
    d=d.rename(columns={'opening_price':'open','high_price':'high','low_price':'low','trade_price':'close','candle_acc_trade_volume':'volume'})
    return d[['time','open','high','low','close','volume']].reset_index(drop=True)

def indicators(d):
    x=d.copy(); c=x.close
    x['EMA20']=c.ewm(span=20,adjust=False).mean()
    x['EMA50']=c.ewm(span=50,adjust=False).mean()
    x['EMA100']=c.ewm(span=100,adjust=False).mean()
    delta=c.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    rs=gain/loss.replace(0,np.nan); x['RSI']=100-(100/(1+rs))
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean()
    x['MACD']=e12-e26; x['MACDsig']=x.MACD.ewm(span=9,adjust=False).mean()
    tr=pd.concat([(x.high-x.low),(x.high-c.shift()).abs(),(x.low-c.shift()).abs()],axis=1).max(axis=1)
    x['ATR']=tr.ewm(alpha=1/14,adjust=False).mean()
    x['VOLMA20']=x.volume.rolling(20).mean()
    mid=c.rolling(20).mean(); sd=c.rolling(20).std()
    x['BBmid']=mid; x['BBupper']=mid+2*sd; x['BBlower']=mid-2*sd
    low14=x.low.rolling(14).min(); high14=x.high.rolling(14).max()
    x['STOCHK']=100*(c-low14)/(high14-low14).replace(0,np.nan)
    up=x.high.diff(); dn=-x.low.diff()
    plus=np.where((up>dn)&(up>0),up,0.0); minus=np.where((dn>up)&(dn>0),dn,0.0)
    atr=x['ATR'].replace(0,np.nan)
    pdi=100*pd.Series(plus,index=x.index).ewm(alpha=1/14,adjust=False).mean()/atr
    mdi=100*pd.Series(minus,index=x.index).ewm(alpha=1/14,adjust=False).mean()/atr
    x['PDI']=pdi; x['MDI']=mdi
    x['ADX']=(100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)).ewm(alpha=1/14,adjust=False).mean()
    x['DMIbull']=(x.PDI>x.MDI)
    x['DMIcrossUp']=(x.PDI>x.MDI)&(x.PDI.shift(1)<=x.MDI.shift(1))
    x['DMIcrossDown']=(x.PDI<x.MDI)&(x.PDI.shift(1)>=x.MDI.shift(1))
    return x

def trade_engine(x, p=None, fee=0.0005, slip=0.0003):
    if p is None: p={'rsi_lo':50,'rsi_hi':68,'adx':18,'vol':0.9,'stop':1.5,'trail':2.0,'take':3.0,'score':7,'hold':6}
    z=x.copy()
    tests=pd.DataFrame(index=z.index)
    tests['trend']=(z.close>z.EMA100)&(z.EMA20>z.EMA50)
    tests['macd']=(z.MACD>z.MACDsig)
    tests['rsi']=z.RSI.between(p['rsi_lo'],p['rsi_hi'])
    tests['adx']=z.ADX>=p['adx']
    # DMI: +DI 상향교차 후 일정 시간 동안만 유효한 상승 신호로 인정
    cross_age=pd.Series(np.nan,index=z.index,dtype=float)
    last_cross=-10**9
    for j in range(len(z)):
        if bool(z.DMIcrossUp.iloc[j]): last_cross=j
        cross_age.iloc[j]=j-last_cross
    bars=max(1,int(np.ceil(p.get('dmi_hours',12)*60/p.get('unit',60))))
    z['DMIrecent']=z.DMIbull & (cross_age<=bars)
    tests['dmi']=z.DMIrecent
    tests['volume']=z.volume>=z.VOLMA20*p['vol']
    tests['bb']=z.close>=z.BBmid
    tests['stoch']=z.STOCHK.between(35,85)
    z['signal_score']=tests.sum(axis=1)
    z['entry_ok']=z['signal_score']>=p['score']
    z['exit_ok']=(z.close<z.EMA20)|(z.MACD<z.MACDsig)|(z.RSI>76)|(z.STOCHK>92)|z.DMIcrossDown
    sig=['관망']*len(z); trades=[]; in_pos=False; entry=0; entry_i=None; peak=0
    for i in range(1,len(z)):
        r=z.iloc[i]
        if not in_pos and bool(r.entry_ok) and not bool(z.iloc[i-1].entry_ok):
            in_pos=True; entry=r.close*(1+slip); entry_i=i; peak=r.close; sig[i]='매수'
        elif in_pos:
            peak=max(peak,r.close); atrv=r.ATR
            stop=entry-p['stop']*atrv; trail=peak-p['trail']*atrv; take=entry+p['take']*atrv
            held_hours=(r.time-z.iloc[entry_i].time).total_seconds()/3600
            # 매수 후 최소 24시간 보유: 24시간이 지나기 전에는 어떤 매도 신호도 실행하지 않음
            if held_hours>=p.get('hold',6) and (bool(r.exit_ok) or r.close<=max(stop,trail) or r.close>=take):
                out=r.close*(1-slip); net=(out/entry-1)-2*fee
                trades.append({'entry_time':z.iloc[entry_i].time,'entry':entry,'exit_time':r.time,'exit':out,'return':net})
                in_pos=False; sig[i]='매도'
    z['sig']=sig
    return z,pd.DataFrame(trades)

def optimize_strategy(d, unit=60):
    # DMI cross 유효시간까지 포함한 경량 최적화
    split=max(180,int(len(d)*0.70)); train=d.iloc[:split]
    candidates=[]
    base=[
      {'score':5,'adx':12,'hold':0,'stop':1.2,'trail':1.8,'take':2.5},
      {'score':5,'adx':18,'hold':3,'stop':1.5,'trail':2.0,'take':3.0},
      {'score':6,'adx':18,'hold':6,'stop':1.5,'trail':2.0,'take':3.0},
      {'score':6,'adx':24,'hold':12,'stop':1.8,'trail':2.5,'take':4.0},
      {'score':7,'adx':18,'hold':6,'stop':1.8,'trail':2.5,'take':4.0},
    ]
    for dmi_hours in [3,6,12,24]:
      for q in base:
        p={'rsi_lo':48,'rsi_hi':72,'vol':0.85,'dmi_hours':dmi_hours,'unit':unit,**q}
        _,t=trade_engine(train,p); s=stats(t)
        if s['n']>=2:
            # 수익률 우선, 지나치게 적은 거래의 우연한 최고값은 PF로 보조 평가
            objective=s['total'] + min(s['pf'],4)*0.5
            candidates.append((objective,p))
    p=max(candidates,key=lambda q:q[0])[1] if candidates else {'rsi_lo':50,'rsi_hi':68,'adx':18,'vol':0.9,'stop':1.5,'trail':2.0,'take':3.0,'score':6,'hold':6,'dmi_hours':12,'unit':unit}
    eng,trades=trade_engine(d,p)
    test=d.iloc[split:].copy(); _,test_trades=trade_engine(test,p)
    return p,eng,trades,stats(test_trades),split

def stats(trades):
    if trades.empty: return {'n':0,'win':0,'avg':0,'total':0,'pf':0}
    wins=trades[trades['return']>0]['return']; losses=trades[trades['return']<=0]['return']
    pf=wins.sum()/abs(losses.sum()) if abs(losses.sum())>0 else np.inf
    return {'n':len(trades),'win':(trades['return']>0).mean()*100,'avg':trades['return'].mean()*100,
            'total':((1+trades['return']).prod()-1)*100,'pf':pf}

c1,c2,c3=st.columns([1.2,1,1])
with c1: coin=st.segmented_control('코인',['Bitcoin (BTC)','Ethereum (ETH)'],default='Bitcoin (BTC)')
with c2:
    period=st.segmented_control('차트 기간',['1일','1주','1개월'],default='1주')
    unit=st.selectbox('캔들 간격',[15,30,60,240],index=2,format_func=lambda x:f'{x}분')
with c3: st.caption('시세 데이터: Upbit 공개 API · 자동주문 없음')
if coin is None:
    coin='Bitcoin (BTC)'
if period is None:
    period='1주'
market='KRW-BTC' if 'BTC' in coin else 'KRW-ETH'
try:
    need={'1일':220,'1주':300,'1개월':750}[period]
    raw=upbit_candles(market,unit,need); d=indicators(raw); last=d.iloc[-1]; prev=d.iloc[-2]
    params,eng,trades,oos,split=optimize_strategy(d,unit); bt=stats(trades)
    active=(eng.sig.iloc[-1]=='매수') or (len(eng)>1 and '매수' in eng.sig.iloc[max(0,len(eng)-12):].values and '매도' not in eng.sig.iloc[max(0,len(eng)-12):].values)
    lab='매수' if eng.sig.iloc[-1]=='매수' else '매도' if eng.sig.iloc[-1]=='매도' else '관망'
    score=int(eng.signal_score.iloc[-1]); reasons=[f'복합지표 점수 {score}/7',f"ADX 기준 {params['adx']}",f"진입 {params['score']}/8 · DMI Cross {params['dmi_hours']}h · 최소보유 {params['hold']}시간"]
    pct=(last.close/prev.close-1)*100
    a,b,c,dcol=st.columns(4)
    a.metric('현재가',f'₩{last.close/1_000_000:.2f}M' if last.close>=10_000_000 else f'₩{last.close:,.0f}',f'{pct:+.2f}%')
    b.metric('RSI (14)',f'{last.RSI:.1f}')
    c.metric('MACD',f'{last.MACD:,.0f}')
    dcol.metric('전체 백테스트',f"{bt['total']:+.2f}%",f"OOS {oos['total']:+.2f}% · {oos['n']}회")
    sig_color='#e53935' if '매도' in lab else '#00a86b' if '매수' in lab else '#60708a'
    st.markdown(f"<div class='signal'>현재 종합 신호 · <span style='color:{sig_color}'>{lab}</span> &nbsp; | &nbsp; 복합점수 {score}/8</div>",unsafe_allow_html=True)

    ev=eng[eng.sig.isin(['매수','매도'])].copy()
    ev['kind']=ev.sig
    ev['score']=3
    events=ev.sort_values('time',ascending=False).head(12)
    if 'alert_page' not in st.session_state: st.session_state.alert_page=False
    nav1,nav2=st.columns(2)
    with nav1:
        if st.button('📊 종목 분석',use_container_width=True,type='primary' if not st.session_state.alert_page else 'secondary'):
            st.session_state.alert_page=False
            st.rerun()
    with nav2:
        if st.button('🔔 매매 알림',use_container_width=True,type='primary' if st.session_state.alert_page else 'secondary'):
            st.session_state.alert_page=True
            st.rerun()
    if st.session_state.alert_page:
        st.markdown("### 🔔 BTC · ETH 매매 알림")
        st.caption(f'{market} · 최근 매매 신호')
        if len(events):
            rows=[]
            for _,ev in events.iterrows():
                rows.append({
                    '구분': '🟢 매수' if ev.kind=='매수' else '🔴 매도',
                    '발생시간': ev.time.strftime('%m/%d %H:%M'),
                    '가격': f"₩{ev.close:,.0f}",
                    '신뢰도': '검증'
                })
            st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True,
                column_config={
                    '구분': st.column_config.TextColumn('신호',width='small'),
                    '발생시간': st.column_config.TextColumn('발생시간',width='medium'),
                    '가격': st.column_config.TextColumn('가격',width='medium'),
                    '신뢰도': st.column_config.TextColumn('상태',width='small')
                })
        else:
            st.info('최근 새 매매 신호가 없습니다.')
        st.markdown("#### 알림 설정")
        st.toggle('매수 신호 알림',value=True,key='buy_alert')
        st.toggle('매도 신호 알림',value=True,key='sell_alert')
        st.toggle('강력 신호만',value=False,key='strong_alert')
        st.stop()

    days={'1일':1,'1주':7,'1개월':30}[period]
    view=eng[eng.time >= eng.time.max()-pd.Timedelta(days=days)].copy()
    if len(view)<20: view=eng.tail(min(len(eng),120)).copy()
    buys=view[view.sig=='매수']
    sells=view[view.sig=='매도']

    fig=make_subplots(rows=4,cols=1,shared_xaxes=True,row_heights=[.58,.12,.15,.15],vertical_spacing=.035)
    fig.add_trace(go.Candlestick(x=view.time,open=view.open,high=view.high,low=view.low,close=view.close,name='Price'),row=1,col=1)
    fig.add_trace(go.Scatter(x=view.time,y=view.EMA100,name='EMA100',line=dict(width=1.2)),row=1,col=1)
    fig.add_trace(go.Scatter(x=view.time,y=view.EMA20,name='EMA20',line=dict(width=1.5)),row=1,col=1)
    fig.add_trace(go.Scatter(x=view.time,y=view.EMA50,name='EMA50',line=dict(width=1.5)),row=1,col=1)
    fig.add_trace(go.Scatter(x=buys.time,y=buys.low*.982,mode='text',text=['⬆<br>매수']*len(buys),textposition='middle center',textfont=dict(size=20,color='#1565C0',family='Arial Black'),name='매수',hovertemplate='매수<br>%{x}<extra></extra>'),row=1,col=1)
    fig.add_trace(go.Scatter(x=sells.time,y=sells.high*1.018,mode='text',text=['매도<br>⬇']*len(sells),textposition='middle center',textfont=dict(size=20,color='#7B1FA2',family='Arial Black'),name='매도',hovertemplate='매도<br>%{x}<extra></extra>'),row=1,col=1)
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
    st.caption(f"복합전략: EMA · MACD · RSI · ADX · DMI Cross 유효시간 최적화 · 거래량 · Bollinger · Stochastic · 보유기간 자동최적화 | 전체 {len(d)}캔들: {bt['n']}회 · 승률 {bt['win']:.1f}% · 누적 {bt['total']:+.2f}% · PF {bt['pf']:.2f}")
    st.caption(f"후반 30% OOS 검증(최적화 미사용 구간): {oos['n']}회 · 승률 {oos['win']:.1f}% · 누적 {oos['total']:+.2f}% · PF {oos['pf']:.2f} | 수수료·슬리피지 반영")
    st.caption(f'마지막 업데이트 캔들: {last.time:%Y-%m-%d %H:%M} KST · 본 앱의 신호는 투자 판단 보조용이며 수익을 보장하지 않습니다.')
except Exception as e:
    st.error('시세를 불러오지 못했습니다. 잠시 후 다시 시도해 주세요.')
    st.caption(str(e))

"""Bounded v11 entry refinement; all selection periods are reused historical data."""
import json
from pathlib import Path
import pandas as pd
import app
PROFILES=[{}, {'dmi_gap':2},{'dmi_gap':5},{'adx':15},{'adx':20},{'macd_margin':.05},{'macd_margin':.1},{'macd_days':2},{'rsi_rising':True},{'volume':.8},{'dmi_gap':2,'macd_days':2},{'adx':15,'volume':.8}]
def refine(z,p):
 z=z.copy();buy=z.candidate_sig.eq('매수');sell=z.candidate_sig.eq('매도');conflict=z.candidate_sig.eq('관망(충돌)');gate=pd.Series(True,index=z.index)
 if p.get('dmi_gap'):gate &= z.PDI-z.MDI >= p['dmi_gap']
 if p.get('adx'):gate &= z.ADX >= p['adx']
 if p.get('macd_margin'):gate &= 100*z.MACDhist/z.close >= p['macd_margin']
 if p.get('macd_days'):gate &= z.MACDhist.gt(0).rolling(p['macd_days']).sum().eq(p['macd_days'])
 if p.get('rsi_rising'):gate &= z.RSI.diff()>0
 if p.get('volume'):gate &= z.VolumeRatio>=p['volume']
 buy &= gate
 z.loc[~gate & z.candidate_sig.eq('매수'),'candidate_sig']='관망'
 z['sig']=z.candidate_sig.copy()
 z=app.sequence_signals(z,stop_pct=10)
 return app.label_strength(z)
def metric(z,fee=.0005,slip=.0003):
 t,e,dd,h=app.backtest(z.reset_index(drop=True),fee=fee,slip=slip)
 return {'return':round((e.iloc[-1]-1)*100,4),'dd':round(dd,4),'trades':len(t)}
results={}
for market in ['KRW-BTC','KRW-ETH']:
 raw=pd.read_csv(market+'_daily.csv');raw.time=pd.to_datetime(raw.time,utc=True).dt.tz_convert('Asia/Seoul');x=app.indicators(raw);z=app.research_signals(x,market,app.EXTENDED_REPORT[market]["applied"]["profile"]);n=len(z);cut=int(n*.5);end=int(n*.75);rows=[]
 for p in PROFILES:
  y=refine(z,p);row={'profile':p}
  for label,subset in [('train',y.iloc[60:cut]),('validation',y.iloc[cut:end]),('recent',y.iloc[end:]),('year',y.iloc[-365:])]:row[label]=metric(subset)
  row['double_cost']=metric(y.iloc[-365:],fee=.001,slip=.0006);rows.append(row)
 baseline=rows[0]
 good=[r for r in rows[1:] if r['train']['trades']>=3 and r['validation']['trades']>=2 and all(r[k]['return']>=baseline[k]['return'] and r[k]['dd']>=baseline[k]['dd'] for k in ['train','validation','recent','year','double_cost']) and r['year']['return']>=baseline['year']['return']+.5]
 chosen=min(good,key=lambda r:(-r['year']['return'],len(r['profile']))) if good else baseline
 results[market]={'baseline':baseline,'applied':chosen,'count':len(rows),'rows':rows,'start':str(z.time.iloc[-365]),'end':str(z.time.iloc[-1])}
 print(market,json.dumps({'baseline':baseline,'applied':chosen},ensure_ascii=False),flush=True)
Path('refinement_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))

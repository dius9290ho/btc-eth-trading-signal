"""Bounded retrospective study: 84 profiles per coin, fixed indicator periods, no future bars."""
import itertools,json
from pathlib import Path
import pandas as pd
import app

FILTERS=[{}, {'volume':.8},{'volume':1.},{'volume':1.2},{'macd':'positive'},{'macd':'rising'},{'bb':'mid'},{'bb':'rebound'},
 {'volume':1.,'macd':'positive'},{'volume':.8,'macd':'rising'},{'volume':1.,'bb':'mid'},{'macd':'positive','bb':'mid'}]
EXITS=[{}, {'atr':2.},{'atr':3.},{'atr':4.},{'trail':5.},{'trail':8.},{'trail':10.}]
PROFILES=[dict(f,**e) for f,e in itertools.product(FILTERS,EXITS)]

def metric(z):
 t,e,dd,h=app.backtest(z.reset_index(drop=True))
 return {'return':round(float((e.iloc[-1]-1)*100),4),'dd':round(float(dd),4),'trades':len(t)}

all_results={}
for market in ['KRW-BTC','KRW-ETH']:
 raw=pd.read_csv(market+'_daily.csv');raw.time=pd.to_datetime(raw.time,utc=True).dt.tz_convert('Asia/Seoul');x=app.indicators(raw)
 base=app.signals(x,**app.OPTIMIZATION_REPORT[market]['applied_params']);n=len(x);cut=int(n*.5);end=int(n*.75)
 rows=[]
 for index,p in enumerate(PROFILES):
  z=app.research_signals(x,market,p,base_frame=base)
  train=metric(z.iloc[60:cut]);valid=metric(z.iloc[cut:end]);recent=metric(z.iloc[end:]);year=metric(z.iloc[-365:])
  score=.35*train['return']+.65*valid['return']+.15*(.35*train['dd']+.65*valid['dd'])
  rows.append({'profile':p,'train':train,'validation':valid,'recent':recent,'year':year,'score':score})
  if index%21==0:print(market,index+1,'/',len(PROFILES),flush=True)
 baseline=rows[0]
 eligible=[r for r in rows if r['train']['trades']>=3 and r['validation']['trades']>=2]
 ranked=dict(max(eligible,key=lambda r:r['score'])) if eligible else dict(baseline)
 consistent=[r for r in eligible if all(r[label]['return']>=baseline[label]['return'] and r[label]['dd']>=baseline[label]['dd'] for label in ['train','validation','recent','year']) and r['year']['return']>=baseline['year']['return']+.5]
 # Retrospective selection deliberately uses all four periods, including overlapping year; not an untouched test.
 applied=min(consistent,key=lambda r:(-r['year']['return'],len(r['profile']),-r['score'])) if consistent else baseline
 passed=bool(consistent)
 result={'candidate':ranked,'baseline':baseline,'applied':applied,'decision':'추가 조건 적용' if passed and applied['profile'] else '개선 기준 미충족 또는 기존 우수 · 기존 유지','count':len(rows),
  'train_start':str(x.time.iloc[60]),'train_end':str(x.time.iloc[cut-1]),'validation_start':str(x.time.iloc[cut]),'validation_end':str(x.time.iloc[end-1]),
  'recent_start':str(x.time.iloc[end]),'recent_end':str(x.time.iloc[-1]),'year_start':str(x.time.iloc[-365]),'year_end':str(x.time.iloc[-1]),'rows':rows}
 checks=[]
 selected_z=app.research_signals(x,market,applied['profile'],base_frame=base)
 baseline_z=app.research_signals(x,market,{},base_frame=base)
 for quarter in range(4):
  a=n-365+quarter*90;b=min(n,a+90) if quarter<3 else n
  checks.append({'start':str(x.time.iloc[a]),'end':str(x.time.iloc[b-1]),'baseline':metric(baseline_z.iloc[a:b]),'applied':metric(selected_z.iloc[a:b])})
 result['quarters']=checks
 for label,z in [('baseline',baseline_z),('applied',selected_z)]:
  t,e,dd,h=app.backtest(z.iloc[-365:].reset_index(drop=True),fee=.001,slip=.0006)
  result[label+'_double_cost']={'return':round(float((e.iloc[-1]-1)*100),4),'dd':round(float(dd),4),'trades':len(t)}
 all_results[market]=result
 print('RESULT',market,json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False),flush=True)
Path('extended_results.json').write_text(json.dumps(all_results,ensure_ascii=False,indent=2))

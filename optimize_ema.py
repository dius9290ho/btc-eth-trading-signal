"""One additional indicator only: EMA. Archived data has been used in v5; not an untouched test."""
import json
from pathlib import Path
import pandas as pd
import app

def metric(z):
    t,e,dd,h=app.backtest(z.reset_index(drop=True))
    return dict(ret=round(float((e.iloc[-1]-1)*100),4),dd=round(float(dd),4),trades=len(t))

results={}
for market in ['KRW-BTC','KRW-ETH']:
    raw=pd.read_csv(market+'_daily.csv');raw.time=pd.to_datetime(raw.time,utc=True).dt.tz_convert('Asia/Seoul');x=app.indicators(raw)
    n=len(x);cut=int(n*.5);end=int(n*.75);base={'window':30 if market=='KRW-BTC' else 14,'min_price':1.,'min_rsi':3.,'div_confirm':'none','min_adx':0,'exit_mode':'dmi_early' if market=='KRW-BTC' else 'confirmed'};rows=[]
    for period in [20,50,100,200]:
        for mode in ['price','slope']:
            for exit_ in [False,True]:
                p=dict(base,ema_period=period,ema_mode=mode,ema_exit=exit_)
                z=app.signals(x.iloc[:end],**p)
                train=metric(z.iloc[200:cut]);valid=metric(z.iloc[cut:end])
                # Validation is primary; training return used only as a tie breaker.
                score=valid['ret']+.1*valid['dd']+.01*train['ret']
                rows.append(dict(params=p,train=train,validation=valid,score=score))
    eligible=[r for r in rows if r['train']['trades']>=2 and r['validation']['trades']>=2]
    candidate=dict(max(eligible,key=lambda r:r['score']))
    z=app.signals(x,**candidate['params']);b=app.signals(x,**base)
    candidate['recent']=metric(z.iloc[end:]);candidate['previous']=metric(b.iloc[end:]);candidate['previous_validation']=metric(b.iloc[cut:end]);candidate['previous_params']=base
    passed=candidate['recent']['ret']>=candidate['previous']['ret'] and candidate['recent']['dd']>=candidate['previous']['dd'] and candidate['validation']['ret']>=candidate['previous_validation']['ret']
    candidate['applied_params']=candidate['params'] if passed else base
    candidate['applied']=candidate['recent'] if passed else candidate['previous']
    candidate['decision']='EMA 추가 적용' if passed else '개선 검증 미충족 · 기존 유지'
    candidate['candidates']=len(rows);candidate['comparison_start']=str(x.time.iloc[end]);candidate['comparison_end']=str(x.time.iloc[-1]);candidate['selection_end']=str(x.time.iloc[end-1])
    candidate['rows']=rows
    results[market]=candidate
    print(market,json.dumps({k:v for k,v in candidate.items() if k!='rows'},ensure_ascii=False),flush=True)
Path('ema_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))

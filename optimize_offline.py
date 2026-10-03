import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import app


def fetch(market):
    path=Path(market+'_daily.csv')
    if path.exists():
        raw=pd.read_csv(path);raw.time=pd.to_datetime(raw.time,utc=True).dt.tz_convert('Asia/Seoul')
    else:
        raw=app.closed_candles(app.daily_candles(market,1000));raw.to_csv(path,index=False)
    return market,app.indicators(raw)


def metric(z):
    t,e,dd,hold=app.backtest(z.reset_index(drop=True))
    return {'return':round((e.iloc[-1]-1)*100,4),'dd':round(dd,4),'trades':len(t)}


all_results={}
with ThreadPoolExecutor(max_workers=2) as pool:
    datasets=dict(pool.map(fetch,['KRW-BTC','KRW-ETH']))
for market,x in datasets.items():
    n=len(x);cut=int(n*.5);end=int(n*.75)
    # Only the first 75% is visible when choosing parameters.
    tuning=x.iloc[:end].copy()
    base={'window':14,'min_price':1.,'min_rsi':3.,'div_confirm':'none','min_adx':0,'exit_mode':'confirmed'}
    candidates=[base]
    for w in [10,14,21,30]:
        for confirmation in ['none','stoch_direction']:
            for adx in [0,15]:
                for early in ['confirmed','dmi_early']:
                    p={**base,'window':w,'div_confirm':confirmation,'min_adx':adx,'exit_mode':early}
                    if p not in candidates:candidates.append(p)
    rows=[]
    for p in candidates:
        z=app.signals(tuning,**p)
        train=metric(z.iloc[60:cut]);valid=metric(z.iloc[cut:end])
        # Profit objective with mild drawdown cost, and a penalty for unstable folds.
        score=0.35*train['return']+0.65*valid['return']+0.1*(0.35*train['dd']+0.65*valid['dd'])
        if train['trades']<3 or valid['trades']<2: score-=10000
        rows.append({'params':p,'train':train,'validation':valid,'score':score})
    eligible=[r for r in rows if r['score']>-10000]
    selected=max(eligible,key=lambda r:r['score']) if eligible else rows[0]
    z=app.signals(x,**selected['params']);baseline=app.signals(x,**base)
    test=metric(z.iloc[end:]);original=metric(baseline.iloc[end:])
    chosen=selected['params']
    result={'params':chosen,'train':selected['train'],'validation':selected['validation'],'holdout':test,'baseline_holdout':original,
        'full':metric(z.iloc[60:]),'baseline_full':metric(baseline.iloc[60:]),'candidates':len(rows),
        'train_start':str(x.time.iloc[60]),'train_end':str(x.time.iloc[cut-1]),
        'validation_start':str(x.time.iloc[cut]),'validation_end':str(x.time.iloc[end-1]),
        'holdout_start':str(x.time.iloc[end]),'holdout_end':str(x.time.iloc[-1]),'fit_end':str(x.time.iloc[end-1])}
    passed=test['return']>=original['return'] and test['dd']>=original['dd']
    result['applied_params']=chosen if passed else base
    result['applied_holdout']=test if passed else original
    result['decision']='비교 후 적용' if passed else '별도 평가에서 악화되어 기존 유지'
    result['baseline_validation']=metric(baseline.iloc[cut:end])
    holdout=x.iloc[end:]
    result['buyhold_holdout']=round((holdout.close.iloc[-1]*(1-.0003)*(1-.0005)/(holdout.open.iloc[0]*(1+.0003)*(1+.0005))-1)*100,4)
    all_results[market]=result
    print(market,json.dumps(result,ensure_ascii=False),flush=True)
Path('optimization_results.json').write_text(json.dumps(all_results,ensure_ascii=False,indent=2))

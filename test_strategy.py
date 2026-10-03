import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import app


def frame(prices):
    close=np.array(prices,dtype=float)
    return pd.DataFrame({'time':pd.date_range('2024-01-01 09:00',periods=len(close),tz='Asia/Seoul'),'open':close,'high':close+2,'low':close-2,'close':close,'volume':100.0})


class StrategyTests(unittest.TestCase):
    def test_v13_restores_previous_coin_rules(self):
        rng=np.random.default_rng(51)
        x=app.indicators(frame(100+np.cumsum(rng.normal(size=230))))
        for market in ['KRW-BTC','KRW-ETH']:
            profile=dict(app.EXTENDED_REPORT[market]['applied']['profile'])
            profile.update(app.REFINEMENT_REPORT[market]['applied']['profile'])
            expected=app.research_signals(x,market,profile) if profile else app.signals(x,stop_pct=10,**app.OPTIMIZATION_REPORT[market]['applied_params'])
            actual=app.active_signals(x,market,strategy='v13')
            pd.testing.assert_frame_equal(actual,expected)
            pd.testing.assert_frame_equal(actual.iloc[:150],app.active_signals(x.iloc[:150],market,strategy='v13'))
        with self.assertRaises(ValueError):
            app.active_signals(x,'KRW-BTC',strategy='wrong')

    def test_active_dmi_warning_once_reset_and_no_buy_upgrade(self):
        x=frame([100]*12)
        x['PDI']=np.array([-2,2,3,-1,-3,-4,-5,2,3,-1,-4,-5])+10
        x['MDI']=10.
        x['ADX']=[10,10,11,18,21,23,24,24,26,26,28,29]
        z=app.active_signals(x,'KRW-BTC')
        self.assertEqual(z[z.alert_event].index.tolist(),[1,3,4,7,9,10])
        self.assertEqual(z[z.upgrade_warning].index.tolist(),[4,10])
        self.assertEqual(z.sig.iloc[4],'관망')
        self.assertEqual(z.signal_state.iloc[4],'매도 이후')
        self.assertFalse(z.alert_event.iloc[8])
        self.assertEqual(z[z.sig.isin(['매수','매도'])].sig.tolist(),['매수','매도','매수','매도'])
        pd.testing.assert_frame_equal(z,app.active_signals(x,'KRW-ETH'))
        for n in [4,5,8,11]:
            pd.testing.assert_frame_equal(z.iloc[:n],app.active_signals(x.iloc[:n],'KRW-BTC'))
        # Strong on the original sell means no later duplicate warning.
        x.loc[3,'ADX']=20
        out=app.active_signals(x,'KRW-BTC')
        self.assertEqual(out.signal_label.iloc[3],'강력매도')
        self.assertFalse(out.upgrade_warning.iloc[4:7].any())

    def test_active_stop_and_strength_do_not_change_trade_events(self):
        x=frame([100,100,100,85,80,79,90])
        x['PDI']=[9,12,13,14,8,7,12];x['MDI']=10.
        x['ADX']=[10,11,12,18,21,23,24]
        z=app.active_signals(x,'KRW-ETH')
        self.assertEqual(z.signal_label.iloc[3],'손절 매도')
        self.assertTrue(z.stop_trigger.iloc[3])
        self.assertTrue(z.upgrade_warning.iloc[4])
        base=app.dmi_only_signals(x,stop_pct=10)
        pd.testing.assert_series_equal(z.sig,base.sig)
        for key in ['RSI','SlowK','SlowD','MACDhist']:
            x[key]=-999.
        pd.testing.assert_series_equal(z.sig,app.active_signals(x,'KRW-ETH').sig)

    def test_dmi_only_crosses_and_no_stop_or_other_filter(self):
        x=frame([100,100,100,100,50,60,70])
        x['PDI']=[np.nan,10,12,13,14,9,12]
        x['MDI']=[np.nan,12,10,10,10,11,10]
        x['ADX']=0.
        z=app.dmi_only_signals(x)
        self.assertEqual(z.sig.tolist(),['관망','관망','매수','관망','관망','매도','매수'])
        self.assertFalse(z.stop_trigger.any())
        stopped=app.dmi_only_signals(x,stop_pct=10)
        self.assertEqual(stopped.sig.iloc[4],'매도')
        self.assertTrue(stopped.stop_trigger.iloc[4])
        self.assertEqual(stopped.sig.iloc[5],'관망')
        self.assertEqual(stopped.sig.iloc[6],'매수')
        for n in [3,5,6]:
            pd.testing.assert_frame_equal(z.iloc[:n],app.dmi_only_signals(x.iloc[:n]))

    def test_refined_dmi_buy_gate_boundary_exit_and_causality(self):
        raw=frame([100,101,102,101,100,103,104])
        x=app.indicators(raw)
        x['PDI']=[10,11.9,12,10,10,12,10];x['MDI']=10.;x['ADX']=25.
        x['RSI']=50.;x['RSIsignal']=50.;x['SlowK']=50.;x['SlowD']=40.
        x['buy_reason']=['','buy','buy','','','buy',''];x['sell_reason']=['','','','sell','','','sell']
        x['candidate_sig']=['관망','매수','매수','매도','관망','매수','매도']
        x['bull_state']=False;x['bear_state']=False
        out=app.research_signals(x,'KRW-ETH',{'dmi_gap':2},base_frame=x)
        self.assertEqual(out.sig.iloc[1],'관망')
        self.assertEqual(out.sig.iloc[2],'매수')
        self.assertEqual(out.sig.iloc[3],'매도')
        self.assertIn('2포인트',out.reason.iloc[2])
        for n in [3,4,6]:
            pd.testing.assert_frame_equal(out.iloc[:n],app.research_signals(x.iloc[:n],'KRW-ETH',{'dmi_gap':2},base_frame=x.iloc[:n]))

    def test_wilder_seed_and_rsi_edges(self):
        self.assertTrue(app.indicators(frame(range(1,60))).RSI.iloc[:14].isna().all())
        self.assertEqual(app.indicators(frame(range(1,60))).RSI.iloc[-1],100)
        self.assertEqual(app.indicators(frame(range(60,1,-1))).RSI.iloc[-1],0)
        self.assertEqual(app.indicators(frame([10]*60)).RSI.iloc[-1],50)
        self.assertAlmostEqual(app.wilder(pd.Series([np.nan,1.,2.,3.,4.]),3).iloc[-1],8/3)

    def test_only_completed_candles(self):
        raw=frame([10,11,12])
        self.assertEqual(len(app.closed_candles(raw,pd.Timestamp('2024-01-03 00:00',tz='UTC'))),2)
        self.assertEqual(len(app.closed_candles(raw,pd.Timestamp('2024-01-02 23:59:59',tz='UTC'))),1)

    def test_stochastic_30_10_10(self):
        raw=frame(range(1,101));z=app.indicators(raw)
        fast=100*(raw.close-raw.low.rolling(30).min())/(raw.high.rolling(30).max()-raw.low.rolling(30).min())
        pd.testing.assert_series_equal(z.SlowK,fast.rolling(10).mean(),check_names=False)
        pd.testing.assert_series_equal(z.SlowD,fast.rolling(10).mean().rolling(10).mean(),check_names=False)
        self.assertEqual(z.SlowK.first_valid_index(),38)
        self.assertEqual(z.SlowD.first_valid_index(),47)
        flat=frame([10]*60);flat['high']=10.;flat['low']=10.
        self.assertEqual(app.indicators(flat).SlowD.iloc[-1],50)

    def test_trend_divergence_onset_persistence_and_both_directions(self):
        z=frame(range(100,130));z['RSI']=np.linspace(70,40,30)
        z['PDI']=20.;z['MDI']=10.;z['SlowK']=60.;z['SlowD']=50.
        out=app.signals(z)
        self.assertEqual(out.sig.iloc[13],'매도')
        self.assertEqual(out.sig.iloc[14],'관망')
        self.assertTrue(out.bear_state.iloc[-1]);self.assertEqual(out.bear_div.sum(),1)
        z.close=z.close.iloc[::-1].to_numpy();z.RSI=z.RSI.iloc[::-1].to_numpy()
        out=app.signals(z)
        self.assertEqual(out.sig.iloc[13],'매수');self.assertTrue(out.bull_state.iloc[-1])
        # An opposing DMI entry on divergence onset must yield a conflict.
        z.loc[12,'PDI']=5.;z.loc[13,'PDI']=20.
        z.close=z.close.iloc[::-1].to_numpy();z.RSI=z.RSI.iloc[::-1].to_numpy()
        self.assertEqual(app.signals(z).sig.iloc[13],'관망(충돌)')

    def test_cross_confirmation_and_no_lookahead(self):
        z=frame([10]*5);z['RSI']=50.;z['PDI']=[10,20,20,10,20];z['MDI']=15.
        z['SlowK']=[40,60,60,40,40];z['SlowD']=50.
        out=app.signals(z)
        self.assertEqual(out.sig.tolist(),['관망','매수','관망','매도','관망'])
        rng=np.random.default_rng(9)
        raw=frame(100+np.cumsum(rng.normal(size=200)))
        full=app.signals(app.indicators(raw))
        for n in [32,70,110,190]:
            partial=app.signals(app.indicators(raw.iloc[:n]))
            pd.testing.assert_frame_equal(full.iloc[:n].reset_index(drop=True),partial.reset_index(drop=True))

    def test_alternating_signals_and_price_only_chart(self):
        z=frame([100]*12);z['RSI']=50.;z['PDI']=[20,20,20,20,10,10,10,10,20,20,20,20];z['MDI']=15.
        z['SlowK']=[40,60,40,60,40,60,40,40,60,40,60,60];z['SlowD']=50.
        out=app.signals(z)
        emitted=out[out.sig.isin(['매수','매도'])].sig.tolist()
        self.assertEqual(emitted,['매수','매도','매수'])
        self.assertEqual(out.candidate_sig.iloc[3],'매수');self.assertEqual(out.sig.iloc[3],'관망')
        self.assertEqual(out.candidate_sig.iloc[6],'매도');self.assertEqual(out.sig.iloc[6],'관망')
        self.assertEqual(out.signal_state.iloc[7],'매도 이후')
        fig=app.chart(out,90)
        self.assertEqual(len(fig.data),1);self.assertEqual(fig.data[0].type,'candlestick')
        self.assertEqual(len(fig.layout.annotations),3)
        self.assertGreaterEqual(fig.layout.height,360)
        self.assertLessEqual(fig.layout.height,450)

    def test_optimized_paths_are_causal_and_alternate(self):
        rng=np.random.default_rng(8)
        raw=frame(100+np.cumsum(rng.normal(size=220)))
        x=app.indicators(raw)
        for market in ['KRW-BTC','KRW-ETH']:
            full=app.active_signals(x,market)
            partial=app.active_signals(app.indicators(raw.iloc[:150]),market)
            pd.testing.assert_frame_equal(full.iloc[:150].reset_index(drop=True),partial.reset_index(drop=True))
            emitted=full[full.sig.isin(['매수','매도'])].sig.tolist()
            self.assertTrue(all(a!=b for a,b in zip(emitted,emitted[1:])))
        self.assertEqual(app.OPTIMIZATION_REPORT['KRW-ETH']['applied_params']['window'],14)
        self.assertEqual(app.OPTIMIZATION_REPORT['KRW-BTC']['applied_params']['exit_mode'],'dmi_early')

    def test_ema_filter_exit_and_warmup(self):
        z=frame([100,100,90,85,110,95,90])
        z['RSI']=50.;z['PDI']=[10,10,20,10,20,20,20];z['MDI']=15.
        z['SlowK']=[40,40,60,40,60,60,60];z['SlowD']=50.
        out=app.signals(z,ema_period=3,ema_exit=True)
        # Falling below EMA blocks otherwise valid DMI buys, without blocking sells.
        self.assertEqual(out.sig.iloc[2],'관망')
        self.assertEqual(out.sig.iloc[3],'매도')
        self.assertEqual(out.sig.iloc[4],'매수')
        self.assertEqual(out.sig.iloc[5],'매도')
        self.assertIn('EMA(3)',out.reason.iloc[5])
        self.assertTrue(out.EMA.iloc[:2].isna().all())
        pd.testing.assert_series_equal(out.EMA,z.close.ewm(span=3,adjust=False,min_periods=3).mean(),check_names=False)
        self.assertEqual(app.OPTIMIZATION_REPORT['KRW-ETH']['applied_params']['ema_period'],50)
        self.assertEqual(app.OPTIMIZATION_REPORT['KRW-BTC']['applied_params'].get('ema_period',0),0)

    def test_daily_close_stop_priority_reset_and_execution(self):
        z=frame([100,90.027,90,110,110])
        z['open']=[100,100,70,90,110]
        z['low']=[80,80,60,80,100]
        z['candidate_sig']=['매수','관망(충돌)','매수','관망','관망']
        z['sig']=z.candidate_sig;z['reason']=''
        out=app.sequence_signals(z,stop_pct=10)
        self.assertEqual(out.sig.tolist(),['매수','매도','매수','관망','관망'])
        self.assertTrue(out.stop_trigger.iloc[1])
        self.assertIn('10% 손절',out.reason.iloc[1])
        self.assertAlmostEqual(out.stop_price.iloc[1],100*(1+.0003)*.9)
        trades,eq,dd,holding=app.backtest(out)
        self.assertEqual(len(trades),1)
        self.assertLess(trades.iloc[0]['수익률(%)'],-29)
        for n in [2,3,4]:
            pd.testing.assert_frame_equal(out.iloc[:n],app.sequence_signals(z.iloc[:n],stop_pct=10))
        # Low below stop alone does not trigger; only confirmed close does.
        z.loc[1,'close']=95
        self.assertFalse(app.sequence_signals(z,stop_pct=10).stop_trigger.iloc[1])
        x=app.indicators(frame([100]*60))
        for market in ['KRW-BTC','KRW-ETH']:
            out=app.active_signals(x,market)
            pd.testing.assert_series_equal(out.sig,app.dmi_only_signals(x,stop_pct=10).sig)

    def test_strength_labels_preserve_events_and_stop_priority(self):
        z=frame([100,110,115,90,85])
        z['RSI']=[45,50,55,40,35];z['RSIsignal']=45.
        z['PDI']=[20,20,20,10,10];z['MDI']=[10,10,10,20,20];z['ADX']=30.
        z['SlowK']=[40,60,65,30,20];z['SlowD']=40.
        z['bull_state']=False;z['bear_state']=False
        z['sig']=['관망','매수','관망','매도','매도'];z['reason']='조건'
        z['stop_trigger']=[False,False,False,False,True]
        out=app.label_strength(z)
        self.assertEqual(out.signal_label.tolist(),['관망','강력매수','관망','강력매도','손절 매도'])
        pd.testing.assert_series_equal(out.sig,z.sig)
        for n in [2,3,4]:
            pd.testing.assert_frame_equal(out.iloc[:n],app.label_strength(z.iloc[:n]))
        z.loc[1,'ADX']=19
        self.assertEqual(app.label_strength(z).signal_label.iloc[1],'매수')
        z.loc[3,'bull_state']=True
        self.assertEqual(app.label_strength(z).signal_label.iloc[3],'매도')

    def test_sell_upgrade_warns_once_and_resets_only_on_buy(self):
        z=frame([100]*9)
        z['RSI']=[60,55,50,45,40,50,45,40,35];z['RSIsignal']=60.
        z['PDI']=10.;z['MDI']=20.;z['ADX']=[10,10,30,30,30,30,10,30,30]
        z['SlowK']=[70,65,60,55,50,45,40,35,30];z['SlowD']=80.
        z['bull_state']=False;z['bear_state']=False;z['stop_trigger']=False
        z['sig']=['관망','매도','관망','관망','관망','매수','매도','관망','관망'];z['reason']='기존'
        out=app.label_strength(z)
        self.assertEqual(out[out.upgrade_warning].index.tolist(),[2,7])
        self.assertEqual(out.signal_label.iloc[2],'강력매도 · 추가 경고')
        self.assertTrue(out.alert_event.iloc[2])
        self.assertEqual(out.event_kind.iloc[2],'추가 매도 경고')
        pd.testing.assert_series_equal(out.sig,z.sig)
        for n in [2,3,6,8]:
            pd.testing.assert_frame_equal(out.iloc[:n],app.label_strength(z.iloc[:n]))
        z.loc[1,'ADX']=30
        self.assertFalse(app.label_strength(z).upgrade_warning.iloc[2])
        z.loc[2,'bull_state']=True
        self.assertFalse(app.label_strength(z).upgrade_warning.iloc[2])

    def test_research_indicators_and_profile_causality(self):
        raw=frame([100]*80);raw.loc[60,'volume']=300
        x=app.indicators(raw)
        self.assertEqual(x.VolumeRatio.iloc[60],3)
        self.assertEqual(x.MACDhist.iloc[-1],0)
        self.assertEqual(x.BBlower.iloc[-1],100)
        rng=np.random.default_rng(15)
        raw=frame(100+np.cumsum(rng.normal(size=220)))
        for market in ['KRW-BTC','KRW-ETH']:
            full=app.research_signals(app.indicators(raw),market,{'volume':.8,'macd':'rising','bb':'rebound','atr':3})
            partial=app.research_signals(app.indicators(raw.iloc[:150]),market,{'volume':.8,'macd':'rising','bb':'rebound','atr':3})
            pd.testing.assert_frame_equal(full.iloc[:150],partial)
            base=app.signals(app.indicators(raw),stop_pct=10,**app.OPTIMIZATION_REPORT[market]['applied_params'])
            pd.testing.assert_series_equal(base.sig,app.research_signals(app.indicators(raw),market,{}).sig)
            emitted=full[full.sig.isin(['매수','매도'])].sig.tolist()
            self.assertTrue(all(a!=b for a,b in zip(emitted,emitted[1:])))

    def test_atr_trailing_line_never_moves_down_and_close_only(self):
        z=frame([100,100,120,115,110]);z['open']=100.
        z['ATR']=[2,2,2,5,10];z['low']=70.
        z['candidate_sig']=['매수','관망','관망','관망','관망'];z['sig']=z.candidate_sig;z['reason']=''
        out=app.sequence_signals(z,stop_pct=10,atr_mult=3)
        self.assertEqual(out.sig.tolist(),['매수','관망','관망','관망','매도'])
        self.assertEqual(out.stop_price.iloc[2:].tolist(),[114.,114.,114.])
        self.assertIn('ATR(14)',out.reason.iloc[-1])
        for n in [2,3,4]:
            pd.testing.assert_frame_equal(out.iloc[:n],app.sequence_signals(z.iloc[:n],stop_pct=10,atr_mult=3))

    def test_next_open_costs_hold_and_pending_final(self):
        z=frame([100,110,120,130,140]);z['sig']=['매수','매도','관망','관망','매수']
        trades,eq,dd,holding=app.backtest(z,fee=.001,slip=.002)
        self.assertEqual(len(trades),1);self.assertFalse(holding)
        self.assertEqual(trades.iloc[0]['매수시각'],z.time.iloc[1])
        self.assertEqual(trades.iloc[0]['매도시각'],z.time.iloc[2])
        self.assertAlmostEqual(eq.iloc[-1],120*(1-.002)*(1-.001)/(110*(1+.002)*(1+.001)))

    def test_app_both_coins_all_periods_and_error(self):
        from streamlit.testing.v1 import AppTest
        rng=np.random.default_rng(10)
        raw=frame(1000000+np.cumsum(rng.normal(0,15000,400)))
        payload=[{'candle_date_time_utc':r.time.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%S'),'opening_price':r.open,'high_price':r.high,'low_price':r.low,'trade_price':r.close,'candle_acc_trade_volume':r.volume} for _,r in raw.iloc[::-1].iterrows()]
        class Response:
            status_code=200
            def raise_for_status(self): pass
            def json(self): return payload[:100]
        app.st.cache_data.clear()
        with patch('requests.get',return_value=Response()):
            at=AppTest.from_file('app.py',default_timeout=30).run()
            self.assertEqual(len(at.exception),0);self.assertEqual(len(at.error),0)
            self.assertEqual(at.radio[0].value,'비트코인 (BTC)')
            self.assertEqual(at.radio[1].value,'v14 · DMI·손절')
            for coin in ['비트코인 (BTC)','이더리움 (ETH)']:
                for period in ['1개월','3개월','6개월','1년','2년','3년']:
                    at.radio[0].set_value(coin);at.selectbox[0].set_value(period);at.run()
                    self.assertEqual(len(at.exception),0);self.assertEqual(len(at.error),0)
                for version in ['v13 · 복합지표','v14 · DMI·손절']:
                    at.radio[1].set_value(version);at.run()
                    self.assertEqual(len(at.exception),0);self.assertEqual(len(at.error),0)
                    self.assertIn(version.split(' · ')[0]+' 적용', ' '.join(c.value for c in at.caption))
                    self.assertNotIn('구분',at.dataframe[0].value.columns)
                    self.assertEqual(list(at.dataframe[0].value.columns),['신호 확정시각(KST)','신호','판단 종가(원)','판단 근거'])
        app.st.cache_data.clear()
        with patch('requests.get',side_effect=app.requests.ConnectionError('test')):
            at=AppTest.from_file('app.py').run()
            self.assertEqual(len(at.exception),0);self.assertEqual(len(at.error),1)


if __name__=='__main__':
    unittest.main()

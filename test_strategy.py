import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import app


def frame(prices):
    close=np.array(prices,dtype=float)
    return pd.DataFrame({'time':pd.date_range('2024-01-01 09:00',periods=len(close),tz='Asia/Seoul'),'open':close,'high':close+2,'low':close-2,'close':close,'volume':100.0})


class StrategyTests(unittest.TestCase):
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
        self.assertGreaterEqual(fig.layout.height,600)

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
            for coin in ['Bitcoin (BTC)','Ethereum (ETH)']:
                for period in ['1개월','3개월','6개월','1년','2년']:
                    at.selectbox[0].set_value(coin);at.selectbox[1].set_value(period);at.run()
                    self.assertEqual(len(at.exception),0);self.assertEqual(len(at.error),0)
        app.st.cache_data.clear()
        with patch('requests.get',side_effect=app.requests.ConnectionError('test')):
            at=AppTest.from_file('app.py').run()
            self.assertEqual(len(at.exception),0);self.assertEqual(len(at.error),1)


if __name__=='__main__':
    unittest.main()

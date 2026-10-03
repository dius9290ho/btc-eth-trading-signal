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

    def test_divergence_confirmation_and_conflict(self):
        z=frame([12,11,10,11,12,11,9,10,11])
        z['RSI']=[50,45,35,45,50,45,40,45,50]
        z['PDI']=10.;z['MDI']=20.;z['ADX']=10.
        out=app.signals(z)
        self.assertFalse(out.bull_div.iloc[6]);self.assertFalse(out.bull_div.iloc[7])
        self.assertTrue(out.bull_div.iloc[8]);self.assertEqual(out.sig.iloc[8],'매수')
        self.assertEqual(int(out.div_from.iloc[8]),2);self.assertEqual(int(out.div_to.iloc[8]),6)
        high=frame([8,9,10,9,8,9,11,10,9])
        high['RSI']=[50,55,65,55,50,55,60,55,50]
        high['PDI']=10.;high['MDI']=20.;high['ADX']=10.
        self.assertTrue(app.signals(high).bear_div.iloc[8])
        z.loc[7,'PDI']=30.;z.loc[8,'PDI']=5.
        conflict=app.signals(z)
        self.assertEqual(conflict.sig.iloc[8],'관망(충돌)')

    def test_cross_rsi_and_no_lookahead(self):
        z=frame([10]*4);z['RSI']=[25,35,75,65];z['PDI']=[10,20,20,10];z['MDI']=15;z['ADX']=10
        out=app.signals(z)
        self.assertEqual(out.sig.tolist(),['관망','매수','관망','매도'])
        rng=np.random.default_rng(9)
        raw=frame(100+np.cumsum(rng.normal(size=200)))
        full=app.signals(app.indicators(raw))
        for n in [32,70,110,190]:
            partial=app.signals(app.indicators(raw.iloc[:n]))
            pd.testing.assert_frame_equal(full.iloc[:n].reset_index(drop=True),partial.reset_index(drop=True))

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

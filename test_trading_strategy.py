import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from trading_strategy import MultiPairTradingStrategy, TradeDirection

class TestMultiPairTradingStrategy(unittest.TestCase):
    def setUp(self):
        self.config = {
            'symbols': ['EURUSD', 'GBPUSD'],
            'timeframes': ['M1', 'M5'],
            'risk_per_trade': 0.01,
            'max_trades_per_day': 10,
            'max_concurrent_trades': 5
        }
        self.strategy = MultiPairTradingStrategy(self.config)

    @patch('trading_strategy.MT5AdvancedInterface')
    @patch('trading_strategy.RiskManagement')
    def test_initialize_mt5(self, mock_risk, mock_mt5):
        mock_mt5.return_value.connect.return_value = True
        self.assertTrue(self.strategy.initialize_mt5())
        with open('logs/trading_strategy.log', 'r') as log_file:
            self.assertIn("MT5 initialized successfully", log_file.read())

    @patch('trading_strategy.MT5AdvancedInterface')
    @patch('trading_strategy.RiskManagement')
    def test_initialize_mt5_failure(self, mock_risk, mock_mt5):
        mock_mt5.return_value.connect.return_value = False
        self.assertFalse(self.strategy.initialize_mt5())
        with open('logs/trading_strategy.log', 'r') as log_file:
            self.assertIn("MT5 initialization failed", log_file.read())

    @patch('trading_strategy.MT5AdvancedInterface')
    @patch('trading_strategy.RiskManagement')
    def test_get_mt5_data(self, mock_risk, mock_mt5):
        mock_data = pd.DataFrame({
            'time': pd.date_range(start='2023-01-01', periods=100, freq='1min'),
            'open': [1.1 + i*0.0001 for i in range(100)],
            'high': [1.1 + i*0.0001 + 0.0002 for i in range(100)],
            'low': [1.1 + i*0.0001 - 0.0002 for i in range(100)],
            'close': [1.1 + i*0.0001 + 0.0001 for i in range(100)],
            'tick_volume': [1000 + i for i in range(100)]
        })
        mock_mt5.return_value.get_prices.return_value = mock_data
        result = self.strategy.get_mt5_data('EURUSD', 'M1', 100)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 100)
        with open('logs/trading_strategy.log', 'r') as log_file:
            self.assertIn("Fetched MT5 data for EURUSD", log_file.read())

    @patch('trading_strategy.MT5AdvancedInterface')
    @patch('trading_strategy.RiskManagement')
    @patch('trading_strategy.MLMI')
    @patch('trading_strategy.QuadraticRegression')
    @patch('trading_strategy.FairValueGap')
    def test_check_entry_conditions(self, mock_fvg, mock_qr, mock_mlmi, mock_risk, mock_mt5):
        mock_mt5.return_value.get_prices.return_value = MagicMock()
        mock_mlmi.return_value.calculate.return_value = {'cross_above_ma': True}
        mock_qr.return_value.calculate.return_value = {'is_bullish': True}
        mock_fvg.return_value.detect_touched_fvg.return_value = {'touched_bullish': True}
        
        entry_signal, direction = self.strategy.check_entry_conditions('EURUSD')
        self.assertTrue(entry_signal)
        self.assertEqual(direction, TradeDirection.LONG)
        with open('logs/trading_strategy.log', 'r') as log_file:
            self.assertIn("Checked entry conditions for EURUSD", log_file.read())

    @patch('trading_strategy.MT5AdvancedInterface')
    @patch('trading_strategy.RiskManagement')
    def test_enter_trade(self, mock_risk, mock_mt5):
        mock_mt5.return_value.place_market_order.return_value = 12345
        mock_risk.return_value.open_trade.return_value = MagicMock()
        self.strategy.enter_trade('EURUSD', TradeDirection.LONG)
        mock_mt5.return_value.place_market_order.assert_called_once()
        mock_risk.return_value.open_trade.assert_called_once()
        with open('logs/trading_strategy.log', 'r') as log_file:
            self.assertIn("Entered LONG trade for EURUSD", log_file.read())

    @patch('trading_strategy.MT5AdvancedInterface')
    @patch('trading_strategy.RiskManagement')
    def test_exit_trade(self, mock_risk, mock_mt5):
        mock_mt5.return_value.close_position.return_value = True
        mock_risk.return_value.close_trade.return_value = MagicMock()
        self.strategy.active_trades['EURUSD'] = MagicMock(order_id=12345)
        self.strategy.exit_trade('EURUSD')
        mock_mt5.return_value.close_position.assert_called_once_with(12345)
        mock_risk.return_value.close_trade.assert_called_once()
        self.assertNotIn('EURUSD', self.strategy.active_trades)
        with open('logs/trading_strategy.log', 'r') as log_file:
            self.assertIn("Exited trade for EURUSD", log_file.read())

    @patch('trading_strategy.MT5AdvancedInterface')
    @patch('trading_strategy.RiskManagement')
    def test_calculate_stop_loss(self, mock_risk, mock_mt5):
        mock_mt5.return_value.get_prices.return_value = pd.DataFrame({
            'high': [1.1, 1.2, 1.3],
            'low': [1.0, 1.1, 1.2],
            'close': [1.05, 1.15, 1.25]
        })
        stop_loss = self.strategy.calculate_stop_loss('EURUSD', TradeDirection.LONG, 1.25)
        self.assertLess(stop_loss, 1.25)
        with open('logs/trading_strategy.log', 'r') as log_file:
            self.assertIn("Calculated stop loss for EURUSD", log_file.read())

    @patch('trading_strategy.MT5AdvancedInterface')
    @patch('trading_strategy.RiskManagement')
    def test_calculate_take_profit(self, mock_risk, mock_mt5):
        mock_mt5.return_value.get_prices.return_value = pd.DataFrame({
            'high': [1.1, 1.2, 1.3],
            'low': [1.0, 1.1, 1.2],
            'close': [1.05, 1.15, 1.25]
        })
        take_profit = self.strategy.calculate_take_profit('EURUSD', TradeDirection.LONG, 1.25)
        self.assertGreater(take_profit, 1.25)
        with open('logs/trading_strategy.log', 'r') as log_file:
            self.assertIn("Calculated take profit for EURUSD", log_file.read())

    @patch('trading_strategy.MT5AdvancedInterface')
    @patch('trading_strategy.RiskManagement')
    @patch('trading_strategy.MLMI')
    @patch('trading_strategy.QuadraticRegression')
    @patch('trading_strategy.FairValueGap')
    def test_run(self, mock_fvg, mock_qr, mock_mlmi, mock_risk, mock_mt5):
        mock_mt5.return_value.connect.return_value = True
        mock_mt5.return_value.get_prices.return_value = MagicMock()
        mock_mlmi.return_value.calculate.return_value = {'cross_above_ma': True}
        mock_qr.return_value.calculate.return_value = {'is_bullish': True}
        mock_fvg.return_value.detect_touched_fvg.return_value = {'touched_bullish': True}
        mock_risk.return_value.open_trade.return_value = MagicMock()
        mock_mt5.return_value.place_market_order.return_value = 12345

        # Mock asyncio.sleep to avoid actual waiting
        with patch('asyncio.sleep', return_value=None):
            self.strategy.run()

        mock_mt5.return_value.get_prices.assert_called()
        mock_risk.return_value.open_trade.assert_called()
        mock_mt5.return_value.place_market_order.assert_called()
        with open('logs/trading_strategy.log', 'r') as log_file:
            self.assertIn("Starting trading strategy", log_file.read())

    def test_get_performance_metrics(self):
        mock_risk_manager = MagicMock()
        mock_risk_manager.get_performance_metrics.return_value = {
            'total_trades': 10,
            'win_rate': 0.6,
            'profit_factor': 1.5,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.1
        }
        self.strategy.risk_manager = mock_risk_manager
        metrics = self.strategy.get_performance_metrics()
        self.assertEqual(metrics['total_trades'], 10)
        self.assertEqual(metrics['win_rate'], 0.6)
        self.assertEqual(metrics['profit_factor'], 1.5)
        self.assertEqual(metrics['sharpe_ratio'], 1.2)
        self.assertEqual(metrics['max_drawdown'], 0.1)

    def tearDown(self):
        patch.stopall()

if __name__ == '__main__':
    unittest.main()

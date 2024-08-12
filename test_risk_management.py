import unittest
from unittest.mock import patch, MagicMock
from risk_management import RiskManagement, TradeConfig, TradeDirection, Trade

class TestRiskManagement(unittest.TestCase):
    def setUp(self):
        self.risk_config = {
            'total_capital': 100000,
            'risk_per_trade': 0.01,
            'max_trades_per_day': 10,
            'max_positions': 5,
            'max_daily_loss': 1000,
            'max_drawdown': 0.1,
            'position_size_atr_multiplier': 1.5
        }
        self.risk_manager = RiskManagement(self.risk_config)

    def test_calculate_position_size(self):
        position_size = self.risk_manager.calculate_position_size('EURUSD', 1.2000, 1.1950, 0.0010)
        self.assertGreater(position_size, 0)
        self.assertLess(position_size, self.risk_config['total_capital'] * self.risk_config['risk_per_trade'] / 0.005)
        with open('logs/risk_management.log', 'r') as log_file:
            self.assertIn("Calculated position size for EURUSD", log_file.read())

    def test_can_open_trade_success(self):
        self.assertTrue(self.risk_manager.can_open_trade())

    def test_can_open_trade_max_positions_reached(self):
        self.risk_manager.active_trades = {f'trade_{i}': MagicMock() for i in range(self.risk_config['max_positions'])}
        self.assertFalse(self.risk_manager.can_open_trade())

    def test_can_open_trade_max_daily_loss_reached(self):
        self.risk_manager.daily_pnl = -self.risk_config['max_daily_loss'] - 1
        self.assertFalse(self.risk_manager.can_open_trade())

    def test_open_trade_success(self):
        trade_config = TradeConfig('EURUSD', 1.2000, 1.1950, 1.2100, TradeDirection.LONG)
        trade = self.risk_manager.open_trade(trade_config, 0.0010)
        self.assertIsInstance(trade, Trade)
        self.assertEqual(trade.config.symbol, 'EURUSD')
        with open('logs/risk_management.log', 'r') as log_file:
            self.assertIn("Opened trade for EURUSD", log_file.read())

    def test_open_trade_failure(self):
        self.risk_manager.active_trades = {f'trade_{i}': MagicMock() for i in range(self.risk_config['max_positions'])}
        trade_config = TradeConfig('EURUSD', 1.2000, 1.1950, 1.2100, TradeDirection.LONG)
        trade = self.risk_manager.open_trade(trade_config, 0.0010)
        self.assertIsNone(trade)
        with open('logs/risk_management.log', 'r') as log_file:
            self.assertIn("Cannot open trade", log_file.read())

    def test_close_trade_success(self):
        trade_config = TradeConfig('EURUSD', 1.2000, 1.1950, 1.2100, TradeDirection.LONG)
        trade = Trade(trade_config, 0.1, 1000000, 12345)
        self.risk_manager.active_trades['EURUSD'] = trade
        closed_trade = self.risk_manager.close_trade('EURUSD', 1.2050, 12345)
        self.assertIsInstance(closed_trade, Trade)
        self.assertNotIn('EURUSD', self.risk_manager.active_trades)
        with open('logs/risk_management.log', 'r') as log_file:
            self.assertIn("Closed trade for EURUSD", log_file.read())

    def test_close_trade_not_found(self):
        closed_trade = self.risk_manager.close_trade('GBPUSD', 1.3000, 12345)
        self.assertIsNone(closed_trade)
        with open('logs/risk_management.log', 'r') as log_file:
            self.assertIn("No active trade found for GBPUSD", log_file.read())

    def test_update_trailing_stop(self):
        trade_config = TradeConfig('EURUSD', 1.2000, 1.1950, 1.2100, TradeDirection.LONG)
        trade = Trade(trade_config, 0.1, 1000000, 12345)
        self.risk_manager.active_trades['EURUSD'] = trade
        new_stop = self.risk_manager.update_trailing_stop('EURUSD', 1.2050)
        self.assertGreater(new_stop, 1.1950)
        with open('logs/risk_management.log', 'r') as log_file:
            self.assertIn("Updated trailing stop for EURUSD", log_file.read())

    def test_get_risk_exposure(self):
        trade_config1 = TradeConfig('EURUSD', 1.2000, 1.1950, 1.2100, TradeDirection.LONG)
        trade_config2 = TradeConfig('GBPUSD', 1.3000, 1.2950, 1.3100, TradeDirection.SHORT)
        self.risk_manager.active_trades['EURUSD'] = Trade(trade_config1, 0.1, 1000000, 12345)
        self.risk_manager.active_trades['GBPUSD'] = Trade(trade_config2, 0.1, 1000000, 12346)
        risk_exposure = self.risk_manager.get_risk_exposure()
        self.assertGreater(risk_exposure, 0)
        self.assertLess(risk_exposure, 1)

    def test_get_performance_metrics(self):
        # Simulate some closed trades
        for i in range(10):
            trade_config = TradeConfig(f'PAIR{i}', 1.0, 0.99, 1.01, TradeDirection.LONG)
            trade = Trade(trade_config, 0.1, 1000000, i)
            trade.exit_price = 1.005 if i % 2 == 0 else 0.995
            trade.pnl = 0.005 if i % 2 == 0 else -0.005
            self.risk_manager.closed_trades.append(trade)

        metrics = self.risk_manager.get_performance_metrics()
        self.assertEqual(metrics['total_trades'], 10)
        self.assertAlmostEqual(metrics['win_rate'], 0.5, places=2)
        self.assertGreater(metrics['profit_factor'], 0)
        self.assertIsInstance(metrics['sharpe_ratio'], float)
        self.assertGreater(metrics['max_drawdown'], 0)

    def tearDown(self):
        patch.stopall()

if __name__ == '__main__':
    unittest.main()

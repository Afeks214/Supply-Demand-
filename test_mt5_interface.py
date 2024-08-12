import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from mt5_interface import MT5AdvancedInterface, ConfigurationError

class TestMT5AdvancedInterface(unittest.TestCase):
    def setUp(self):
        self.interface = MT5AdvancedInterface()

    @patch('mt5_interface.mt5')
    def test_connect_success(self, mock_mt5):
        mock_mt5.initialize.return_value = True
        self.interface.connect()
        self.assertTrue(self.interface.connected)
        with open('logs/mt5_interface.log', 'r') as log_file:
            self.assertIn("Connected to MetaTrader 5", log_file.read())

    @patch('mt5_interface.mt5')
    def test_connect_failure(self, mock_mt5):
        mock_mt5.initialize.return_value = False
        with self.assertRaises(ConfigurationError):
            self.interface.connect()
        with open('logs/mt5_interface.log', 'r') as log_file:
            self.assertIn("Failed to connect to MetaTrader 5", log_file.read())

    @patch('mt5_interface.mt5')
    def test_get_prices_success(self, mock_mt5):
        mock_data = [
            (0, 1.1234, 1.1235, 1.1233, 1.1234, 100, 0, 0),
            (1, 1.1235, 1.1236, 1.1234, 1.1235, 110, 0, 0)
        ]
        mock_mt5.copy_rates_from_pos.return_value = mock_data
        df = self.interface.get_prices('EURUSD', 'M1', 2)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[1]['close'], 1.1235)
        with open('logs/mt5_interface.log', 'r') as log_file:
            self.assertIn("Fetched price data for EURUSD", log_file.read())

    @patch('mt5_interface.mt5')
    def test_get_prices_failure(self, mock_mt5):
        mock_mt5.copy_rates_from_pos.return_value = None
        with self.assertRaises(ValueError):
            self.interface.get_prices('INVALID', 'M1', 1)
        with open('logs/mt5_interface.log', 'r') as log_file:
            self.assertIn("Failed to get price data", log_file.read())

    @patch('mt5_interface.mt5')
    def test_place_market_order_success(self, mock_mt5):
        mock_result = MagicMock()
        mock_result.retcode = mock_mt5.TRADE_RETCODE_DONE
        mock_result.order = 12345
        mock_mt5.order_send.return_value = mock_result
        order_id = self.interface.place_market_order('EURUSD', 0.1, 'BUY')
        self.assertEqual(order_id, 12345)
        with open('logs/mt5_interface.log', 'r') as log_file:
            self.assertIn("Placed market order for EURUSD", log_file.read())

    @patch('mt5_interface.mt5')
    def test_place_market_order_failure(self, mock_mt5):
        mock_result = MagicMock()
        mock_result.retcode = mock_mt5.TRADE_RETCODE_ERROR
        mock_mt5.order_send.return_value = mock_result
        with self.assertRaises(ValueError):
            self.interface.place_market_order('EURUSD', 0.1, 'BUY')
        with open('logs/mt5_interface.log', 'r') as log_file:
            self.assertIn("Failed to place market order", log_file.read())

    @patch('mt5_interface.mt5')
    def test_close_position_success(self, mock_mt5):
        mock_result = MagicMock()
        mock_result.retcode = mock_mt5.TRADE_RETCODE_DONE
        mock_mt5.order_send.return_value = mock_result
        self.assertTrue(self.interface.close_position(12345))
        with open('logs/mt5_interface.log', 'r') as log_file:
            self.assertIn("Closed position", log_file.read())

    @patch('mt5_interface.mt5')
    def test_close_position_failure(self, mock_mt5):
        mock_result = MagicMock()
        mock_result.retcode = mock_mt5.TRADE_RETCODE_ERROR
        mock_mt5.order_send.return_value = mock_result
        self.assertFalse(self.interface.close_position(12345))
        with open('logs/mt5_interface.log', 'r') as log_file:
            self.assertIn("Failed to close position", log_file.read())

    def test_get_account_info(self):
        with patch.object(self.interface, 'connected', True):
            with patch('mt5_interface.mt5') as mock_mt5:
                mock_account_info = MagicMock()
                mock_account_info.balance = 10000
                mock_account_info.equity = 10100
                mock_mt5.account_info.return_value = mock_account_info
                info = self.interface.get_account_info()
                self.assertEqual(info['balance'], 10000)
                self.assertEqual(info['equity'], 10100)

    def tearDown(self):
        patch.stopall()

if __name__ == '__main__':
    unittest.main()

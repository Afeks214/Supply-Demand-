import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import logging
from dataclasses import dataclass
from enum import Enum
import asyncio

from risk_management import RiskManagement, TradeConfig, TradeDirection
from config_manager import get_mt5_config, update_mt5_config, load_mt5_config
from indicators.mlmi import MLMI
from indicators.quadratic_regression import QuadraticRegression
from indicators.fair_value_gap import FairValueGap

class TimeFrame(Enum):
    M1 = mt5.TIMEFRAME_M1
    M5 = mt5.TIMEFRAME_M5

@dataclass
class MarketData:
    df_1m: pd.DataFrame
    df_5m: pd.DataFrame

class MultiPairTradingStrategy:
    def __init__(self, config_path: str):
        self.logger = self.setup_logger()
        self.load_config(config_path)
        self.risk_manager = RiskManagement(self.config.risk_management)
        self.mlmi = MLMI(self.config.signal.mlmi_neighbors, self.config.signal.mlmi_momentum_window)
        self.qr = QuadraticRegression(self.config.signal.qr_window_size, self.config.signal.qr_degree)
        self.fvg = FairValueGap(self.config.signal.fvg_threshold)
        self.active_trades: Dict[str, mt5.OrderSendResult] = {}
        self.market_data: Dict[str, MarketData] = {}

    def setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('MultiPairTradingStrategy')
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler('trading_strategy.log')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    def load_config(self, config_path: str):
        load_mt5_config(config_path)
        self.config = get_mt5_config()
        self.symbols = [symbol.name for symbol in self.config.trading.symbols]

    async def initialize_mt5(self) -> bool:
        if not mt5.initialize(
            login=self.config.connection.account,
            server=self.config.connection.server,
            password=self.config.connection.password,
            path=self.config.connection.path
        ):
            self.logger.error("MT5 initialization failed")
            mt5.shutdown()
            return False
        self.logger.info("MT5 initialized successfully")
        return True

    async def get_mt5_data(self, symbol: str, timeframe: TimeFrame, num_candles: int) -> Optional[pd.DataFrame]:
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe.value, 0, num_candles)
            if rates is None:
                self.logger.error(f"Failed to get data for {symbol} on {timeframe.name} timeframe")
                return None
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
        except Exception as e:
            self.logger.error(f"Error getting data for {symbol}: {str(e)}")
            return None

    async def update_market_data(self):
        for symbol in self.symbols:
            df_1m = await self.get_mt5_data(symbol, TimeFrame.M1, 100)
            df_5m = await self.get_mt5_data(symbol, TimeFrame.M5, 100)
            if df_1m is not None and df_5m is not None:
                self.market_data[symbol] = MarketData(df_1m, df_5m)

    async def check_entry_conditions(self, symbol: str) -> Tuple[bool, Optional[TradeDirection]]:
        if symbol not in self.market_data:
            return False, None

        data = self.market_data[symbol]
        mlmi_signal = self.mlmi.calculate(data.df_5m)
        qr_signal = self.qr.calculate(data.df_5m)
        fvg_signal = self.fvg.detect_touched_fvg(data.df_1m)

        conditions = {
            'mlmi_bullish': mlmi_signal['cross_above_ma'],
            'qr_bullish': qr_signal['is_bullish'],
            'fvg_bullish': fvg_signal['touched_bullish'],
            'mlmi_bearish': mlmi_signal['cross_below_ma'],
            'qr_bearish': qr_signal['is_bearish'],
            'fvg_bearish': fvg_signal['touched_bearish']
        }

        bullish_conditions = sum([conditions['mlmi_bullish'], conditions['qr_bullish'], conditions['fvg_bullish']])
        bearish_conditions = sum([conditions['mlmi_bearish'], conditions['qr_bearish'], conditions['fvg_bearish']])

        if bullish_conditions >= 2:
            return True, TradeDirection.LONG
        elif bearish_conditions >= 2:
            return True, TradeDirection.SHORT

        return False, None

    async def check_exit_conditions(self, symbol: str, direction: TradeDirection) -> bool:
        if symbol not in self.market_data:
            return False

        data = self.market_data[symbol]
        mlmi_signal = self.mlmi.calculate(data.df_5m)
        qr_signal = self.qr.calculate(data.df_5m)

        if direction == TradeDirection.LONG:
            return mlmi_signal['cross_below_ma'] or qr_signal['is_bearish']
        else:
            return mlmi_signal['cross_above_ma'] or qr_signal['is_bullish']

    async def enter_trade(self, symbol: str, direction: TradeDirection):
        try:
            current_price = mt5.symbol_info_tick(symbol).ask if direction == TradeDirection.LONG else mt5.symbol_info_tick(symbol).bid
            
            stop_loss = await self.calculate_stop_loss(symbol, direction, current_price)
            take_profit = await self.calculate_take_profit(symbol, direction, current_price)
            
            trade_config = TradeConfig(
                symbol=symbol,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                direction=direction
            )
            
            trade = self.risk_manager.open_trade(trade_config)
            if trade:
                order_type = mt5.ORDER_TYPE_BUY if direction == TradeDirection.LONG else mt5.ORDER_TYPE_SELL
                order = mt5.order_send({
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": trade.position_size,
                    "type": order_type,
                    "price": current_price,
                    "sl": stop_loss,
                    "tp": take_profit,
                    "magic": self.config.trading.magic_number,
                    "comment": "python script open",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                })
                
                if order.retcode == mt5.TRADE_RETCODE_DONE:
                    self.active_trades[symbol] = order
                    self.logger.info(f"Entered {direction.name} trade for {symbol}")
                else:
                    self.logger.error(f"Order for {symbol} failed, retcode: {order.retcode}")
            else:
                self.logger.warning(f"Risk manager rejected trade for {symbol}")
        except Exception as e:
            self.logger.error(f"Error entering trade for {symbol}: {str(e)}")

    async def exit_trade(self, symbol: str):
        try:
            if symbol in self.active_trades:
                order = self.active_trades[symbol]
                close_order = mt5.order_send({
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": order.volume,
                    "type": mt5.ORDER_TYPE_SELL if order.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "position": order.order,
                    "price": mt5.symbol_info_tick(symbol).bid if order.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask,
                    "magic": self.config.trading.magic_number,
                    "comment": "python script close",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                })
                
                if close_order.retcode == mt5.TRADE_RETCODE_DONE:
                    del self.active_trades[symbol]
                    self.risk_manager.close_trade(order.order)
                    self.logger.info(f"Exited trade for {symbol}")
                else:
                    self.logger.error(f"Failed to exit trade for {symbol}, retcode: {close_order.retcode}")
            else:
                self.logger.warning(f"No active trade found for {symbol}")
        except Exception as e:
            self.logger.error(f"Error exiting trade for {symbol}: {str(e)}")

    async def calculate_stop_loss(self, symbol: str, direction: TradeDirection, entry_price: float) -> float:
        atr = await self.calculate_atr(symbol)
        if direction == TradeDirection.LONG:
            return entry_price - 2 * atr
        else:
            return entry_price + 2 * atr

    async def calculate_take_profit(self, symbol: str, direction: TradeDirection, entry_price: float) -> float:
        atr = await self.calculate_atr(symbol)
        if direction == TradeDirection.LONG:
            return entry_price + 3 * atr
        else:
            return entry_price - 3 * atr

    async def calculate_atr(self, symbol: str, period: int = 14) -> float:
        if symbol not in self.market_data:
            return 0.0
        df = self.market_data[symbol].df_5m
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean().iloc[-1]

    async def run(self):
        if not await self.initialize_mt5():
            return

        self.logger.info("Starting trading strategy")
        while True:
            try:
                await self.update_market_data()
                await asyncio.gather(*[self.process_symbol(symbol) for symbol in self.symbols])
                await asyncio.sleep(1)  # Wait for 1 second before the next iteration
            except Exception as e:
                self.logger.error(f"Error in main loop: {str(e)}")
                await asyncio.sleep(5)  # Wait for 5 seconds before retrying

    async def process_symbol(self, symbol: str):
        try:
            if symbol in self.active_trades:
                # Check exit conditions
                direction = TradeDirection.LONG if self.active_trades[symbol].type == mt5.ORDER_TYPE_BUY else TradeDirection.SHORT
                if await self.check_exit_conditions(symbol, direction):
                    await self.exit_trade(symbol)
            else:
                # Check entry conditions
                entry_signal, direction = await self.check_entry_conditions(symbol)
                if entry_signal and len(self.active_trades) < self.config.risk_management.max_positions:
                    await self.enter_trade(symbol, direction)
        except Exception as e:
            self.logger.error(f"Error processing symbol {symbol}: {str(e)}")

    def get_performance_metrics(self) -> Dict[str, float]:
        return self.risk_manager.get_performance_metrics()

if __name__ == "__main__":
    config_path = "path/to/your/strategy_config.json"
    strategy = MultiPairTradingStrategy(config_path)
    
    try:
        asyncio.run(strategy.run())
    except KeyboardInterrupt:
        print("Strategy stopped by user.")
    finally:
        mt5.shutdown()
        print("MT5 connection closed.")
        
        # Print performance metrics
        metrics = strategy.get_performance_metrics()
        print("\nPerformance Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.2f}")

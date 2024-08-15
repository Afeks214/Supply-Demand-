import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import logging
import MetaTrader5 as mt5

class TradeDirection(Enum):
    LONG = "long"
    SHORT = "short"

@dataclass
class TradeConfig:
    symbol: str
    entry_price: float
    stop_loss: float
    take_profit: float
    direction: TradeDirection

@dataclass
class Trade:
    config: TradeConfig
    position_size: float
    entry_time: float
    order_id: int
    exit_time: Optional[float] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None

@dataclass
class RiskConfig:
    total_capital: float
    risk_per_trade: float
    max_trades_per_day: int
    max_positions: int
    max_daily_loss: float
    max_drawdown: float

class RiskManagement:
    def __init__(self, config: RiskConfig):
        self.config = config
        self.active_trades: Dict[str, Trade] = {}
        self.closed_trades: List[Trade] = []
        self.daily_trades = 0
        self.daily_pnl = 0
        self.peak_capital = config.total_capital
        self.current_capital = config.total_capital
        self.logger = self.setup_logger()

    def setup_logger(self):
        logger = logging.getLogger('RiskManagement')
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler('risk_management.log')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float) -> float:
        risk_amount = self.current_capital * self.config.risk_per_trade
        risk_per_unit = abs(entry_price - stop_loss)
        
        # Get symbol information from MT5
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            self.logger.error(f"Failed to get symbol info for {symbol}")
            return 0
        
        contract_size = symbol_info.trade_contract_size
        tick_size = symbol_info.trade_tick_size
        tick_value = symbol_info.trade_tick_value
        
        # Calculate position size in lots
        position_size_units = risk_amount / (risk_per_unit * contract_size * (tick_value / tick_size))
        
        # Round to the nearest valid lot size
        lot_step = symbol_info.volume_step
        position_size_lots = round(position_size_units / lot_step) * lot_step
        
        # Ensure position size is within allowed limits
        min_lot = symbol_info.volume_min
        max_lot = symbol_info.volume_max
        position_size_lots = max(min(position_size_lots, max_lot), min_lot)
        
        self.logger.info(f"Calculated position size for {symbol}: {position_size_lots} lots")
        return position_size_lots

    def can_open_trade(self) -> bool:
        if self.daily_trades >= self.config.max_trades_per_day:
            self.logger.warning("Maximum daily trades reached")
            return False
        if len(self.active_trades) >= self.config.max_positions:
            self.logger.warning("Maximum concurrent positions reached")
            return False
        if self.daily_pnl <= -self.config.max_daily_loss:
            self.logger.warning("Maximum daily loss reached")
            return False
        current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
        if current_drawdown >= self.config.max_drawdown:
            self.logger.warning("Maximum drawdown reached")
            return False
        return True

    def open_trade(self, trade_config: TradeConfig) -> Optional[Trade]:
        if not self.can_open_trade():
            return None

        position_size = self.calculate_position_size(
            trade_config.symbol, 
            trade_config.entry_price, 
            trade_config.stop_loss
        )
        
        if position_size == 0:
            self.logger.warning(f"Calculated position size is 0 for {trade_config.symbol}")
            return None

        trade = Trade(
            config=trade_config,
            position_size=position_size,
            entry_time=mt5.symbol_info_tick(trade_config.symbol).time,
            order_id=-1  # This will be updated when the order is actually placed
        )
        self.active_trades[trade_config.symbol] = trade
        self.daily_trades += 1
        self.logger.info(f"Opened trade for {trade_config.symbol}: {trade}")
        return trade

    def close_trade(self, symbol: str, exit_price: float, order_id: int) -> Optional[Trade]:
        if symbol not in self.active_trades:
            self.logger.warning(f"No active trade found for {symbol}")
            return None

        trade = self.active_trades.pop(symbol)
        trade.exit_time = mt5.symbol_info_tick(symbol).time
        trade.exit_price = exit_price
        trade.order_id = order_id
        
        if trade.config.direction == TradeDirection.LONG:
            trade.pnl = (exit_price - trade.config.entry_price) * trade.position_size
        else:
            trade.pnl = (trade.config.entry_price - exit_price) * trade.position_size

        self.closed_trades.append(trade)
        self.daily_pnl += trade.pnl
        self.current_capital += trade.pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)

        self.logger.info(f"Closed trade for {symbol}: {trade}")
        return trade

    def update_trailing_stop(self, symbol: str, current_price: float) -> Optional[float]:
        if symbol not in self.active_trades:
            return None

        trade = self.active_trades[symbol]
        if trade.config.direction == TradeDirection.LONG:
            new_stop = max(trade.config.stop_loss, current_price - (current_price - trade.config.entry_price) * 0.5)
        else:
            new_stop = min(trade.config.stop_loss, current_price + (trade.config.entry_price - current_price) * 0.5)

        if new_stop != trade.config.stop_loss:
            trade.config.stop_loss = new_stop
            self.logger.info(f"Updated trailing stop for {symbol}: {new_stop}")
        return new_stop

    def get_active_trades(self) -> Dict[str, Trade]:
        return self.active_trades

    def get_closed_trades(self) -> List[Trade]:
        return self.closed_trades

    def reset_daily_stats(self):
        self.daily_trades = 0
        self.daily_pnl = 0
        self.logger.info("Reset daily trading stats")

    def get_risk_exposure(self) -> float:
        total_risk = sum(
            trade.position_size * abs(trade.config.entry_price - trade.config.stop_loss)
            for trade in self.active_trades.values()
        )
        return total_risk / self.current_capital

    def get_performance_metrics(self) -> Dict[str, float]:
        if not self.closed_trades:
            return {"total_trades": 0, "win_rate": 0, "profit_factor": 0, "sharpe_ratio": 0, "max_drawdown": 0}

        wins = sum(1 for trade in self.closed_trades if trade.pnl > 0)
        total_profit = sum(trade.pnl for trade in self.closed_trades if trade.pnl > 0)
        total_loss = sum(abs(trade.pnl) for trade in self.closed_trades if trade.pnl < 0)

        win_rate = wins / len(self.closed_trades)
        profit_factor = total_profit / total_loss if total_loss != 0 else float('inf')

        returns = [trade.pnl / self.config.total_capital for trade in self.closed_trades]
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) != 0 else 0

        drawdowns = []
        peak = self.config.total_capital
        for trade in self.closed_trades:
            peak = max(peak, peak + trade.pnl)
            drawdown = (peak - (peak + trade.pnl)) / peak
            drawdowns.append(drawdown)
        max_drawdown = max(drawdowns) if drawdowns else 0

        return {
            "total_trades": len(self.closed_trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        }

# Example usage
if __name__ == "__main__":
    risk_config = RiskConfig(
        total_capital=100000,
        risk_per_trade=0.01,
        max_trades_per_day=10,
        max_positions=5,
        max_daily_loss=1000,
        max_drawdown=0.1
    )
    risk_manager = RiskManagement(risk_config)

    # Example trade
    trade_config = TradeConfig("EURUSD", 1.1000, 1.0990, 1.1020, TradeDirection.LONG)
    trade = risk_manager.open_trade(trade_config)
    if trade:
        print(f"Opened trade: {trade}")
        
        # Simulate closing the trade
        closed_trade = risk_manager.close_trade("EURUSD", 1.1015, 12345)
        if closed_trade:
            print(f"Closed trade: {closed_trade}")
    
    # Print performance metrics
    print(risk_manager.get_performance_metrics())

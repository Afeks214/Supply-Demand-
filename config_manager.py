import json
from typing import Dict, Any, List
from dataclasses import dataclass, field, asdict
import os
from dotenv import load_dotenv

load_dotenv()

class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass

@dataclass
class ConnectionConfig:
    account: int = field(default_factory=lambda: int(os.getenv('MT5_ACCOUNT', '0')))
    password: str = field(default_factory=lambda: os.getenv('MT5_PASSWORD', ''))
    server: str = field(default_factory=lambda: os.getenv('MT5_SERVER', ''))
    timeout: int = 60000
    path: str = ""

@dataclass
class SymbolConfig:
    name: str
    timeframes: List[str]
    chart_timeframe: str
    max_spread: float
    swap_long: float
    swap_short: float
    margin_rate: float

@dataclass
class TradingConfig:
    symbols: List[SymbolConfig] = field(default_factory=list)
    default_volume: float = 0.01
    default_deviation: int = 20
    magic_number: int = 123456
    trading_hours: Dict[str, List[str]] = field(default_factory=lambda: {
        "Monday": ["00:00-23:59"],
        "Tuesday": ["00:00-23:59"],
        "Wednesday": ["00:00-23:59"],
        "Thursday": ["00:00-23:59"],
        "Friday": ["00:00-23:59"]
    })

@dataclass
class RiskManagementConfig:
    max_positions: int = 5
    max_daily_loss: float = 100.0
    max_daily_profit: float = 500.0
    max_equity_risk_percent: float = 2.0
    default_stop_loss_pips: int = 50
    default_take_profit_pips: int = 100
    use_trailing_stop: bool = True
    trailing_stop_pips: int = 30

@dataclass
class SignalConfig:
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    ma_fast_period: int = 10
    ma_slow_period: int = 20
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    atr_period: int = 14
    mlmi_neighbors: int = 200
    mlmi_momentum_window: int = 20
    qr_window_size: int = 20
    qr_degree: int = 2
    fvg_threshold: float = 0.001

@dataclass
class LoggingConfig:
    level: str = "INFO"
    file_path: str = "mt5_trading_bot.log"
    max_file_size: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5

@dataclass
class MT5Config:
    connection: ConnectionConfig = field(default_factory=ConnectionConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    risk_management: RiskManagementConfig = field(default_factory=RiskManagementConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

class MT5ConfigurationManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.config = MT5Config()
        return cls._instance

    def get_config(self) -> MT5Config:
        return self.config

    def update_config(self, config_dict: Dict[str, Any]):
        def update_nested(obj, data):
            for key, value in data.items():
                if hasattr(obj, key):
                    if isinstance(value, dict) and not isinstance(getattr(obj, key), dict):
                        update_nested(getattr(obj, key), value)
                    else:
                        setattr(obj, key, value)

        update_nested(self.config, config_dict)

    def save_config(self, filename: str = 'strategy_config.json'):
        try:
            with open(filename, 'w') as f:
                json.dump(asdict(self.config), f, indent=4)
        except IOError as e:
            raise ConfigurationError(f"Error saving config to file: {e}")

    def load_config(self, filename: str = 'strategy_config.json'):
        try:
            with open(filename, 'r') as f:
                config_dict = json.load(f)
            self.update_config(config_dict)
        except (IOError, json.JSONDecodeError) as e:
            raise ConfigurationError(f"Error loading config from file: {e}")

    def validate_config(self):
        # Add validation logic here
        pass

# Usage
config_manager = MT5ConfigurationManager()

def get_mt5_config() -> MT5Config:
    return config_manager.get_config()

def update_mt5_config(config_dict: Dict[str, Any]):
    config_manager.update_config(config_dict)

def save_mt5_config(filename: str = 'strategy_config.json'):
    config_manager.save_config(filename)

def load_mt5_config(filename: str = 'strategy_config.json'):
    config_manager.load_config(filename)

# Example usage
if __name__ == "__main__":
    # Load configuration
    load_mt5_config()
    
    # Get current configuration
    config = get_mt5_config()
    print(f"Current MT5 account: {config.connection.account}")
    
    # Update configuration
    update_mt5_config({
        "connection": {
            "account": 87654321
        },
        "trading": {
            "symbols": [
                {
                    "name": "EURUSD",
                    "timeframes": ["M1", "M5"],
                    "chart_timeframe": "M5",
                    "max_spread": 1.5,
                    "swap_long": -1.2,
                    "swap_short": -0.8,
                    "margin_rate": 0.05
                }
            ]
        }
    })
    
    # Save updated configuration
    save_mt5_config()
    
    print(f"Updated MT5 account: {config.connection.account}")
    print(f"EURUSD settings: {config.trading.symbols[0]}")

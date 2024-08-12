import asyncio
import logging
from logging.handlers import RotatingFileHandler
import sys
import signal
from typing import Dict, Any
import json
import os
from datetime import datetime, timedelta

from mt5_interface import MT5AdvancedInterface
from risk_management import RiskManagement, RiskConfig
from trading_strategy import MultiPairTradingStrategy
from config_manager import load_mt5_config, get_mt5_config, MT5Config
from indicators.mlmi import MLMI
from indicators.quadratic_regression import QuadraticRegression
from indicators.fair_value_gap import FairValueGap

# Global variables for graceful shutdown
shutdown_event = asyncio.Event()
active_tasks = set()

def setup_logging() -> None:
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    os.makedirs('logs', exist_ok=True)
    file_handler = RotatingFileHandler('logs/trading_bot.log', maxBytes=5*1024*1024, backupCount=5)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

def signal_handler(signum, frame):
    logging.info(f"Received signal {signum}. Initiating graceful shutdown...")
    shutdown_event.set()

async def graceful_shutdown(strategy: MultiPairTradingStrategy, mt5_interface: MT5AdvancedInterface):
    logging.info("Performing graceful shutdown...")
    
    # Close all open positions
    for symbol in strategy.active_trades.keys():
        await strategy.exit_trade(symbol)
    
    # Disconnect from MT5
    await mt5_interface.disconnect()
    
    # Cancel all active tasks
    for task in active_tasks:
        task.cancel()
    await asyncio.gather(*active_tasks, return_exceptions=True)
    
    logging.info("Graceful shutdown completed.")

async def monitor_performance(strategy: MultiPairTradingStrategy):
    while not shutdown_event.is_set():
        metrics = strategy.get_performance_metrics()
        logging.info(f"Current performance metrics: {metrics}")
        await asyncio.sleep(3600)  # Update every hour

async def run_trading_bot(config_path: str):
    setup_logging()
    logging.info("Starting trading bot...")

    try:
        # Load configuration
        load_mt5_config(config_path)
        config: MT5Config = get_mt5_config()
        
        # Initialize MT5 interface
        mt5_interface = MT5AdvancedInterface(config.connection)
        if not await mt5_interface.connect():
            logging.error("Failed to connect to MT5. Exiting...")
            return

        # Initialize risk management
        risk_config = RiskConfig(**config.risk_management.__dict__)
        risk_manager = RiskManagement(risk_config)
        
        # Initialize indicators
        mlmi = MLMI(config.signal.mlmi_neighbors, config.signal.mlmi_momentum_window)
        qr = QuadraticRegression(config.signal.qr_window_size, config.signal.qr_degree)
        fvg = FairValueGap(config.signal.fvg_threshold)
        
        # Initialize trading strategy
        strategy = MultiPairTradingStrategy(
            config=config.trading,
            risk_manager=risk_manager,
            mt5_interface=mt5_interface,
            mlmi=mlmi,
            qr=qr,
            fvg=fvg
        )

        # Start the strategy
        strategy_task = asyncio.create_task(strategy.run())
        active_tasks.add(strategy_task)

        # Start performance monitoring
        monitor_task = asyncio.create_task(monitor_performance(strategy))
        active_tasks.add(monitor_task)

        # Main loop
        while not shutdown_event.is_set():
            # Check for any system-wide conditions or periodic tasks
            await asyncio.sleep(1)

            # Example: Check if it's time for daily reset
            if datetime.now().time() == config.trading.daily_reset_time:
                logging.info("Performing daily reset...")
                risk_manager.reset_daily_stats()
                # Any other daily reset tasks...

    except Exception as e:
        logging.exception(f"An unexpected error occurred: {str(e)}")
    finally:
        await graceful_shutdown(strategy, mt5_interface)
        logging.info("Trading bot stopped.")

if __name__ == "__main__":
    config_path = "config/strategy_config.json"
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the trading bot
    asyncio.run(run_trading_bot(config_path))

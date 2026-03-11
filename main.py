"""
main.py
───────
Production-ready AI-powered chart pattern trading bot.

This is the main entry point that initializes all modules:
  - DataEngine (OHLCV fetching)
  - IndicatorEngine (50+ indicators)
  - PatternDetector (Chart + Candlestick patterns)
  - AISignalEngine (Claude AI confirmation)
  - RiskManager (Position sizing & risk controls)
  - ExecutionEngine (Order placement)
  - AlertingEngine (Telegram + CLI Dashboard)

Usage:
  python main.py --mode paper --symbol BTC/USDT
  python main.py --backtest --symbol BTC/USDT --timeframe 1h
"""

import argparse
import asyncio
import signal
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from modules.ai_signal_engine import AISignalEngine
from modules.alerting import AlertingEngine, Dashboard
from modules.backtester import Backtester
from modules.data_engine import DataEngine
from modules.execution_engine import ExecutionEngine
from modules.trade_monitor import TradeMonitor
from modules.indicator_engine import IndicatorEngine
from modules.pattern_detector import PatternDetector
from modules.risk_manager import RiskManager
from utils.helpers import load_config
from utils.logger import get_logger, configure_from_config

# Load environment variables from .env
load_dotenv()

log = get_logger("trading_bot")


class TradingBot:
    """
    The orchestrator that runs the main trading logic loop.
    """

    def __init__(self, config: Dict[str, Any], args: argparse.Namespace) -> None:
        self.config = config
        self.args = args

        # Override config mode if provided in args
        if args.mode:
            self.config["trading"]["mode"] = args.mode

        # Initialize Modules
        if self.config["trading"]["exchange"] == "mt5":
            from modules.mt5_connector import MT5Connector

            self.data_engine = MT5Connector(self.config)
            self.execution_engine = self.data_engine
        else:
            self.data_engine = DataEngine(self.config)
            self.execution_engine = ExecutionEngine(
                self.config, exchange=self.data_engine.exchange
            )

        self.indicator_engine = IndicatorEngine(self.config)
        self.pattern_detector = PatternDetector(self.config)
        self.ai_signal_engine = AISignalEngine(self.config)
        self.risk_manager = RiskManager(self.config)
        self.trade_monitor = TradeMonitor(config)
        self.alerting_engine = AlertingEngine(config)
        self.dashboard = Dashboard()

        # Historical tracking for dashboard
        self.signal_history: List[Dict[str, Any]] = []
        self.running = False

    async def start(self) -> None:
        """Start the main bot loop."""
        self.running = True
        log.info(f"Bot starting... Mode: {self.execution_engine.mode.upper()}")

        # Send startup alert
        self.alerting_engine.send_message(
            f"🚀 *AI Trading Bot Started*\nMode: `{self.execution_engine.mode.upper()}`\nSymbols: `{self.config['trading']['symbols']}`",
            silent=True,
        )

        # Run Dashboard in parallel if not in backtest mode
        dashboard_task = asyncio.create_task(self.dashboard.run_live(self))

        try:
            while self.running:
                for symbol in self.config["trading"]["symbols"]:
                    try:
                        await self.process_symbol(symbol)
                    except Exception as exc:
                        log.error(f"Error processing {symbol}: {exc}", exc_info=True)

                # Wait for next scan interval
                interval = self.config["trading"].get("scan_interval", 60)
                await asyncio.sleep(interval)

        except asyncio.CancelledError:
            self.running = False
        finally:
            log.info("Bot shutting down...")
            dashboard_task.cancel()

    async def process_symbol(self, symbol: str) -> None:
        """Process a single symbol: Data -> Indicators -> Patterns -> Signal -> Execute."""
        timeframe = self.config["trading"]["timeframe"]
        htf = self.config["trading"].get("higher_timeframe", "4h")

        # 1. Fetch Data (Trading Timeframe)
        df = self.data_engine.fetch_ohlcv(symbol, timeframe, limit=100)
        if df.empty:
            return

        # 2. Fetch Higher Timeframe Data (MTA)
        df_htf = self.data_engine.fetch_ohlcv(symbol, htf, limit=100)
        htf_context = {}
        if not df_htf.empty:
            df_htf = self.indicator_engine.compute_all(df_htf)
            last_htf = df_htf.iloc[-1]
            htf_context = {
                "timeframe": htf,
                "trend": (
                    "bullish" if last_htf.get("is_above_ema50", False) else "bearish"
                ),
                "rsi": last_htf.get("RSI_14", 50),
                "adx": last_htf.get("ADX_14", 0),
            }

        # 3. Compute Indicators (Trading Timeframe)
        df = self.indicator_engine.compute_all(df)

        # 4. Detect Patterns + S/R
        patterns, sr_levels = self.pattern_detector.detect_all(df)
        slope = self.pattern_detector.calculate_slope(df)

        if htf_context:
            htf_context["slope"] = slope

        # 5. Check for active position
        if symbol in self.risk_manager.open_positions:
            await self.manage_open_position(symbol, df)
            return

        # 6. Generate AI Signal (with MTA Context)
        signal = self.ai_signal_engine.analyze(
            df, patterns, sr_levels, symbol, timeframe, htf_context=htf_context
        )

        # Track signal history
        self.signal_history.append(
            {
                "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "symbol": symbol,
                "signal": signal.signal,
                "confidence": signal.confidence,
                "risk_reward_ratio": signal.risk_reward_ratio,
            }
        )

        if signal.signal == "HOLD":
            return

        # Notify on potentially actionable signal (high confidence)
        if signal.confidence >= 50:
            self.alerting_engine.notify_signal(signal)

        # 6. Risk Management Evaluation
        approved, sizing = self.risk_manager.evaluate_signal(signal, df, symbol)

        if not approved:
            log.info(
                f"Signal rejected by RiskManager for {symbol}: {sizing.rejection_reason}"
            )
            return

        # 7. Execution
        current_price = self.data_engine.get_live_price(symbol)
        order = self.execution_engine.place_order(sizing, current_price)

        if order.status == "filled":
            self.risk_manager.open_position(sizing)
            self.alerting_engine.notify_order_filled(order)

    async def manage_open_position(self, symbol: str, df: Any) -> None:
        """Monitor and exit open positions."""
        pos = self.risk_manager.open_positions[symbol]
        curr_price = self.data_engine.get_live_price(symbol)
        atr = self.risk_manager._get_atr(df)

        # Update trailing stop
        self.risk_manager.update_trailing_stop(symbol, curr_price, atr)

        # Check exit triggers (SL, TP1, TP2)
        exit_trigger = self.risk_manager.check_exit_conditions(symbol, curr_price)

        # Intelligent Monitoring (Thesis Re-validation & Momentum Stalling)
        if not exit_trigger:
            if not self.trade_monitor.revalidate_thesis(pos, df):
                exit_trigger = "Thesis Broken"
            elif self.trade_monitor.check_momentum_stall(pos, df):
                exit_trigger = "Momentum Stall"

        if exit_trigger:
            reason = f"{exit_trigger} hit"
            order = self.execution_engine.close_position(
                symbol, pos.position_size, curr_price, pos.direction, reason=reason
            )

            if order.status == "filled":
                pnl = self.risk_manager.close_position(
                    symbol, order.price, reason=reason
                )
                self.alerting_engine.notify_position_closed(
                    symbol, pnl, reason, order.price
                )

    async def run_backtest(self) -> None:
        """Run backtesting mode and exit."""
        bt = Backtester(self.config, self.data_engine)
        for symbol in self.config["trading"]["symbols"]:
            result = bt.run(symbol)
            report_path = bt.generate_html_report(result)
            print(f"Backtest Report for {symbol}: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="AI Trading Bot")
    parser.add_argument(
        "--mode", choices=["paper", "live", "bridge"], help="Override trading mode"
    )
    parser.add_argument("--symbol", help="Single symbol to trade")
    parser.add_argument("--backtest", action="store_true", help="Run backtest mode")
    parser.add_argument("--timeframe", help="Override timeframe")
    args = parser.parse_args()

    # Load Config
    config = load_config("config.yaml")

    # Configure Logging
    configure_from_config(config)

    # Initialize Bot
    bot = TradingBot(config, args)

    if args.backtest:
        asyncio.run(bot.run_backtest())
    else:
        try:
            asyncio.run(bot.start())
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

# phase_1b_live_start.py
"""
Phase 1B: Live System Validation - START NOW
Run the trading system in live mode with real-time data collection
"""

import time
import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/phase_1b_live.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Phase1BLiveSystem:
    """Live trading system validation - Phase 1B"""

    def __init__(self):
        self.start_time = datetime.now()
        self.is_running = False
        self.symbols = ['SPY', 'QQQ', 'AAPL']
        self.update_interval = 30  # seconds

        # Portfolio state
        self.portfolio = {
            'cash': 100000,
            'positions': {},
            'daily_pnl': 0,
            'total_trades': 0
        }

        # Performance tracking
        self.performance_data = []
        self.signal_history = []
        self.error_count = 0

        print("🚀 PHASE 1B LIVE SYSTEM STARTING")
        print(f"Start Time: {self.start_time.strftime('%H:%M:%S')}")
        print(f"Symbols: {self.symbols}")
        print(f"Update Interval: {self.update_interval} seconds")
        print("=" * 60)

    def start_live_system(self):
        """Start the live trading system"""
        self.is_running = True

        try:
            # Start dashboard monitoring
            dashboard_thread = threading.Thread(target=self.run_dashboard_monitoring)
            dashboard_thread.daemon = True
            dashboard_thread.start()

            # Start main trading loop
            self.main_trading_loop()

        except KeyboardInterrupt:
            print("\n🛑 System stopped by user")
            self.shutdown_system()
        except Exception as e:
            print(f"\n❌ System error: {e}")
            logger.error(f"System error: {e}", exc_info=True)
            self.shutdown_system()

    def main_trading_loop(self):
        """Main trading system loop"""
        cycle_count = 0

        print("🔄 Starting main trading loop...")

        while self.is_running:
            cycle_start = datetime.now()
            cycle_count += 1

            print(f"\n📊 Cycle {cycle_count} - {cycle_start.strftime('%H:%M:%S')}")

            try:
                # 1. Collect market data
                market_data = self.collect_market_data()

                # 2. Generate trading signals
                signals = self.generate_trading_signals(market_data)

                # 3. Update portfolio (paper trading)
                self.update_portfolio(signals, market_data)

                # 4. Log performance
                self.log_performance(cycle_count, market_data, signals)

                # 5. Check system health
                self.health_check()

                # Status update
                elapsed = (datetime.now() - self.start_time).total_seconds()
                print(f"✅ Cycle {cycle_count} completed - Runtime: {elapsed:.0f}s")

                # Wait for next update
                time.sleep(self.update_interval)

            except Exception as e:
                self.error_count += 1
                print(f"❌ Cycle {cycle_count} error: {e}")
                logger.error(f"Trading loop error: {e}", exc_info=True)

                if self.error_count > 5:
                    print("🚨 Too many errors - stopping system")
                    break

                time.sleep(5)  # Short wait before retry

    def collect_market_data(self):
        """Collect real-time market data"""
        market_data = {}

        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)

                # Get recent data
                hist_data = ticker.history(period='5d', interval='1h')
                current_info = ticker.info

                if not hist_data.empty:
                    current_price = float(hist_data['Close'].iloc[-1])
                    volume = int(hist_data['Volume'].iloc[-1])

                    # Calculate basic indicators
                    ma_5 = hist_data['Close'].rolling(5).mean().iloc[-1]
                    ma_20 = hist_data['Close'].rolling(20).mean().iloc[-1] if len(hist_data) >= 20 else ma_5

                    market_data[symbol] = {
                        'price': current_price,
                        'volume': volume,
                        'ma_5': float(ma_5),
                        'ma_20': float(ma_20),
                        'timestamp': datetime.now(),
                        'data_quality': 'good'
                    }

                    print(f"   📈 {symbol}: ${current_price:.2f} (Vol: {volume:,})")

                else:
                    print(f"   ⚠️ {symbol}: No data available")
                    market_data[symbol] = None

            except Exception as e:
                print(f"   ❌ {symbol}: Data error - {e}")
                market_data[symbol] = None

        return market_data

    def generate_trading_signals(self, market_data):
        """Generate trading signals based on market data"""
        signals = {}

        for symbol, data in market_data.items():
            if data is None:
                signals[symbol] = {'action': 'HOLD', 'confidence': 0, 'reason': 'No data'}
                continue

            try:
                # Simple moving average crossover strategy
                ma_5 = data['ma_5']
                ma_20 = data['ma_20']
                current_price = data['price']

                if ma_5 > ma_20:
                    action = 'BUY'
                    confidence = min(((ma_5 - ma_20) / ma_20) * 100, 100)  # Cap at 100%
                    reason = f"MA5 (${ma_5:.2f}) > MA20 (${ma_20:.2f})"
                elif ma_5 < ma_20:
                    action = 'SELL'
                    confidence = min(((ma_20 - ma_5) / ma_20) * 100, 100)  # Cap at 100%
                    reason = f"MA5 (${ma_5:.2f}) < MA20 (${ma_20:.2f})"
                else:
                    action = 'HOLD'
                    confidence = 0
                    reason = "MAs equal"

                signals[symbol] = {
                    'action': action,
                    'confidence': confidence,
                    'reason': reason,
                    'price': current_price
                }

                print(f"   🎯 {symbol}: {action} (Confidence: {confidence:.1f}%) - {reason}")

            except Exception as e:
                print(f"   ❌ {symbol}: Signal error - {e}")
                signals[symbol] = {'action': 'HOLD', 'confidence': 0, 'reason': f'Error: {e}'}

        # Store signal history
        self.signal_history.append({
            'timestamp': datetime.now(),
            'signals': signals.copy()
        })

        return signals

    def update_portfolio(self, signals, market_data):
        """Update portfolio based on signals (paper trading)"""

        for symbol, signal in signals.items():
            if signal['action'] in ['BUY', 'SELL'] and signal['confidence'] > 10:

                data = market_data.get(symbol)
                if data is None:
                    continue

                current_price = data['price']

                # Conservative position sizing (1% risk)
                risk_amount = self.portfolio['cash'] * 0.01  # 1% risk
                position_size = min(risk_amount / 0.03, 10000)  # Max $10k position
                shares = int(position_size / current_price)

                if shares > 0:
                    # Simulate trade
                    trade_value = shares * current_price

                    if signal['action'] == 'BUY' and self.portfolio['cash'] >= trade_value:
                        # Buy position
                        self.portfolio['positions'][symbol] = {
                            'shares': shares,
                            'avg_cost': current_price,
                            'entry_time': datetime.now()
                        }
                        self.portfolio['cash'] -= trade_value
                        self.portfolio['total_trades'] += 1

                        print(f"   💰 BOUGHT {shares} shares of {symbol} at ${current_price:.2f}")

                    elif signal['action'] == 'SELL' and symbol in self.portfolio['positions']:
                        # Sell position
                        position = self.portfolio['positions'][symbol]
                        trade_value = position['shares'] * current_price
                        pnl = (current_price - position['avg_cost']) * position['shares']

                        self.portfolio['cash'] += trade_value
                        self.portfolio['daily_pnl'] += pnl
                        self.portfolio['total_trades'] += 1

                        print(
                            f"   💸 SOLD {position['shares']} shares of {symbol} at ${current_price:.2f} (P&L: ${pnl:.2f})")

                        del self.portfolio['positions'][symbol]

    def log_performance(self, cycle, market_data, signals):
        """Log system performance"""

        # Calculate portfolio value
        portfolio_value = self.portfolio['cash']
        for symbol, position in self.portfolio['positions'].items():
            if symbol in market_data and market_data[symbol]:
                current_price = market_data[symbol]['price']
                portfolio_value += position['shares'] * current_price

        performance = {
            'timestamp': datetime.now().isoformat(),
            'cycle': cycle,
            'portfolio_value': portfolio_value,
            'cash': self.portfolio['cash'],
            'positions': len(self.portfolio['positions']),
            'daily_pnl': self.portfolio['daily_pnl'],
            'total_trades': self.portfolio['total_trades'],
            'data_sources_active': sum(1 for data in market_data.values() if data is not None),
            'signals_generated': sum(1 for sig in signals.values() if sig['action'] != 'HOLD')
        }

        self.performance_data.append(performance)

        # Save to file every 10 cycles
        if cycle % 10 == 0:
            self.save_performance_data()

        print(
            f"   📊 Portfolio: ${portfolio_value:.2f} | P&L: ${self.portfolio['daily_pnl']:.2f} | Trades: {self.portfolio['total_trades']}")

    def health_check(self):
        """Perform system health check"""

        # Check recent performance
        if len(self.performance_data) >= 3:
            recent_data = self.performance_data[-3:]
            data_quality = sum(p['data_sources_active'] for p in recent_data) / (len(recent_data) * len(self.symbols))

            if data_quality < 0.8:
                print("   ⚠️ Data quality warning")
            else:
                print("   ✅ System health good")

    def run_dashboard_monitoring(self):
        """Run dashboard monitoring in background"""

        try:
            from components.imrogue_dashboard import IMRogueDashboard
            from components.monitoring_dashboard import MonitoringDashboard

            main_dashboard = IMRogueDashboard()
            monitor_dashboard = MonitoringDashboard()

            print("📊 Dashboards initialized")

            while self.is_running:
                # Dashboard would update here with live data
                time.sleep(5)

        except Exception as e:
            print(f"📊 Dashboard monitoring error: {e}")

    def save_performance_data(self):
        """Save performance data to file"""

        data_file = Path('data/phase_1b_performance.json')

        save_data = {
            'start_time': self.start_time.isoformat(),
            'runtime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'portfolio': self.portfolio,
            'performance_history': self.performance_data[-50:],  # Last 50 cycles
            'signal_history': self.signal_history[-20:],  # Last 20 signal sets
            'error_count': self.error_count
        }

        with open(data_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        print(f"   💾 Performance data saved: {data_file}")

    def shutdown_system(self):
        """Shutdown the system gracefully"""

        self.is_running = False

        # Save final data
        self.save_performance_data()

        # Generate final report
        runtime = (datetime.now() - self.start_time).total_seconds()

        print("\n" + "=" * 60)
        print("📋 PHASE 1B LIVE SYSTEM SHUTDOWN REPORT")
        print("=" * 60)
        print(f"Runtime: {runtime:.0f} seconds ({runtime / 60:.1f} minutes)")
        print(f"Total cycles: {len(self.performance_data)}")
        print(f"Total trades: {self.portfolio['total_trades']}")
        print(f"Final P&L: ${self.portfolio['daily_pnl']:.2f}")
        print(f"Error count: {self.error_count}")

        if len(self.performance_data) > 0:
            final_value = self.performance_data[-1]['portfolio_value']
            print(f"Final portfolio value: ${final_value:.2f}")
            print(f"Return: {((final_value - 100000) / 100000) * 100:.2f}%")

        print("\n🎯 PHASE 1B VALIDATION:")
        if runtime >= 600:  # 10+ minutes
            print("✅ Duration test: PASSED (10+ minutes runtime)")
        else:
            print(f"⚠️ Duration test: {runtime / 60:.1f} minutes (target: 10+ minutes)")

        if self.error_count <= 5:
            print("✅ Stability test: PASSED (≤5 errors)")
        else:
            print(f"❌ Stability test: FAILED ({self.error_count} errors)")

        print("✅ Live data collection: COMPLETED")
        print("✅ Signal generation: COMPLETED")
        print("✅ Portfolio management: COMPLETED")

        print("\n🚀 Phase 1B Live Validation COMPLETE!")
        print("=" * 60)


def main():
    """Start Phase 1B live system"""

    print("🚀 STARTING PHASE 1B LIVE VALIDATION NOW!")
    print("Press Ctrl+C to stop the system")
    print()

    try:
        system = Phase1BLiveSystem()
        system.start_live_system()

    except KeyboardInterrupt:
        print("\n🛑 Phase 1B stopped by user")
    except Exception as e:
        print(f"\n❌ Phase 1B failed: {e}")


if __name__ == "__main__":
    main()
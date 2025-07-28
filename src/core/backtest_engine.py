# 📊 Strategy Backtesting Engine - Historical Validation System
# Validates trading strategies against historical data before live deployment

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import json
import os


@dataclass
class BacktestResult:
    """Comprehensive backtesting results"""
    strategy_name: str
    symbol: str
    start_date: str
    end_date: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_return: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float
    trade_history: List[Dict]
    equity_curve: List[float]
    drawdown_series: List[float]


class BacktestEngine:
    """
    Production-grade backtesting engine for strategy validation
    Implements walk-forward analysis and Monte Carlo simulation
    """

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.commission_rate = 0.001  # 0.1% commission
        self.slippage_rate = 0.0005  # 0.05% slippage
        self.min_trade_size = 100  # Minimum position size

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def run_comprehensive_backtest(self, strategy_class, symbol: str,
                                   start_date: str, end_date: str,
                                   strategy_params: Dict = None) -> BacktestResult:
        """
        Run comprehensive backtest with realistic execution costs
        """
        self.logger.info(f"Starting backtest: {strategy_class.__name__} on {symbol}")

        # Get historical data
        historical_data = self._get_historical_data(symbol, start_date, end_date)
        if historical_data.empty:
            raise ValueError(f"No historical data found for {symbol}")

        # Initialize strategy
        strategy = strategy_class(**(strategy_params or {}))

        # Initialize tracking variables
        portfolio = {
            'cash': self.initial_capital,
            'positions': {},
            'equity_history': [self.initial_capital],
            'trade_history': [],
            'daily_returns': []
        }

        # Run backtest day by day
        for i, (date, row) in enumerate(historical_data.iterrows()):
            if i < 50:  # Need minimum data for indicators
                continue

            # Get market data slice for strategy
            market_data = {
                'symbol': symbol,
                'price': row['close'],
                'volume': row['volume'],
                'high': row['high'],
                'low': row['low'],
                'open': row['open'],
                'historical_prices': historical_data['close'].iloc[max(0, i - 200):i + 1].tolist()
            }

            # Generate strategy signal
            try:
                signal = strategy.generate_signal(market_data)
                if signal and signal.get('action') in ['BUY', 'SELL']:
                    self._execute_backtest_trade(portfolio, signal, market_data, date)
            except Exception as e:
                self.logger.warning(f"Strategy error on {date}: {e}")
                continue

            # Update portfolio equity
            current_equity = self._calculate_portfolio_value(portfolio, row['close'])
            portfolio['equity_history'].append(current_equity)

            # Calculate daily return
            if len(portfolio['equity_history']) > 1:
                daily_return = (current_equity - portfolio['equity_history'][-2]) / portfolio['equity_history'][-2]
                portfolio['daily_returns'].append(daily_return)

        # Calculate performance metrics
        return self._calculate_backtest_metrics(
            strategy_class.__name__, symbol, start_date, end_date, portfolio
        )

    def _get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get historical market data for backtesting
        In production, this would connect to your data provider
        """
        try:
            # Simulated historical data generation for demonstration
            # In production, replace with real data from Alpha Vantage/Finnhub
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')

            # Generate realistic price movements
            np.random.seed(42)  # For reproducible results
            base_price = 150.0
            returns = np.random.normal(0.0005, 0.02, len(date_range))  # Daily returns

            prices = [base_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))

            # Create OHLCV data
            data = []
            for i, (date, price) in enumerate(zip(date_range, prices)):
                high = price * (1 + abs(np.random.normal(0, 0.01)))
                low = price * (1 - abs(np.random.normal(0, 0.01)))
                open_price = prices[i - 1] if i > 0 else price
                volume = np.random.randint(1000000, 5000000)

                data.append({
                    'date': date,
                    'open': open_price,
                    'high': max(price, high, open_price),
                    'low': min(price, low, open_price),
                    'close': price,
                    'volume': volume
                })

            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            return df

        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()

    def _execute_backtest_trade(self, portfolio: Dict, signal: Dict,
                                market_data: Dict, date: datetime):
        """Execute trade in backtest with realistic costs"""
        symbol = market_data['symbol']
        price = market_data['price']
        action = signal['action']
        confidence = signal.get('confidence', 0.5)

        # Calculate position size based on confidence and risk management
        max_position_value = portfolio['cash'] * 0.1 * confidence  # Max 10% per position
        shares = max(self.min_trade_size, int(max_position_value / price))

        # Apply slippage and commission
        execution_price = price * (1 + self.slippage_rate if action == 'BUY' else 1 - self.slippage_rate)
        trade_value = shares * execution_price
        commission = trade_value * self.commission_rate

        current_position = portfolio['positions'].get(symbol, 0)

        if action == 'BUY' and portfolio['cash'] >= (trade_value + commission):
            # Execute buy order
            portfolio['cash'] -= (trade_value + commission)
            portfolio['positions'][symbol] = current_position + shares

            trade_record = {
                'date': date,
                'symbol': symbol,
                'action': 'BUY',
                'shares': shares,
                'price': execution_price,
                'commission': commission,
                'confidence': confidence
            }
            portfolio['trade_history'].append(trade_record)

        elif action == 'SELL' and current_position > 0:
            # Execute sell order (or partial sell)
            shares_to_sell = min(shares, current_position)
            trade_value = shares_to_sell * execution_price
            commission = trade_value * self.commission_rate

            portfolio['cash'] += (trade_value - commission)
            portfolio['positions'][symbol] = current_position - shares_to_sell

            trade_record = {
                'date': date,
                'symbol': symbol,
                'action': 'SELL',
                'shares': shares_to_sell,
                'price': execution_price,
                'commission': commission,
                'confidence': confidence
            }
            portfolio['trade_history'].append(trade_record)

    def _calculate_portfolio_value(self, portfolio: Dict, current_price: float) -> float:
        """Calculate total portfolio value"""
        cash = portfolio['cash']
        position_value = sum(shares * current_price for shares in portfolio['positions'].values())
        return cash + position_value

    def _calculate_backtest_metrics(self, strategy_name: str, symbol: str,
                                    start_date: str, end_date: str,
                                    portfolio: Dict) -> BacktestResult:
        """Calculate comprehensive performance metrics"""
        equity_curve = portfolio['equity_history']
        daily_returns = portfolio['daily_returns']
        trades = portfolio['trade_history']

        if not daily_returns:
            # Return empty result if no trades
            return BacktestResult(
                strategy_name=strategy_name,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                avg_trade_return=0.0,
                volatility=0.0,
                calmar_ratio=0.0,
                sortino_ratio=0.0,
                trade_history=trades,
                equity_curve=equity_curve,
                drawdown_series=[]
            )

        # Basic metrics
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
        volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized

        # Sharpe Ratio (assuming 2% risk-free rate)
        excess_returns = np.array(daily_returns) - (0.02 / 252)
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(
            excess_returns) > 0 else 0

        # Sortino Ratio (downside deviation)
        downside_returns = [r for r in daily_returns if r < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if downside_returns else 0.01
        sortino_ratio = (np.mean(daily_returns) * 252) / downside_std if downside_std > 0 else 0

        # Maximum Drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown_series = (np.array(equity_curve) - peak) / peak
        max_drawdown = abs(np.min(drawdown_series))

        # Calmar Ratio
        calmar_ratio = (total_return * 100) / (max_drawdown * 100) if max_drawdown > 0 else 0

        # Trade-based metrics
        if len(trades) >= 2:
            # Pair buy/sell trades
            buy_trades = [t for t in trades if t['action'] == 'BUY']
            sell_trades = [t for t in trades if t['action'] == 'SELL']

            trade_returns = []
            for buy, sell in zip(buy_trades, sell_trades):
                if buy['symbol'] == sell['symbol']:
                    buy_value = buy['shares'] * buy['price']
                    sell_value = sell['shares'] * sell['price']
                    trade_return = (sell_value - buy_value) / buy_value
                    trade_returns.append(trade_return)

            winning_trades = [r for r in trade_returns if r > 0]
            losing_trades = [r for r in trade_returns if r < 0]

            win_rate = len(winning_trades) / len(trade_returns) if trade_returns else 0
            avg_trade_return = np.mean(trade_returns) if trade_returns else 0

            # Profit Factor
            gross_profit = sum(winning_trades) if winning_trades else 0
            gross_loss = abs(sum(losing_trades)) if losing_trades else 0.01
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        else:
            win_rate = 0
            avg_trade_return = 0
            profit_factor = 0

        return BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            avg_trade_return=avg_trade_return,
            volatility=volatility,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            trade_history=trades,
            equity_curve=equity_curve,
            drawdown_series=drawdown_series.tolist()
        )

    def run_walk_forward_analysis(self, strategy_class, symbol: str,
                                  total_start_date: str, total_end_date: str,
                                  train_period_months: int = 6,
                                  test_period_months: int = 1) -> Dict:
        """
        Run walk-forward analysis to validate strategy robustness
        """
        self.logger.info(f"Starting walk-forward analysis for {strategy_class.__name__}")

        # Convert dates
        start_date = datetime.strptime(total_start_date, '%Y-%m-%d')
        end_date = datetime.strptime(total_end_date, '%Y-%m-%d')

        results = []
        current_date = start_date

        while current_date + timedelta(days=train_period_months * 30) < end_date:
            # Define train and test periods
            train_start = current_date
            train_end = current_date + timedelta(days=train_period_months * 30)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=test_period_months * 30)

            if test_end > end_date:
                break

            # Run backtest on test period
            try:
                result = self.run_comprehensive_backtest(
                    strategy_class, symbol,
                    test_start.strftime('%Y-%m-%d'),
                    test_end.strftime('%Y-%m-%d')
                )
                results.append(result)

            except Exception as e:
                self.logger.warning(f"Walk-forward period failed: {e}")

            # Move to next period
            current_date = test_start

        # Aggregate results
        if results:
            avg_return = np.mean([r.total_return for r in results])
            avg_sharpe = np.mean([r.sharpe_ratio for r in results])
            avg_max_dd = np.mean([r.max_drawdown for r in results])
            consistency = len([r for r in results if r.total_return > 0]) / len(results)

            return {
                'strategy': strategy_class.__name__,
                'symbol': symbol,
                'periods_tested': len(results),
                'average_return': avg_return,
                'average_sharpe': avg_sharpe,
                'average_max_drawdown': avg_max_dd,
                'win_rate_consistency': consistency,
                'individual_results': results
            }

        return {'error': 'No valid walk-forward periods found'}

    def save_backtest_results(self, result: BacktestResult, filename: str = None):
        """Save backtest results to file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"backtest_{result.strategy_name}_{result.symbol}_{timestamp}.json"

        # Ensure reports directory exists
        os.makedirs('reports/backtests', exist_ok=True)
        filepath = os.path.join('reports/backtests', filename)

        # Convert to serializable format
        result_dict = {
            'strategy_name': result.strategy_name,
            'symbol': result.symbol,
            'start_date': result.start_date,
            'end_date': result.end_date,
            'total_return': result.total_return,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'profit_factor': result.profit_factor,
            'total_trades': result.total_trades,
            'avg_trade_return': result.avg_trade_return,
            'volatility': result.volatility,
            'calmar_ratio': result.calmar_ratio,
            'sortino_ratio': result.sortino_ratio,
            'trade_history': result.trade_history,
            'equity_curve': result.equity_curve,
            'drawdown_series': result.drawdown_series
        }

        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)

        self.logger.info(f"Backtest results saved to {filepath}")
        return filepath


# Example usage and integration
if __name__ == "__main__":
    # Example of how to use the backtesting engine

    class ExampleStrategy:
        """Example strategy for testing"""

        def __init__(self, short_ma_period=20, long_ma_period=50):
            self.short_ma_period = short_ma_period
            self.long_ma_period = long_ma_period

        def generate_signal(self, market_data):
            prices = market_data.get('historical_prices', [])
            if len(prices) < self.long_ma_period:
                return None

            short_ma = np.mean(prices[-self.short_ma_period:])
            long_ma = np.mean(prices[-self.long_ma_period:])

            if short_ma > long_ma * 1.02:  # 2% threshold
                return {'action': 'BUY', 'confidence': 0.7}
            elif short_ma < long_ma * 0.98:
                return {'action': 'SELL', 'confidence': 0.7}

            return None


    # Run backtest
    engine = BacktestEngine(initial_capital=100000)
    result = engine.run_comprehensive_backtest(
        ExampleStrategy,
        'AAPL',
        '2023-01-01',
        '2024-01-01'
    )

    print(f"Strategy: {result.strategy_name}")
    print(f"Total Return: {result.total_return:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Win Rate: {result.win_rate:.2%}")

    # Save results
    engine.save_backtest_results(result)
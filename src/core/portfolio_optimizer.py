# 📊 Portfolio Optimizer - Intelligent Position Sizing & Risk Management
# Implements CVaR optimization, Hierarchical Risk Parity, and AI consultation insights

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


@dataclass
class OptimizationResult:
    """Portfolio optimization result"""
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    cvar: float
    max_drawdown: float
    method_used: str
    confidence_level: float
    total_allocation: float
    rebalance_needed: bool


@dataclass
class PortfolioState:
    """Current portfolio state"""
    positions: Dict[str, int]  # Symbol -> shares
    cash: float
    total_value: float
    weights: Dict[str, float]
    last_rebalance: datetime


class PortfolioOptimizer:
    """
    Advanced portfolio optimizer implementing AI consultation insights:
    - CVaR optimization (Gemini recommendation)
    - Hierarchical Risk Parity for small portfolios (ChatGPT insight)
    - Robust covariance estimation (Co-Pilot recommendation)
    - Dynamic position sizing with confidence weighting
    """

    def __init__(self, confidence_level: float = 0.99, lookback_days: int = 252):
        self.confidence_level = confidence_level  # CVaR confidence level
        self.lookback_days = lookback_days  # Historical data window
        self.min_weight = 0.01  # Minimum 1% allocation
        self.max_weight = 0.20  # Maximum 20% allocation (per AI consultation)
        self.risk_free_rate = 0.02  # 2% annual risk-free rate

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Portfolio state tracking
        self.current_state = None
        self.rebalance_threshold = 0.05  # 5% deviation triggers rebalance

    def optimize_portfolio(self, returns_data: pd.DataFrame,
                           strategy_signals: Dict[str, Dict],
                           current_prices: Dict[str, float],
                           available_cash: float,
                           method: str = "cvar_optimal") -> OptimizationResult:
        """
        Main portfolio optimization function

        Args:
            returns_data: Historical returns for symbols
            strategy_signals: Current strategy signals with confidence scores
            current_prices: Current market prices
            available_cash: Available cash for investment
            method: Optimization method (cvar_optimal, hrp, equal_weight, signal_weighted)
        """

        self.logger.info(f"Starting portfolio optimization using {method}")

        # Validate inputs
        if returns_data.empty:
            raise ValueError("No returns data provided")

        symbols = list(returns_data.columns)

        # Filter signals to only include symbols we have data for
        filtered_signals = {s: strategy_signals.get(s, {'confidence': 0.0, 'action': 'HOLD'})
                            for s in symbols}

        try:
            if method == "cvar_optimal":
                result = self._optimize_cvar(returns_data, filtered_signals, current_prices, available_cash)
            elif method == "hrp":
                result = self._optimize_hrp(returns_data, filtered_signals, current_prices, available_cash)
            elif method == "signal_weighted":
                result = self._optimize_signal_weighted(returns_data, filtered_signals, current_prices, available_cash)
            elif method == "equal_weight":
                result = self._optimize_equal_weight(returns_data, filtered_signals, current_prices, available_cash)
            else:
                self.logger.warning(f"Unknown method {method}, falling back to CVaR optimization")
                result = self._optimize_cvar(returns_data, filtered_signals, current_prices, available_cash)

            # Validate result
            total_weight = sum(result.weights.values())
            if abs(total_weight - 1.0) > 0.01:  # Allow 1% tolerance
                self.logger.warning(f"Portfolio weights sum to {total_weight:.3f}, normalizing")
                result.weights = {s: w / total_weight for s, w in result.weights.items()}

            self.logger.info(f"Optimization complete. Method: {result.method_used}, "
                             f"Expected Return: {result.expected_return:.2%}, "
                             f"Volatility: {result.volatility:.2%}, "
                             f"Sharpe: {result.sharpe_ratio:.2f}")

            return result

        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            # Fallback to equal weight
            return self._optimize_equal_weight(returns_data, filtered_signals, current_prices, available_cash)

    def _optimize_cvar(self, returns_data: pd.DataFrame, signals: Dict[str, Dict],
                       prices: Dict[str, float], cash: float) -> OptimizationResult:
        """
        CVaR (Conditional Value at Risk) optimization
        Recommended by Gemini AI consultation as superior to VaR for tail risk
        """

        # Calculate returns and robust covariance matrix
        returns = returns_data.values
        mean_returns = np.mean(returns, axis=0)

        # Use Ledoit-Wolf shrinkage for robust covariance estimation (Co-Pilot insight)
        lw = LedoitWolf()
        cov_matrix = lw.fit(returns).covariance_

        n_assets = len(returns_data.columns)
        symbols = list(returns_data.columns)

        # Incorporate signal confidence into expected returns
        adjusted_returns = self._adjust_returns_for_signals(mean_returns, symbols, signals)

        # CVaR optimization objective function
        def cvar_objective(weights):
            portfolio_returns = np.dot(returns, weights)

            # Calculate VaR
            var = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)

            # Calculate CVaR (average of returns below VaR)
            cvar_returns = portfolio_returns[portfolio_returns <= var]
            cvar = np.mean(cvar_returns) if len(cvar_returns) > 0 else var

            # Minimize negative CVaR (maximize CVaR)
            return -cvar

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # Weights sum to 1
        ]

        # Bounds (min and max allocation per asset)
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]

        # Initial guess (signal-weighted)
        initial_weights = self._get_signal_based_initial_weights(symbols, signals)

        # Optimize
        result = minimize(
            cvar_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if not result.success:
            self.logger.warning("CVaR optimization failed, using signal weights")
            weights_dict = {symbols[i]: initial_weights[i] for i in range(n_assets)}
        else:
            weights_dict = {symbols[i]: result.x[i] for i in range(n_assets)}

        # Calculate portfolio metrics
        portfolio_return = np.dot(adjusted_returns, list(weights_dict.values()))
        portfolio_vol = np.sqrt(np.dot(list(weights_dict.values()),
                                       np.dot(cov_matrix, list(weights_dict.values()))))
        sharpe = (portfolio_return - self.risk_free_rate / 252) / portfolio_vol if portfolio_vol > 0 else 0

        # Calculate CVaR
        portfolio_returns = np.dot(returns, list(weights_dict.values()))
        var = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
        cvar_returns = portfolio_returns[portfolio_returns <= var]
        cvar = np.mean(cvar_returns) if len(cvar_returns) > 0 else var

        return OptimizationResult(
            weights=weights_dict,
            expected_return=portfolio_return * 252,  # Annualized
            volatility=portfolio_vol * np.sqrt(252),  # Annualized
            sharpe_ratio=sharpe * np.sqrt(252),  # Annualized
            cvar=cvar,
            max_drawdown=self._estimate_max_drawdown(portfolio_returns),
            method_used="CVaR Optimization",
            confidence_level=self.confidence_level,
            total_allocation=1.0,
            rebalance_needed=self._check_rebalance_needed(weights_dict)
        )

    def _optimize_hrp(self, returns_data: pd.DataFrame, signals: Dict[str, Dict],
                      prices: Dict[str, float], cash: float) -> OptimizationResult:
        """
        Hierarchical Risk Parity optimization
        Recommended by ChatGPT for small portfolios to avoid covariance matrix inversion issues
        """

        returns = returns_data.values
        symbols = list(returns_data.columns)

        # Calculate correlation matrix
        corr_matrix = np.corrcoef(returns.T)

        # Calculate distance matrix
        distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))

        # Hierarchical clustering (simplified version)
        # In production, use scipy.cluster.hierarchy for full implementation
        n_assets = len(symbols)

        # Calculate inverse volatility weights
        volatilities = np.std(returns, axis=0)
        inv_vol_weights = (1.0 / volatilities) / np.sum(1.0 / volatilities)

        # Adjust weights based on signal confidence
        signal_multipliers = np.array([
            max(0.5, signals.get(symbol, {}).get('confidence', 0.5))
            for symbol in symbols
        ])

        # Apply signal adjustments
        adjusted_weights = inv_vol_weights * signal_multipliers
        adjusted_weights = adjusted_weights / np.sum(adjusted_weights)

        weights_dict = {symbols[i]: adjusted_weights[i] for i in range(n_assets)}

        # Calculate portfolio metrics
        mean_returns = np.mean(returns, axis=0)
        portfolio_return = np.dot(adjusted_weights, mean_returns)
        portfolio_vol = np.sqrt(np.dot(adjusted_weights, np.dot(np.cov(returns.T), adjusted_weights)))
        sharpe = (portfolio_return - self.risk_free_rate / 252) / portfolio_vol if portfolio_vol > 0 else 0

        # Calculate CVaR
        portfolio_returns = np.dot(returns, adjusted_weights)
        var = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
        cvar_returns = portfolio_returns[portfolio_returns <= var]
        cvar = np.mean(cvar_returns) if len(cvar_returns) > 0 else var

        return OptimizationResult(
            weights=weights_dict,
            expected_return=portfolio_return * 252,
            volatility=portfolio_vol * np.sqrt(252),
            sharpe_ratio=sharpe * np.sqrt(252),
            cvar=cvar,
            max_drawdown=self._estimate_max_drawdown(portfolio_returns),
            method_used="Hierarchical Risk Parity",
            confidence_level=self.confidence_level,
            total_allocation=1.0,
            rebalance_needed=self._check_rebalance_needed(weights_dict)
        )

    def _optimize_signal_weighted(self, returns_data: pd.DataFrame, signals: Dict[str, Dict],
                                  prices: Dict[str, float], cash: float) -> OptimizationResult:
        """
        Signal confidence weighted portfolio
        Allocates based on strategy signal confidence levels
        """

        symbols = list(returns_data.columns)
        returns = returns_data.values

        # Extract confidence scores
        confidences = np.array([
            signals.get(symbol, {}).get('confidence', 0.0)
            for symbol in symbols
        ])

        # Only allocate to symbols with positive confidence
        positive_conf_mask = confidences > 0.0
        if not np.any(positive_conf_mask):
            # No positive signals, use equal weight
            weights = np.ones(len(symbols)) / len(symbols)
        else:
            weights = np.zeros(len(symbols))
            positive_confidences = confidences[positive_conf_mask]
            normalized_confidences = positive_confidences / np.sum(positive_confidences)
            weights[positive_conf_mask] = normalized_confidences

        # Apply position size limits
        weights = np.clip(weights, self.min_weight, self.max_weight)
        weights = weights / np.sum(weights)  # Renormalize

        weights_dict = {symbols[i]: weights[i] for i in range(len(symbols))}

        # Calculate portfolio metrics
        mean_returns = np.mean(returns, axis=0)
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(np.cov(returns.T), weights)))
        sharpe = (portfolio_return - self.risk_free_rate / 252) / portfolio_vol if portfolio_vol > 0 else 0

        # Calculate CVaR
        portfolio_returns = np.dot(returns, weights)
        var = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
        cvar_returns = portfolio_returns[portfolio_returns <= var]
        cvar = np.mean(cvar_returns) if len(cvar_returns) > 0 else var

        return OptimizationResult(
            weights=weights_dict,
            expected_return=portfolio_return * 252,
            volatility=portfolio_vol * np.sqrt(252),
            sharpe_ratio=sharpe * np.sqrt(252),
            cvar=cvar,
            max_drawdown=self._estimate_max_drawdown(portfolio_returns),
            method_used="Signal Weighted",
            confidence_level=self.confidence_level,
            total_allocation=1.0,
            rebalance_needed=self._check_rebalance_needed(weights_dict)
        )

    def _optimize_equal_weight(self, returns_data: pd.DataFrame, signals: Dict[str, Dict],
                               prices: Dict[str, float], cash: float) -> OptimizationResult:
        """Equal weight portfolio (fallback method)"""

        symbols = list(returns_data.columns)
        returns = returns_data.values
        n_assets = len(symbols)

        # Equal weights
        weights = np.ones(n_assets) / n_assets
        weights_dict = {symbols[i]: weights[i] for i in range(n_assets)}

        # Calculate portfolio metrics
        mean_returns = np.mean(returns, axis=0)
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(np.cov(returns.T), weights)))
        sharpe = (portfolio_return - self.risk_free_rate / 252) / portfolio_vol if portfolio_vol > 0 else 0

        # Calculate CVaR
        portfolio_returns = np.dot(returns, weights)
        var = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
        cvar_returns = portfolio_returns[portfolio_returns <= var]
        cvar = np.mean(cvar_returns) if len(cvar_returns) > 0 else var

        return OptimizationResult(
            weights=weights_dict,
            expected_return=portfolio_return * 252,
            volatility=portfolio_vol * np.sqrt(252),
            sharpe_ratio=sharpe * np.sqrt(252),
            cvar=cvar,
            max_drawdown=self._estimate_max_drawdown(portfolio_returns),
            method_used="Equal Weight",
            confidence_level=self.confidence_level,
            total_allocation=1.0,
            rebalance_needed=self._check_rebalance_needed(weights_dict)
        )

    def _adjust_returns_for_signals(self, base_returns: np.ndarray,
                                    symbols: List[str],
                                    signals: Dict[str, Dict]) -> np.ndarray:
        """Adjust expected returns based on current strategy signals"""

        adjusted_returns = base_returns.copy()

        for i, symbol in enumerate(symbols):
            signal = signals.get(symbol, {})
            confidence = signal.get('confidence', 0.0)
            action = signal.get('action', 'HOLD')

            # Adjust expected return based on signal
            if action == 'BUY' and confidence > 0.5:
                # Increase expected return for strong buy signals
                adjusted_returns[i] *= (1 + confidence * 0.5)
            elif action == 'SELL' and confidence > 0.5:
                # Decrease expected return for strong sell signals
                adjusted_returns[i] *= (1 - confidence * 0.3)

        return adjusted_returns

    def _get_signal_based_initial_weights(self, symbols: List[str],
                                          signals: Dict[str, Dict]) -> np.ndarray:
        """Generate initial weights based on signal confidence"""

        n_assets = len(symbols)
        weights = np.ones(n_assets) / n_assets  # Start with equal weights

        # Adjust based on signal confidence
        for i, symbol in enumerate(symbols):
            signal = signals.get(symbol, {})
            confidence = signal.get('confidence', 0.5)
            action = signal.get('action', 'HOLD')

            if action == 'BUY':
                weights[i] *= (1 + confidence)
            elif action == 'SELL':
                weights[i] *= (1 - confidence * 0.5)

        # Normalize and apply bounds
        weights = np.clip(weights, self.min_weight, self.max_weight)
        weights = weights / np.sum(weights)

        return weights

    def _estimate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from return series"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)

    def _check_rebalance_needed(self, target_weights: Dict[str, float]) -> bool:
        """Check if portfolio needs rebalancing"""
        if self.current_state is None:
            return True

        # Compare current weights with target weights
        total_deviation = 0
        for symbol, target_weight in target_weights.items():
            current_weight = self.current_state.weights.get(symbol, 0)
            total_deviation += abs(target_weight - current_weight)

        return total_deviation > self.rebalance_threshold

    def calculate_position_sizes(self, optimization_result: OptimizationResult,
                                 current_prices: Dict[str, float],
                                 available_cash: float,
                                 current_positions: Dict[str, int] = None) -> Dict[str, int]:
        """
        Calculate actual position sizes (number of shares) based on optimization result
        """
        if current_positions is None:
            current_positions = {}

        # Calculate target dollar amounts
        total_portfolio_value = available_cash + sum(
            current_positions.get(symbol, 0) * current_prices.get(symbol, 0)
            for symbol in current_positions.keys()
        )

        target_positions = {}

        for symbol, weight in optimization_result.weights.items():
            if weight < self.min_weight:
                target_positions[symbol] = 0
                continue

            price = current_prices.get(symbol, 0)
            if price <= 0:
                target_positions[symbol] = 0
                continue

            # Calculate target dollar amount
            target_value = weight * total_portfolio_value

            # Calculate target shares (round down to avoid over-allocation)
            target_shares = int(target_value / price)
            target_positions[symbol] = max(0, target_shares)

        return target_positions

    def generate_rebalancing_orders(self, target_positions: Dict[str, int],
                                    current_positions: Dict[str, int],
                                    current_prices: Dict[str, float]) -> List[Dict]:
        """Generate list of orders needed to rebalance portfolio"""

        orders = []

        # Get all symbols
        all_symbols = set(list(target_positions.keys()) + list(current_positions.keys()))

        for symbol in all_symbols:
            current_qty = current_positions.get(symbol, 0)
            target_qty = target_positions.get(symbol, 0)

            if current_qty == target_qty:
                continue

            price = current_prices.get(symbol, 0)
            if price <= 0:
                continue

            # Determine order action and quantity
            if target_qty > current_qty:
                # Buy order
                order_qty = target_qty - current_qty
                action = "BUY"
            else:
                # Sell order
                order_qty = current_qty - target_qty
                action = "SELL"

            orders.append({
                'symbol': symbol,
                'action': action,
                'quantity': order_qty,
                'price': price,
                'order_value': order_qty * price,
                'order_type': 'MARKET'  # Could be enhanced with limit orders
            })

        return orders

    def update_portfolio_state(self, positions: Dict[str, int],
                               cash: float,
                               current_prices: Dict[str, float]):
        """Update current portfolio state for tracking"""

        total_value = cash + sum(
            positions.get(symbol, 0) * current_prices.get(symbol, 0)
            for symbol in positions.keys()
        )

        # Calculate current weights
        weights = {}
        for symbol, qty in positions.items():
            if qty > 0:
                position_value = qty * current_prices.get(symbol, 0)
                weights[symbol] = position_value / total_value if total_value > 0 else 0

        self.current_state = PortfolioState(
            positions=positions,
            cash=cash,
            total_value=total_value,
            weights=weights,
            last_rebalance=datetime.now()
        )

    def get_portfolio_analytics(self, returns_data: pd.DataFrame,
                                current_weights: Dict[str, float]) -> Dict:
        """Calculate comprehensive portfolio analytics"""

        if not current_weights or returns_data.empty:
            return {}

        returns = returns_data.values
        symbols = list(returns_data.columns)

        # Align weights with returns data
        weights = np.array([current_weights.get(symbol, 0) for symbol in symbols])
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights

        # Portfolio returns
        portfolio_returns = np.dot(returns, weights)

        # Risk metrics
        portfolio_vol = np.std(portfolio_returns) * np.sqrt(252)
        portfolio_return = np.mean(portfolio_returns) * 252
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0

        # CVaR calculation
        var = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
        cvar_returns = portfolio_returns[portfolio_returns <= var]
        cvar = np.mean(cvar_returns) if len(cvar_returns) > 0 else var

        # Maximum drawdown
        max_dd = self._estimate_max_drawdown(portfolio_returns)

        # Correlation analysis
        corr_matrix = np.corrcoef(returns.T)
        avg_correlation = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])

        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe,
            'cvar': cvar,
            'var': var,
            'max_drawdown': max_dd,
            'average_correlation': avg_correlation,
            'portfolio_concentration': max(weights) if len(weights) > 0 else 0,
            'effective_number_of_positions': 1 / np.sum(weights ** 2) if np.sum(weights ** 2) > 0 else 0,
            'confidence_level': self.confidence_level
        }


# Example usage integration
if __name__ == "__main__":
    # Example of how to integrate with your working_gui.py

    # Sample data
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')

    # Generate sample returns data
    np.random.seed(42)
    returns_data = pd.DataFrame(
        np.random.normal(0.001, 0.02, (len(dates), len(symbols))),
        index=dates,
        columns=symbols
    )

    # Sample strategy signals
    strategy_signals = {
        'AAPL': {'action': 'BUY', 'confidence': 0.8},
        'GOOGL': {'action': 'BUY', 'confidence': 0.6},
        'MSFT': {'action': 'HOLD', 'confidence': 0.5},
        'TSLA': {'action': 'SELL', 'confidence': 0.7},
        'NVDA': {'action': 'BUY', 'confidence': 0.9}
    }

    # Sample current prices
    current_prices = {symbol: 150 + np.random.uniform(-50, 50) for symbol in symbols}

    # Initialize optimizer
    optimizer = PortfolioOptimizer(confidence_level=0.99)

    # Run optimization
    result = optimizer.optimize_portfolio(
        returns_data=returns_data,
        strategy_signals=strategy_signals,
        current_prices=current_prices,
        available_cash=100000,
        method="cvar_optimal"
    )

    print("Portfolio Optimization Results")
    print("=" * 40)
    print(f"Method: {result.method_used}")
    print(f"Expected Return: {result.expected_return:.2%}")
    print(f"Volatility: {result.volatility:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"CVaR ({result.confidence_level:.0%}): {result.cvar:.2%}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print("\nOptimal Weights:")
    for symbol, weight in result.weights.items():
        if weight > 0.01:  # Only show significant allocations
            print(f"  {symbol}: {weight:.1%}")

    # Calculate position sizes
    positions = optimizer.calculate_position_sizes(
        result, current_prices, 100000
    )

    print(f"\nPosition Sizes:")
    for symbol, shares in positions.items():
        if shares > 0:
            value = shares * current_prices[symbol]
            print(f"  {symbol}: {shares} shares (${value:,.0f})")
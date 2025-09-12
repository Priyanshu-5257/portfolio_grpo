import pandas as pd
import numpy as np

# A very large number to represent a practical limit, preventing overflow
MAX_VALUE = np.finfo(np.float64).max / 10

class PortfolioManager:
    def __init__(self, symbols, initial_cash=1000000.0):
        self.symbols = symbols
        self.initial_cash = np.float64(initial_cash)
        self.cash_balance = self.initial_cash
        self.holdings = self._initialize_holdings()
        self.portfolio_value = pd.DataFrame(columns=['value'])

    def _initialize_holdings(self):
        # Use a numpy array for holdings for numerical stability
        return {symbol: np.float64(0) for symbol in self.symbols}

    def update_holdings(self, symbol, quantity, price):
        # Ensure inputs are stable
        quantity = np.float64(quantity)
        price = np.float64(price)

        # Cap values to prevent overflow before calculation
        self.cash_balance = min(self.cash_balance, MAX_VALUE)
        self.holdings[symbol] = min(self.holdings.get(symbol, 0), MAX_VALUE)

        # Calculate cost safely
        cost = np.multiply(quantity, price)
        if not np.isfinite(cost):
            cost = MAX_VALUE if cost > 0 else -MAX_VALUE

        # Update cash and holdings safely
        self.cash_balance = np.subtract(self.cash_balance, cost)
        self.holdings[symbol] = np.add(self.holdings[symbol], quantity)

        # Final check to prevent infinite values
        if not np.isfinite(self.cash_balance):
            self.cash_balance = MAX_VALUE if self.cash_balance > 0 else 0
        if not np.isfinite(self.holdings[symbol]):
            self.holdings[symbol] = MAX_VALUE

    def record_portfolio_value(self, date, current_prices):
        # Start with cash balance, ensuring it's finite
        value = min(self.cash_balance, MAX_VALUE)

        for symbol, quantity in self.holdings.items():
            # Ensure quantity is finite
            quantity = min(quantity, MAX_VALUE)
            
            price_series = current_prices.get(symbol)
            price = 0.0

            if price_series is not None:
                if isinstance(price_series, (pd.Series, np.ndarray)) and not price_series.empty:
                    price = price_series.item()
                elif isinstance(price_series, (int, float, np.number)):
                    price = price_series
            
            if pd.notna(price) and np.isfinite(price) and price > 0:
                # Calculate asset value safely
                asset_value = np.multiply(quantity, price)
                if np.isfinite(asset_value):
                    value = np.add(value, asset_value)

        # Ensure total value is finite before recording
        if not np.isfinite(value):
            value = MAX_VALUE
            
        new_value_df = pd.DataFrame({'value': [value]}, index=[pd.to_datetime(date)])
        self.portfolio_value = pd.concat([self.portfolio_value, new_value_df])

    def get_holdings(self):
        return self.holdings

    def get_cash_balance(self):
        return self.cash_balance
    
    def get_total_portfolio_value(self, current_prices):
        """
        Calculates the live, real-time total value of the portfolio.

        This is the sum of the current cash balance and the market value
        of all held assets based on the provided current prices.
        
        Args:
            current_prices (pd.Series): A pandas Series with symbols as index
                                        and their latest prices as values.

        Returns:
            float: The total current value of the portfolio.
        """
        market_value = 0.0
        for symbol, quantity in self.holdings.items():
            if quantity > 0:
                price = current_prices.get(symbol, 0)
                if price is not None and pd.notna(price.item()):
                    market_value += quantity * price.item()

        return self.cash_balance + market_value
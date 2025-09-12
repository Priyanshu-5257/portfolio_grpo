import pandas as pd

class ExecutionHandler:
    def __init__(self, commission_rate=0.001, slippage_rate=0.0005):
        """
        Initializes the ExecutionHandler.

        Args:
            commission_rate (float): The commission fee as a fraction of trade value.
            slippage_rate (float): The slippage as a fraction of price for simulating
                                   market impact.
        """
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

    def execute_order(self, signal, current_prices):
        """
        Executes a trade signal, calculating cost and simulating market effects.
        
        Returns:
            A dictionary with the transaction details including total cost, or None.
        """
        symbol = signal['symbol']
        action = signal['action']
        quantity = signal['quantity']
        
        price = self._get_price_from_series(current_prices.get(symbol))
        if price is None:
            return None
        
        # --- Simulate Slippage ---
        # Buys execute at a slightly higher price, Sells at a slightly lower price
        if action == 'BUY':
            execution_price = price * (1 + self.slippage_rate)
        elif action == 'SELL':
            execution_price = price * (1 - self.slippage_rate)
        else:
            return None # Invalid action

        trade_value = quantity * execution_price
        commission = trade_value * self.commission_rate
        
        # Total cost is the trade value plus commission
        total_cost = trade_value + commission

        if action == 'BUY':
            return {
                'symbol': symbol, 
                'quantity': quantity,      # Positive for BUY
                'price': execution_price,
                'cost': total_cost         # <<< The required 'cost' key
            }
        elif action == 'SELL':
            # For sells, quantity is positive to indicate how much was sold.
            # The environment logic handles making this negative for portfolio updates.
            # The "cost" for a sell is the commission paid.
            return {
                'symbol': symbol,
                'quantity': quantity,      # Positive for SELL quantity
                'price': execution_price,
                'cost': commission         # <<< The 'cost' for a SELL is the commission fee
            }
            
        return None

    def _get_price_from_series(self, price_series):
        """Safely extracts a scalar price from a pandas Series or scalar."""
        if price_series is None or price_series.empty:
            return None
        
        try:
            price = price_series.item()
        except (AttributeError, ValueError):
            price = price_series
        
        if pd.isna(price) or price <= 1e-6:
            return None
            
        return price
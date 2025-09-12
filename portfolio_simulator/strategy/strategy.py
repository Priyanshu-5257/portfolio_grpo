from abc import ABC, abstractmethod
import pandas as pd

class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, date, data, portfolio_manager):
        pass

class BuyAndHoldStrategy(Strategy):
    def __init__(self, symbols):
        self.symbols = symbols
        self.invested = False

    def generate_signals(self, date, data, portfolio_manager):
        signals = []
        if not self.invested:
            cash_per_asset = portfolio_manager.get_cash_balance() / len(self.symbols)
            for symbol in self.symbols:
                price_series = data.get(symbol)
                if price_series is None or price_series.empty:
                    continue
                price = price_series.item()
                if pd.notna(price) and price > 0:
                    quantity = int(cash_per_asset / price)
                    if quantity > 0:
                        signals.append({'symbol': symbol, 'action': 'BUY', 'quantity': quantity})
            self.invested = True
        return signals

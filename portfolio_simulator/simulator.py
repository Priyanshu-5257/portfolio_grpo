from data.data_handler import DataHandler
from portfolio.portfolio_manager import PortfolioManager
from strategy.strategy import BuyAndHoldStrategy
from execution.execution_handler import ExecutionHandler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_simulation():
    # --- Configuration ---
    assets_filepath = 'portfolio_simulator/assets.txt'
    start_date = '2004-01-01'
    end_date = '2024-01-01'
    initial_cash = 1000000.0

    # --- Initialization ---
    data_handler = DataHandler(assets_filepath, start_date, end_date)
    symbols = data_handler.symbols
    portfolio_manager = PortfolioManager(symbols, initial_cash)
    strategy = BuyAndHoldStrategy(symbols)
    execution_handler = ExecutionHandler()

    # --- Simulation Loop ---
    print("\nStarting simulation...")
    for date, prices in data_handler.stream_data():
        # 1. Generate signals from the strategy
        signals = strategy.generate_signals(date, prices, portfolio_manager)

        # 2. Execute signals
        for signal in signals:
            transaction = execution_handler.execute_order(signal, prices)
            if transaction:
                # 3. Update portfolio
                portfolio_manager.update_holdings(
                    transaction['symbol'],
                    transaction['quantity'],
                    transaction['price']
                )

        # 4. Record portfolio value for the day
        portfolio_manager.record_portfolio_value(date, prices)

    print("Simulation finished.")

    # --- Results ---
    print("\n--- Simulation Results ---")
    final_value = portfolio_manager.portfolio_value['value'].iloc[-1].item()
    print(f"Initial Portfolio Value: ${initial_cash:,.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    
    total_return = (final_value / initial_cash - 1) * 100
    print(f"Total Return: {total_return:.2f}%")

    # Plotting the results
    plt.figure(figsize=(12, 8))
    plt.plot(portfolio_manager.portfolio_value.index, portfolio_manager.portfolio_value['value'])
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.savefig('portfolio_value.png')
    print("\nPortfolio performance chart saved to portfolio_value.png")


if __name__ == "__main__":
    run_simulation()

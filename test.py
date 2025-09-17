# file: test_advanced.py

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
# Import the necessary classes
from portfolio_simulator.rl_agent.environment import PortfolioEnv
from portfolio_simulator.rl_agent.grpo_agent import GRPOAgent

# --- Configuration ---
MODEL_PATH = "GRPO_models_SS_M4/GRPO_portfolio_51552.pth" 
TEST_START_DATE = '2024-01-01'
TEST_END_DATE = '2024-08-01'
ASSETS_FILEPATH = 'portfolio_simulator/assets.txt'
INITIAL_CASH = 5000.0
LOOKBACK_WINDOW = 60
MAX_TEST_DAYS = (datetime.strptime(TEST_END_DATE, '%Y-%m-%d') - datetime.strptime(TEST_START_DATE, '%Y-%m-%d')).days

# --- Output Configuration ---
OUTPUT_DIR = "test_results"
DAILY_LOG_FILE = os.path.join(OUTPUT_DIR, "daily_log.csv")
TRANSACTION_LOG_FILE = os.path.join(OUTPUT_DIR, "transaction_log.csv")
PLOT_FILE = os.path.join(OUTPUT_DIR, "advanced_performance.png")

# --- Main Test Logic ---

def load_agent(checkpoint_path, env, device):
    """Loads a trained GRPO agent from a checkpoint."""
    agent = GRPOAgent(env.state_space_dim, env.action_space_dim, 0.0, 0.0, 0, 0.0, 0.6, device, 0.0)
    try:
        agent.load(checkpoint_path)
        print(f"Successfully loaded agent from: {checkpoint_path}")
    except Exception:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        print("Successfully loaded policy network weights only.")
    agent.policy.eval()
    return agent

def run_test_and_log(agent, env, device):
    """Runs a deterministic test episode and logs detailed information."""
    state = env.reset()
    done = False
    
    daily_logs = []
    transaction_logs = []

    print("Running test and generating detailed logs...")
    step_count = 0
    while not done:
        action, _ = agent.select_action(state, deterministic=True)
        state, _, done, info = env.step(action)
        step_count += 1

        # Log daily data for every step that has info
        if info:
            daily_info = {
                'date': info['current_date'],
                'portfolio_value': info['final_portfolio_value'],
                'cash': info['cash'],
                'assets_value': info['assets_value'],
                'daily_return_pct': info['daily_return_ratio'] * 100
            }
            daily_logs.append(daily_info)

            # Log transaction data
            for trx in info.get('transactions', []):
                trx_info = {
                    'date': info['current_date'],
                    'symbol': trx['symbol'],
                    'action': trx['action'],
                    'quantity': trx['quantity'],
                    'price': trx['price'],
                    'cost': trx.get('cost', trx['quantity'] * trx['price'])
                }
                transaction_logs.append(trx_info)
    
    print("Test finished. Saving logs...")
    # Create DataFrames and save to CSV
    daily_df = pd.DataFrame(daily_logs)
    daily_df.set_index('date', inplace=True)
    
    trx_df = pd.DataFrame(transaction_logs)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    daily_df.to_csv(DAILY_LOG_FILE)
    trx_df.to_csv(TRANSACTION_LOG_FILE, index=False)
    
    print(f"Daily log saved to {DAILY_LOG_FILE}")
    print(f"Transaction log saved to {TRANSACTION_LOG_FILE}")
    
    return daily_df

def analyze_and_plot(daily_df, initial_cash):
    """Generates advanced analytics and plots."""
    if daily_df.empty:
        print("No data for analysis.")
        return

    print(f"Data points available: {len(daily_df)}")
    print(f"Date range: {daily_df.index.min().date()} to {daily_df.index.max().date()}")

    # --- Calculate Metrics ---
    final_value = daily_df['portfolio_value'].iloc[-1]
    total_return_pct = ((final_value / initial_cash) - 1) * 100
    
    # Clean and filter daily returns (remove zeros and outliers)
    returns_series = daily_df['daily_return_pct']
    non_zero_returns = returns_series[returns_series != 0.0]
    
    # Sharpe Ratio
    if len(non_zero_returns) > 1:
        mean_return = non_zero_returns.mean()
        std_return = non_zero_returns.std()
        sharpe_ratio = (mean_return / (std_return + 1e-9)) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0
    
    # Max Drawdown
    cumulative_returns = (1 + daily_df['daily_return_pct'] / 100).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown_pct = drawdown.min() * 100
    
    # Print summary
    print("\n--- Performance Metrics ---")
    print(f"Test Period: {daily_df.index.min().date()} to {daily_df.index.max().date()}")
    print(f"Initial Portfolio Value: ${initial_cash:,.2f}")
    print(f"Final Portfolio Value:   ${final_value:,.2f}")
    print(f"Total Return:            {total_return_pct:.2f}%")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown:            {max_drawdown_pct:.2f}%")
    print(f"Non-zero return days:    {len(non_zero_returns)}")
    print("---------------------------\n")

    # --- Generate Plots ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
    fig.suptitle('Advanced GRPO Agent Performance Analysis', fontsize=18)

    # 1. Equity Curve and Cash/Assets Allocation
    ax1.plot(daily_df.index, daily_df['portfolio_value'], label='Total Portfolio Value', color='blue', lw=2)
    ax1.stackplot(daily_df.index, daily_df['cash'], daily_df['assets_value'], 
                  labels=['Cash', 'Assets Value'], colors=['#CCCCCC', '#5DADE2'], alpha=0.6)
    ax1.axhline(y=initial_cash, color='grey', linestyle='--', label=f'Initial Cash')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title('Portfolio Value & Asset Allocation')
    ax1.legend(loc='upper left')
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # 2. Drawdown Plot
    ax2.plot(drawdown.index, drawdown * 100, label='Drawdown', color='red', lw=1.5)
    ax2.fill_between(drawdown.index, drawdown * 100, 0, color='red', alpha=0.3)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_title('Portfolio Drawdown')
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.0f}%'))

    # 3. Daily Returns Plot
    ax3.bar(daily_df.index, daily_df['daily_return_pct'], color=np.where(daily_df['daily_return_pct'] < 0, 'crimson', 'forestgreen'))
    ax3.set_ylabel('Return (%)')
    ax3.set_title('Daily Returns')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.1f}%'))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(PLOT_FILE)
    print(f"Advanced plot saved to {PLOT_FILE}")
    plt.show()

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        return

    device = torch.device('cpu')
    
    # Setup Environment
    env = PortfolioEnv(
        assets_filepath=ASSETS_FILEPATH, start_date=TEST_START_DATE, end_date=TEST_END_DATE,
        initial_cash=INITIAL_CASH, lookback_window=LOOKBACK_WINDOW,
        randomize_start=False, min_episode_days=1
    )
    env.max_days = MAX_TEST_DAYS

    # Load Agent
    agent = load_agent(MODEL_PATH, env, device)
    if not agent: return

    # Run Test and Logging
    daily_results_df = run_test_and_log(agent, env, device)
    
    # Analyze and Plot
    analyze_and_plot(daily_results_df, INITIAL_CASH)

if __name__ == "__main__":
    main()
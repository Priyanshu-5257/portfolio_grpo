import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
# Import the necessary classes from your project structure
from portfolio_simulator.rl_agent.environment import PortfolioEnv
from portfolio_simulator.rl_agent.grpo_agent import GRPOAgent
import matplotlib
matplotlib.use('Agg')
# --- Configuration ---

# Path to the trained model checkpoint you want to test
MODEL_PATH = "GRPO_models_SS_M4/GRPO_portfolio_51552.pth" #<-- CHANGE THIS to your best model

# Testing Period (should ideally be data the model has not seen during training)
TEST_START_DATE = '2024-01-01'
TEST_END_DATE = '2025-01-01'

# Environment and Agent parameters (should match the training configuration)
ASSETS_FILEPATH = 'portfolio_simulator/assets.txt'
INITIAL_CASH = 10000.0
LOOKBACK_WINDOW = 60
RANDOMIZE_START_DURING_TEST = False # For reproducibility, test from a fixed start date
MIN_EPISODE_DAYS = 250 # Should be long enough for the test period
MAX_TEST_DAYS = (datetime.strptime(TEST_END_DATE, '%Y-%m-%d') - datetime.strptime(TEST_START_DATE, '%Y-%m-%d')).days

# --- Helper Functions ---

def load_agent(checkpoint_path, env, device):
    """Loads a trained GRPO agent from a checkpoint."""
    # These hyperparameters don't affect inference, but are needed for object creation
    lr_actor = 0.0
    gamma = 0.0
    K_epochs = 0
    eps_clip = 0.0
    action_std_init = 0.6 # Use the final min_action_std for deterministic evaluation
    beta_kl = 0.0
    
    agent = GRPOAgent(env.state_space_dim, env.action_space_dim, lr_actor, gamma, K_epochs, eps_clip, action_std_init, device, beta_kl)
    
    try:
        agent.load(checkpoint_path)
        print(f"Successfully loaded agent from: {checkpoint_path}")
    except Exception as e:
        print(f"Error loading agent: {e}")
        # Try loading only the policy network if the full checkpoint fails
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            agent.policy.load_state_dict(checkpoint['policy_state_dict'])
            print("Successfully loaded policy network weights only.")
        except Exception as e_inner:
            print(f"Could not load policy weights either: {e_inner}")
            return None
    
    # Set the agent's policy to evaluation mode
    agent.policy.eval()
    return agent

def run_test_episode(agent, env, device):
    """Runs a single episode with the agent in a deterministic mode."""
    state = env.reset()
    done = False
    
    while not done:
        # Convert state to tensor for the agent
        state_tensor = torch.FloatTensor(state).to(device)
        
        # Get a deterministic action from the agent's policy
        # We don't need the logprob for testing
        action, _ = agent.select_action(state, deterministic=True)
        
        # Take a step in the environment
        state, _, done, info = env.step(action)

    print("Test episode finished.")
    return info.get('history') # The 'history' DataFrame is returned when done

def plot_performance(history_df, initial_cash):
    """Generates and displays a plot of portfolio value over time."""
    if history_df is None or history_df.empty:
        print("No history data to plot.")
        return

    # Set up the plot
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plot portfolio value
    ax.plot(history_df.index, history_df['value'], label='Agent Portfolio Value', color='royalblue', linewidth=2)
    
    # Plot baseline for starting cash
    ax.axhline(y=initial_cash, color='grey', linestyle='--', label=f'Initial Cash (${initial_cash:,.2f})')

    # Formatting
    ax.set_title('GRPO Agent Portfolio Performance (Test Period)', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    
    # Use a currency formatter for the y-axis
    formatter = plt.FuncFormatter(lambda x, pos: f'${x:,.0f}')
    ax.yaxis.set_major_formatter(formatter)
    
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plot_filename = "test_results/grpo_test_performance.png"
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")

    # Show the plot
    plt.show()

def calculate_metrics(history_df, initial_cash, days):
    """Calculates and prints key performance metrics."""
    if history_df is None or len(history_df) < 2:
        print("Not enough data to calculate metrics.")
        return
        
    final_value = history_df['value'].iloc[-1]
    total_return_pct = ((final_value / initial_cash) - 1) * 100
    
    daily_returns = history_df['value'].pct_change().dropna()
    
    # Annualized Sharpe Ratio (assuming 252 trading days in a year)
    sharpe_ratio = (daily_returns.mean() / (daily_returns.std() + 1e-8)) * np.sqrt(252)
    
    # Max Drawdown
    cumulative_returns = (1 + daily_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown_pct = drawdown.min() * 100

    print("\n--- Performance Metrics ---")
    print(f"Test Period: {history_df.index.min().date()} to {history_df.index.max().date()}")
    print(f"Initial Portfolio Value: ${initial_cash:,.2f}")
    print(f"Final Portfolio Value:   ${final_value:,.2f}")
    print(f"Total Return:            {total_return_pct:.2f}%")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown:            {max_drawdown_pct:.2f}%")
    print("---------------------------\n")

def main():
    """Main function to run the test."""
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        return

    device = torch.device('cpu') # Use CPU for inference, it's usually sufficient
    print(f"Using device: {device}")

    # 1. Set up the test environment
    print("Setting up test environment...")
    test_env = PortfolioEnv(
        assets_filepath=ASSETS_FILEPATH,
        start_date=TEST_START_DATE,
        end_date=TEST_END_DATE,
        initial_cash=INITIAL_CASH,
        lookback_window=LOOKBACK_WINDOW,
        randomize_start=RANDOMIZE_START_DURING_TEST,
        min_episode_days=MIN_EPISODE_DAYS,
    )
    test_env.max_days = MAX_TEST_DAYS # Ensure the episode runs for the full test period

    # 2. Load the trained agent
    print("Loading trained GRPO agent...")
    agent = load_agent(MODEL_PATH, test_env, device)
    
    if agent is None:
        print("Failed to load agent. Exiting.")
        return

    # 3. Run the test
    print("Running test episode...")
    portfolio_history = run_test_episode(agent, test_env, device)
    
    # 4. Calculate metrics and plot results
    if portfolio_history is not None:
        calculate_metrics(portfolio_history, INITIAL_CASH, len(portfolio_history))
        plot_performance(portfolio_history, INITIAL_CASH)
    else:
        print("Episode did not produce a valid history.")


if __name__ == "__main__":
    main()
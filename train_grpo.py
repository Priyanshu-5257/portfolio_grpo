# file: train.py

import os
import time
import pandas as pd
import warnings
import torch
import numpy as np
from portfolio_simulator.rl_agent.grpo_environment import SingleStateGRPOEnv # <-- IMPORT THE NEW WRAPPER
from portfolio_simulator.rl_agent.grpo_agent import GRPOAgent

warnings.filterwarnings("ignore", category=FutureWarning)

def train():
    print("============================================================================================")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("============================================================================================")

    # --- Environment Configuration ---
    assets_filepath = 'portfolio_simulator/assets.txt'
    start_date = '2004-01-01'
    end_date = '2020-01-01'
    initial_cash = 100000.0
    group_size = 16 
    env = SingleStateGRPOEnv(group_size, assets_filepath, start_date, end_date, initial_cash)

    # --- GRPO Agent Hyperparameters ---
    K_epochs = 20                   # mu: Gradient updates per training cycle
    eps_clip = 0.2
    gamma = 0.99
    lr_actor = 1e-5                
    action_std_init = 0.6          
    beta_kl = 0.01                  
    min_action_std = 0.60         
    action_std_decay_rate = 0.05   
    
    agent = GRPOAgent(env.state_space_dim, env.action_space_dim, lr_actor, gamma, K_epochs, eps_clip, action_std_init, device, beta_kl)

    # ### MODIFIED ###: Training loop configuration
    max_training_timesteps = int(3e6)
    save_model_freq = 50000
    log_freq = 1 # Log after every major training update

    # --- HYPERPARAMETER M ---w2
    # M: Number of data collection cycles to run before a training update.
    data_collection_cycles_M = 4 

    # --- Logging Setup ---
    log_dir, model_dir = "GRPO_logs_SS_M4", "GRPO_models_SS_M4"
    os.makedirs(log_dir, exist_ok=True); os.makedirs(model_dir, exist_ok=True)
    summary_log_path = os.path.join(log_dir, "training_summary.csv")
    if not os.path.exists(summary_log_path):
        with open(summary_log_path, "w") as f:
            f.write("training_cycle,total_timesteps,avg_batch_reward,loss,kl_divergence,entropy\n")

    print("Starting SINGLE-STATE, MULTI-PATH GRPO Training (with M > 1)...")
    print(f"Hyperparameters: Group Size G={group_size}, Collection Cycles M={data_collection_cycles_M}, Update Epochs K={K_epochs}")
    print("============================================================================================")

    time_step = 0
    training_update_cycle = 0 # This now counts the number of times we call agent.train()
    env.reset()

    # ### MODIFIED ###: New Training Loop Structure
    while time_step <= max_training_timesteps:
        
        # --- I-Loop (Major Iteration / Training Update Cycle) ---
        training_update_cycle += 1
        
        # 1. UPDATE REFERENCE POLICY
        agent.reference_policy.load_state_dict(agent.policy.state_dict())
        
        # --- M-Loop (Data Collection) ---
        cycle_group_rewards, cycle_timesteps_collected = [], 0
        
        for m_step in range(data_collection_cycles_M):
            # Advance base env by one step to get a new 'prompt' state
            action_for_step, _ = agent.select_action(env.base_env._get_state())
            _, _, done, _ = env.base_env.step(action_for_step)
            if done: env.base_env.reset()

            # Collect data for one group and fill the agent's buffer
            group_stats = env.collect_group_data_into_buffer(agent) 
            
            # Store stats from this collection step for aggregated logging
            cycle_group_rewards.append(group_stats['group_mean_reward'])
            cycle_timesteps_collected += group_stats['timesteps_this_cycle']
        
        time_step += cycle_timesteps_collected
        
        # 2. TRAIN ON ACCUMULATED DATA (M*G trajectories)
        training_stats = agent.train() # The buffer now contains M batches of data

        # 3. LOGGING (after each major training update)
        if training_update_cycle % log_freq == 0:
            avg_batch_reward = np.mean(cycle_group_rewards) if cycle_group_rewards else 0
            loss = training_stats.get('policy_loss', 0)
            kl = training_stats.get('kl_divergence', 0)
            entropy = training_stats.get('entropy', 0)

            print(f"Cycle {training_update_cycle} | T: {time_step} | Avg Batch R: {avg_batch_reward:.2f} | KL: {kl:.4f} | Loss: {loss:.6f}")
            
            with open(summary_log_path, "a") as f:
                f.write(f"{training_update_cycle},{time_step},{avg_batch_reward},{loss},{kl},{entropy}\n")

        # 4. SAVING and DECAY
        if time_step // save_model_freq > (time_step - cycle_timesteps_collected) // save_model_freq:
            print("--------------------------------------------------------------------------------------------")
            print(f"Saving model at timestep {time_step}")
            agent.save(os.path.join(model_dir, f"GRPO_portfolio_{time_step}.pth"))
            print("--------------------------------------------------------------------------------------------")
        
        agent.set_action_std(max(min_action_std, agent.action_std - (action_std_decay_rate/max_training_timesteps) * cycle_timesteps_collected))

if __name__ == "__main__":
    train()
# file: portfolio_simulator/rl_agent/grpo_env_wrapper.py

import numpy as np
import copy
import torch
from .environment import PortfolioEnv

class SingleStateGRPOEnv:
    """
    An environment wrapper to implement the "single-state, multi-path" GRPO paradigm.
    Manages a single "base" environment to generate states and can run group simulations.
    """
    def __init__(self, group_size, assets_filepath, start_date, end_date, initial_cash):
        self.group_size = group_size
        self.env_params = {
            'assets_filepath': assets_filepath, 'start_date': start_date, 
            'end_date': end_date, 'initial_cash': initial_cash
        }
        self.base_env = PortfolioEnv(**self.env_params)
        self.state_space_dim = self.base_env.state_space_dim
        self.action_space_dim = self.base_env.action_space_dim
        
    def reset(self):
        """Resets the base environment and returns the initial state."""
        return self.base_env.reset()

    def _run_single_path(self, agent, start_env, first_action):
        """
        Runs one full simulation path from a given starting environment state.
        (This function is unchanged.)
        """
        env, states, actions, logprobs, rewards = start_env, [], [], [], []
        state = env._get_state()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(agent.device)
            action_tensor = torch.FloatTensor(first_action).to(agent.device)
            logprob_tensor, _ = agent.policy.evaluate(state_tensor.unsqueeze(0), action_tensor.unsqueeze(0))
            logprob = logprob_tensor.item()
        
        states.append(state)
        actions.append(first_action)
        logprobs.append(logprob)
        next_state, reward, done, info = env.step(first_action)
        rewards.append(reward)

        while not done:
            states.append(next_state)
            action, logprob = agent.select_action(next_state)
            actions.append(action)
            logprobs.append(logprob if logprob is not None else 0.0)
            next_state, reward, done, info = env.step(action)
            rewards.append(reward)

        return {'states': states, 'actions': actions, 'logprobs': logprobs, 'rewards': rewards,
                'total_reward': sum(rewards), 'episode_length': len(rewards), 'info': info}

    ### NEW ###
    def collect_group_data_into_buffer(self, agent):
        """
        Performs the data collection part of a cycle:
        1. Runs a group of simulations from the current base_env state.
        2. Calculates advantages and fills the agent's buffer.
        DOES NOT train the agent. Returns stats for logging.
        """
        current_state = self.base_env._get_state()
        group_first_actions, _ = zip(*agent.generate_group_actions(current_state, self.group_size))

        group_data = []
        for first_action in group_first_actions:
            cloned_start_env = copy.deepcopy(self.base_env)
            episode_data = self._run_single_path(agent, cloned_start_env, first_action)
            group_data.append(episode_data)
        
        group_total_rewards = [ep['total_reward'] for ep in group_data]
        group_advantages = agent.calculate_group_advantages(group_total_rewards)
        
        timesteps_in_cycle = 0
        for episode_data, advantage in zip(group_data, group_advantages):
            ep_len = episode_data['episode_length']
            timesteps_in_cycle += ep_len
            
            ep_states = [torch.FloatTensor(s).to(agent.device) for s in episode_data['states']]
            ep_actions = [torch.FloatTensor(a).to(agent.device) for a in episode_data['actions']]
            ep_logprobs = [torch.FloatTensor([lp]).to(agent.device) for lp in episode_data['logprobs']]
            
            ep_advantages = [advantage] * ep_len
            agent.buffer.store_group_episode(ep_states, ep_actions, ep_logprobs, episode_data['rewards'], ep_advantages)
        
        stats_to_return = {
            'group_mean_reward': np.mean(group_total_rewards),
            'group_max_reward': np.max(group_total_rewards),
            'timesteps_this_cycle': timesteps_in_cycle
        }
        return stats_to_return

    ### MODIFIED ###
    def step_and_train_on_group(self, agent):
        """
        DEPRECATED: The training loop in train.py now orchestrates collection and training.
        """
        raise DeprecationWarning("step_and_train_on_group is deprecated. "
                               "Use the M-loop in train.py and call collect_group_data_into_buffer instead.")
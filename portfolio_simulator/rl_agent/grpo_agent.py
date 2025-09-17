import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import numpy as np
import copy

class SwiGLU(nn.Module):
    """ SwiGLU Activation Function """
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

class GRPOActorNetwork(nn.Module):
    """
    Actor-only network for GRPO using the same hybrid CNN-MLP architecture.
    Removes the critic head from the original ActorCritic.
    """
    def __init__(self, state_dim, action_dim, action_std_init, device, n_assets=13, lookback_window=60):
        super(GRPOActorNetwork, self).__init__()

        self.device = device
        self.n_assets = n_assets
        self.lookback_window = lookback_window
        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # --- CNN Branch for Market Data (same as PPO) ---
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(in_channels=n_assets, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Calculate the flattened size after CNN
        cnn_output_size = 128 * lookback_window

        # --- MLP Branch for Vector Data (Cash + Holdings) ---
        vector_input_dim = 1 + n_assets
        self.mlp_branch = nn.Sequential(
            nn.Linear(vector_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )

        # --- Combined Network with SwiGLU (Actor Head Only) ---
        combined_dim = cnn_output_size + 128
        
        # Actor Head (same as PPO)
        self.actor_head = nn.Sequential(
            nn.Linear(combined_dim, 256 * 2), # Double size for SwiGLU
            SwiGLU(),
            nn.Linear(256, 256 * 2), # Double size for SwiGLU
            SwiGLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def _split_state(self, state):
        # State shape: (batch_size, 1 + n_assets + n_assets * lookback_window)
        vector_data = state[:, :1 + self.n_assets]
        market_data = state[:, 1 + self.n_assets:]
        market_data = market_data.view(-1, self.n_assets, self.lookback_window)
        return vector_data, market_data

    def forward(self, state):
        # Split the input state
        vector_data, market_data = self._split_state(state)

        # Process through branches
        cnn_features = self.cnn_branch(market_data)
        mlp_features = self.mlp_branch(vector_data)

        # Combine features
        combined_features = torch.cat((cnn_features, mlp_features), dim=1)

        # Get action mean (no critic value)
        action_mean = self.actor_head(combined_features)
        
        return action_mean

    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)

    def act(self, state, deterministic=False):
        # Add batch dimension if it's a single state
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        action_mean = self.forward(state)
        
        if deterministic:
            return action_mean.detach(), None
            
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_mean = self.forward(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, dist_entropy

class GRPOBuffer:
    """
    Buffer for storing group trajectories and advantages for GRPO.
    """
    def __init__(self):
        self.clear()

    def store_group_episode(self, states, actions, logprobs, rewards, group_advantages):
        """
        Store a complete group episode.
        
        Args:
            states: List of state tensors for the episode
            actions: List of action tensors for the episode  
            logprobs: List of log probability tensors
            rewards: List of rewards (scalar values)
            group_advantages: List of advantage values (calculated from group)
        """
        self.states.extend(states)
        self.actions.extend(actions)
        self.logprobs.extend(logprobs)
        self.rewards.extend(rewards)
        self.advantages.extend(group_advantages)

    def get_batch(self):
        """
        Return all stored data as tensors.
        """
        if len(self.states) == 0:
            return None, None, None, None
            
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        logprobs = torch.stack(self.logprobs).squeeze(-1)  # Remove extra dimension
        advantages = torch.tensor(self.advantages, dtype=torch.float32)
        
        return states, actions, logprobs, advantages

    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.advantages = []

class GRPOAgent:
    """
    GRPO Agent that uses group-relative advantages instead of a critic network.
    (This class is already correctly implemented for batch training and needs no changes.)
    """
    def __init__(self, state_dim, action_dim, lr_actor, gamma, K_epochs, eps_clip, action_std_init, device, beta_kl=0.01):
        self.device = device
        self.action_std = action_std_init
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.beta_kl = beta_kl  # KL divergence coefficient
        
        self.buffer = GRPOBuffer()

        # Initialize the actor network
        self.policy = GRPOActorNetwork(state_dim, action_dim, action_std_init, device).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)

        # Reference policy for KL divergence (frozen copy of initial policy)
        self.reference_policy = GRPOActorNetwork(state_dim, action_dim, action_std_init, device).to(device)
        self.reference_policy.load_state_dict(self.policy.state_dict())
        
        # Freeze reference policy
        for param in self.reference_policy.parameters():
            param.requires_grad = False

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)

    def select_action(self, state, deterministic=False):
        """Select action for a single state."""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob = self.policy.act(state, deterministic)

        if action_logprob is not None:
            return action.cpu().numpy().flatten(), action_logprob.cpu().item()
        else:
            return action.cpu().numpy().flatten(), None

    def select_actions_for_vec(self, states):
        """Select actions for a batch of states from vectorized environments."""
        with torch.no_grad():
            states = torch.FloatTensor(states).to(self.device)
            actions, action_logprobs = self.policy.act(states)

        return actions.cpu().numpy()

    def generate_group_actions(self, state, group_size=4):
        """
        Generate multiple diverse actions for the same state.
        """
        actions_and_logprobs = []
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            for _ in range(group_size):
                action, logprob = self.policy.act(state_tensor, deterministic=False)
                actions_and_logprobs.append((
                    action.cpu().numpy().flatten(),
                    logprob.cpu().item() if logprob is not None else None
                ))
        return actions_and_logprobs

    def calculate_group_advantages(self, group_rewards, normalize=True):
        """
        Calculate group-relative advantages from episode rewards.
        """
        group_rewards = np.array(group_rewards)
        if normalize and len(group_rewards) > 1:
            mean_reward = np.mean(group_rewards)
            std_reward = np.std(group_rewards) + 1e-8
            advantages = (group_rewards - mean_reward) / std_reward
        else:
            mean_reward = np.mean(group_rewards)
            advantages = group_rewards - mean_reward
        return advantages.tolist()

    def train(self):
        """
        Train the policy using GRPO algorithm.
        """
        if len(self.buffer.states) == 0:
            return {}

        batch_data = self.buffer.get_batch()
        if batch_data[0] is None: return {}
            
        old_states, old_actions, old_logprobs, advantages = batch_data
        old_states, old_actions, old_logprobs, advantages = \
            old_states.to(self.device), old_actions.to(self.device), old_logprobs.to(self.device), advantages.to(self.device)

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        training_stats = {'policy_loss': [], 'kl_divergence': [], 'entropy': [], 'total_loss': []}

        for _ in range(self.K_epochs):
            logprobs, entropy = self.policy.evaluate(old_states, old_actions)
            ratios = torch.exp(logprobs - old_logprobs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            with torch.no_grad():
                ref_logprobs, _ = self.reference_policy.evaluate(old_states, old_actions)
            kl_div = torch.mean(ref_logprobs-logprobs) 
            total_loss = policy_loss + self.beta_kl * kl_div - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            training_stats['policy_loss'].append(policy_loss.item())
            training_stats['kl_divergence'].append(kl_div.item())
            training_stats['entropy'].append(entropy.mean().item())
            training_stats['total_loss'].append(total_loss.item())

        self.buffer.clear()
        return {key: np.mean(values) for key, values in training_stats.items()}

    def save(self, checkpoint_path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'action_std': self.action_std,
        }, checkpoint_path)

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.action_std = checkpoint['action_std']
        self.policy.set_action_std(self.action_std)

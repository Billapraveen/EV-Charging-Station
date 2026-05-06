import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque
from schedulers import get_scheduler

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Action definitions ────────────────────────────────────────────────────────
# Include the new priority scheduler and LLF
ACTION_NAMES = ["FIFO", "SJF", "EDF", "LLF", "Priority"]
N_ACTIONS = len(ACTION_NAMES)

def extract_state(obs):
    """
    Extracts continuous state features from observation dict.
    obs keys: t, hour, queue_len, free_ports, active_count, avg_slack, load, load_norm
    """
    return np.array([
        obs["hour"] / 24.0,           # Normalized hour
        obs["queue_len"],
        obs["free_ports"],
        obs["active_count"],
        obs["avg_slack"],
        obs["load_norm"]              # Normalized site load
    ], dtype=np.float32)

STATE_DIM = 6

# ── Deep Q-Network Architecture ───────────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ── Replay Buffer ─────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.stack(states), actions, rewards, np.stack(next_states), dones
    
    def __len__(self):
        return len(self.buffer)

# ── DQN Agent ─────────────────────────────────────────────────────────────────
class DQNAgent:
    """
    Deep Q-Network Agent for EV charging scheduling.
    """
    def __init__(self, gamma=0.95, lr=1e-3, batch_size=64, capacity=10000, 
                 epsilon=1.0, eps_min=0.01, eps_decay=0.995, seed=42):
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.seed = seed
        
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Networks
        self.policy_net = DQN(STATE_DIM, N_ACTIONS).to(device)
        self.target_net = DQN(STATE_DIM, N_ACTIONS).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(capacity)
        self.loss_fn = nn.MSELoss()
        
        # Action mappings
        self.actions = {i: get_scheduler(name) for i, name in enumerate(ACTION_NAMES)}

        # Training logs
        self.episode_rewards = []
        self.episode_wait_times = []
        self.episode_completions = []

    def select_action(self, obs, greedy=False):
        """epsilon-greedy action selection."""
        state = extract_state(obs)
        if not greedy and random.random() < self.epsilon:
            a_idx = random.randrange(N_ACTIONS)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_t)
                a_idx = q_values.argmax(dim=1).item()
                
        return a_idx, self.actions[a_idx]

    def update(self):
        """Sample from replay buffer and update network."""
        if len(self.memory) < self.batch_size:
            return
            
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)
        
        # Compute Q(s, a)
        q_values = self.policy_net(states).gather(1, actions)
        
        # Compute max Q(s', a)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            
        # Target = r + gamma * max_a Q(s', a) * (1 - done)
        targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        loss = self.loss_fn(q_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def update_target_network(self):
        """Copy weights from policy net to target net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    # ── Training loop ─────────────────────────────────────────────────────────
    def train(self, sim_factory, n_episodes=300, target_update_freq=5, verbose=True):
        print(f"\n{'='*50}")
        print(f"DQN TRAINING  ({n_episodes} episodes) ON {device}")
        print(f"gamma={self.gamma}, epsilon_0={self.epsilon}, decay={self.eps_decay}")
        print(f"Actions: {ACTION_NAMES}")
        print(f"{'='*50}\n")

        for ep in range(1, n_episodes + 1):
            sim = sim_factory()
            obs = sim.reset()
            done = False
            ep_reward = 0.0

            while not done:
                a_idx, action_fn = self.select_action(obs)
                next_obs, reward, done, _ = sim.step(action_fn)
                
                state = extract_state(obs)
                next_state = extract_state(next_obs)
                
                self.memory.push(state, a_idx, reward, next_state, done)
                self.update()
                
                obs = next_obs
                ep_reward += reward

            if ep % target_update_freq == 0:
                self.update_target_network()

            metrics = sim.get_metrics()
            self.episode_rewards.append(ep_reward)
            self.episode_wait_times.append(metrics["avg_waiting_time_min"])
            self.episode_completions.append(metrics["completion_rate_pct"])
            self.decay_epsilon()

            if verbose and ep % 10 == 0:
                avg_r = np.mean(self.episode_rewards[-10:])
                avg_w = np.mean(self.episode_wait_times[-10:])
                print(f"  Ep {ep:4d}/{n_episodes} | "
                      f"Avg reward: {avg_r:8.1f} | "
                      f"Avg wait: {avg_w:6.1f} min | "
                      f"eps={self.epsilon:.3f}")

        print("\nTraining complete.")
        return self

    def greedy_policy(self):
        """Callable policy for evaluation."""
        agent = self
        def _policy(queue, free_ports, t):
            obs = {
                "t": t,
                "hour": (t * 5 / 60) % 24,
                "queue_len": len(queue),
                "free_ports": free_ports,
                "active_count": 0, # Approximation for standalone policy
                "avg_slack": float(np.mean([max(0, (ev["deadline_step"] - t) * 5) for ev in queue])) if queue else 0.0,
                "load": 0.0,
                "load_norm": 0.0
            }
            _, action_fn = agent.select_action(obs, greedy=True)
            return action_fn(queue, free_ports, t)
        return _policy

    def save(self, path="dqn_model.pth"):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_wait_times': self.episode_wait_times,
            'episode_completions': self.episode_completions
        }, path)
        print(f"[DQNAgent] Saved to {path}")

    def load(self, path="dqn_model.pth"):
        checkpoint = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.eps_min)
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_wait_times = checkpoint.get('episode_wait_times', [])
        self.episode_completions = checkpoint.get('episode_completions', [])
        print(f"[DQNAgent] Loaded from {path}")
        return self

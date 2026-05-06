"""
rl_agent.py
Tabular Q-learning agent for EV charging scheduling.
Action space: choose one of FIFO / SJF / EDF at each time step.
"""

import numpy as np
import os
import pickle
from schedulers import fifo, sjf, edf

# ── Action definitions ────────────────────────────────────────────────────────
ACTIONS = {0: fifo, 1: sjf, 2: edf}
ACTION_NAMES = {0: "FIFO", 1: "SJF", 2: "EDF"}
N_ACTIONS = len(ACTIONS)

# ── State discretisation ─────────────────────────────────────────────────────
HOUR_BINS        = list(range(24))                    # 0-23
QUEUE_BINS       = [0, 5, 10, 15, 20, 999]           # 5 buckets
FREE_PORT_BINS   = [0, 5, 10, 15, 999]               # 4 buckets
URGENCY_BINS     = [0, 30, 60, 120, 240, 480, 99999] # 6 buckets (minutes)


def discretise_state(obs):
    """
    Convert a raw observation dict to a discrete state tuple.
    obs keys: hour, queue_len, free_ports, avg_slack
    Returns: (hour_bucket, queue_bucket, port_bucket, urgency_bucket)
    """
    hour    = int(obs["hour"]) % 24

    qlen    = obs["queue_len"]
    q_bucket = sum(1 for b in QUEUE_BINS[1:] if qlen >= b)
    q_bucket = min(q_bucket, len(QUEUE_BINS) - 2)

    fp      = obs["free_ports"]
    p_bucket = sum(1 for b in FREE_PORT_BINS[1:] if fp >= b)
    p_bucket = min(p_bucket, len(FREE_PORT_BINS) - 2)

    slack   = obs["avg_slack"]
    u_bucket = sum(1 for b in URGENCY_BINS[1:] if slack >= b)
    u_bucket = min(u_bucket, len(URGENCY_BINS) - 2)

    return (hour, q_bucket, p_bucket, u_bucket)


def state_space_size():
    return (
        24,
        len(QUEUE_BINS)     - 1,
        len(FREE_PORT_BINS) - 1,
        len(URGENCY_BINS)   - 1,
    )


# ── Q-learning agent ──────────────────────────────────────────────────────────
class QLearningAgent:
    """
    Tabular Q-learning agent.

    Parameters
    ----------
    alpha   : float  learning rate (default 0.1)
    gamma   : float  discount factor (default 0.95)
    epsilon : float  initial exploration rate (default 1.0)
    eps_min : float  minimum epsilon (default 0.01)
    eps_decay : float per-episode decay factor (default 0.995)
    """

    def __init__(self, alpha=0.1, gamma=0.95,
                 epsilon=1.0, eps_min=0.01, eps_decay=0.995,
                 seed=42):
        self.alpha     = alpha
        self.gamma     = gamma
        self.epsilon   = epsilon
        self.eps_min   = eps_min
        self.eps_decay = eps_decay
        self.rng       = np.random.default_rng(seed)

        # Q-table: shape (24, 5, 4, 6, 3)
        self.Q = np.zeros(state_space_size() + (N_ACTIONS,))

        # Training logs
        self.episode_rewards    = []
        self.episode_wait_times = []
        self.episode_completions = []

    # ── Core methods ─────────────────────────────────────────────────────────
    def select_action(self, obs, greedy=False):
        """
        epsilon-greedy action selection.
        Returns an int (action index) and the scheduler function.
        """
        s = discretise_state(obs)
        if not greedy and self.rng.random() < self.epsilon:
            a = int(self.rng.integers(N_ACTIONS))
        else:
            a = int(np.argmax(self.Q[s]))
        return a, ACTIONS[a]

    def update(self, obs, action_idx, reward, next_obs):
        """Bellman update for one (s, a, r, s') transition."""
        s  = discretise_state(obs)
        s2 = discretise_state(next_obs)
        td_target = reward + self.gamma * np.max(self.Q[s2])
        td_error  = td_target - self.Q[s + (action_idx,)]
        self.Q[s + (action_idx,)] += self.alpha * td_error

    def decay_epsilon(self):
        """Call once per episode."""
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    # ── Training loop ─────────────────────────────────────────────────────────
    def train(self, sim_factory, n_episodes=300, verbose=True):
        """
        Train the agent.

        Parameters
        ----------
        sim_factory : callable -> ChargingSimulator
            Called at the start of each episode to get a fresh simulator.
        n_episodes  : int
        verbose     : bool
        """
        print(f"\n{'='*50}")
        print(f"Q-LEARNING TRAINING  ({n_episodes} episodes)")
        print(f"alpha={self.alpha}, gamma={self.gamma}, "
              f"epsilon_0={self.epsilon}, decay={self.eps_decay}")
        print(f"State space: {state_space_size()} | "
              f"Actions: {list(ACTION_NAMES.values())}")
        print(f"{'='*50}\n")

        for ep in range(1, n_episodes + 1):
            sim  = sim_factory()
            obs  = sim.reset()
            done = False
            ep_reward = 0.0

            while not done:
                a_idx, action_fn = self.select_action(obs)
                next_obs, reward, done, _ = sim.step(action_fn)
                self.update(obs, a_idx, reward, next_obs)
                obs        = next_obs
                ep_reward += reward

            metrics = sim.get_metrics()
            self.episode_rewards.append(ep_reward)
            self.episode_wait_times.append(metrics["avg_waiting_time_min"])
            self.episode_completions.append(metrics["completion_rate_pct"])
            self.decay_epsilon()

            if verbose and ep % 50 == 0:
                avg_r  = np.mean(self.episode_rewards[-50:])
                avg_w  = np.mean(self.episode_wait_times[-50:])
                avg_cr = np.mean(self.episode_completions[-50:])
                print(f"  Ep {ep:4d}/{n_episodes} | "
                      f"Avg reward: {avg_r:8.1f} | "
                      f"Avg wait: {avg_w:6.1f} min | "
                      f"Completion: {avg_cr:5.1f}% | "
                      f"eps={self.epsilon:.3f}")

        print("\nTraining complete.")
        return self

    # ── Greedy policy wrapper (for use as action_fn in sim.step) ──────────────
    def greedy_policy(self):
        """
        Returns a callable suitable as action_fn for ChargingSimulator.step().
        Uses greedy (no exploration) action selection.
        """
        agent = self   # capture

        def _policy(queue, free_ports, t):
            # Build a minimal obs from current queue / sim state
            # (We pass a partial obs; full obs is built inside the agent)
            obs = {
                "hour":       (t * 5 / 60) % 24,
                "queue_len":  len(queue),
                "free_ports": free_ports,
                "avg_slack":  float(np.mean([
                    max(0, (ev["deadline_step"] - t) * 5)
                    for ev in queue
                ])) if queue else 0.0,
                "load": 0.0,
                "load_norm": 0.0,
            }
            _, action_fn = agent.select_action(obs, greedy=True)
            return action_fn(queue, free_ports, t)

        return _policy

    # ── Save / load ───────────────────────────────────────────────────────────
    def save(self, path="qtable.pkl"):
        with open(path, "wb") as f:
            pickle.dump({
                "Q": self.Q,
                "epsilon": self.epsilon,
                "episode_rewards": self.episode_rewards,
            }, f)
        print(f"[QLearningAgent] Saved Q-table to {path}")

    def load(self, path="qtable.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.Q               = data["Q"]
        self.epsilon         = data.get("epsilon", self.eps_min)
        self.episode_rewards = data.get("episode_rewards", [])
        print(f"[QLearningAgent] Loaded Q-table from {path}")
        return self

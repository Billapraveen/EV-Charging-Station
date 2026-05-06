"""
main.py
Entry point for the EV Charging RL Scheduling project.
Runs preprocessing -> baseline evaluation -> Q-learning training -> comparison.

Usage:
    python main.py                      # run on synthetic data
    python main.py --data sessions.json # run on real ACN-Data JSON export
"""

import argparse
import os
import numpy as np

from simulator   import ChargingSimulator, make_synthetic_sessions
from schedulers  import get_scheduler
from rl_agent    import QLearningAgent
from dqn_agent   import DQNAgent
from evaluation  import (compare_all, plot_load_profiles,
                          plot_metrics_comparison, plot_training_curve,
                          print_summary_table)


def parse_args():
    p = argparse.ArgumentParser(
        description="EV Charging RL Scheduling — Mid Review")
    p.add_argument("--data",     default=None,
                   help="Path to ACN-Data JSON file (omit for synthetic data)")
    p.add_argument("--site",     default="caltech")
    p.add_argument("--start",    default=None, help="YYYY-MM-DD")
    p.add_argument("--end",      default=None, help="YYYY-MM-DD")
    p.add_argument("--n-ports",  type=int,   default=54)
    p.add_argument("--cap",      type=float, default=150.0,
                   help="Site power cap in kW")
    p.add_argument("--episodes", type=int,   default=300)
    p.add_argument("--no-train", action="store_true",
                   help="Skip RL training, only run classical baselines")
    p.add_argument("--load-qt",  default=None,
                   help="Path to saved Q-table (.pkl) to skip training")
    p.add_argument("--save-qt",  default="qtable.pkl")
    p.add_argument("--n-synthetic", type=int, default=500)
    p.add_argument("--v2g", action="store_true", help="Enable V2G discharging")
    p.add_argument("--dqn", action="store_true", help="Train DQN agent instead of tabular Q-learning")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 1. Load / generate sessions ──────────────────────────────────────────
    print("\n" + "="*55)
    print("  EV CHARGING RL SCHEDULING — IIITDM KANCHEEPURAM")
    print("  CS22B1001 — Mid Review")
    print("="*55)

    if args.data:
        from data_loader import load_sessions
        print(f"\n[Step 1] Loading ACN-Data from: {args.data}")
        sessions, n_steps = load_sessions(
            args.data, site=args.site,
            start_date=args.start, end_date=args.end)
    else:
        print(f"\n[Step 1] Generating synthetic sessions (n={args.n_synthetic})")
        sessions, n_steps = make_synthetic_sessions(
            n=args.n_synthetic, seed=42)

    print(f"         Sessions ready: {len(sessions)} | Steps: {n_steps}")

    # ── 2. Factory function for simulator ────────────────────────────────────
    def sim_factory():
        return ChargingSimulator(
            sessions, n_ports=args.n_ports, power_cap=args.cap, v2g_enabled=args.v2g)

    # ── 3. Agents ────────────────────────────────────────────────────────────
    q_agent = None
    dqn_agent = None
    if not args.no_train:
        if args.dqn:
            dqn_agent = DQNAgent(seed=42)
            print(f"\n[Step 2] Training DQN agent "
                  f"({args.episodes} episodes)...")
            dqn_agent.train(sim_factory, n_episodes=args.episodes, verbose=True)
            dqn_agent.save("dqn_model.pth")
        else:
            q_agent = QLearningAgent(alpha=0.1, gamma=0.95,
                                   epsilon=1.0, eps_min=0.01,
                                   eps_decay=0.995, seed=42)
            if args.load_qt and os.path.exists(args.load_qt):
                q_agent.load(args.load_qt)
            else:
                print(f"\n[Step 2] Training Q-learning agent "
                      f"({args.episodes} episodes)...")
                q_agent.train(sim_factory, n_episodes=args.episodes, verbose=True)
                q_agent.save(args.save_qt)

    # ── 4. Evaluate all algorithms ───────────────────────────────────────────
    print("\n[Step 3] Evaluating all algorithms on test sessions...")
    df_metrics, histories = compare_all(
        sessions, q_agent=q_agent, dqn_agent=dqn_agent,
        n_ports=args.n_ports, power_cap=args.cap, v2g_enabled=args.v2g)

    print_summary_table(df_metrics)

    # ── 5. Save results ──────────────────────────────────────────────────────
    df_metrics.to_csv("results_comparison.csv")
    print("[Step 4] Results saved to results_comparison.csv")

    # ── 6. Plots ─────────────────────────────────────────────────────────────
    print("\n[Step 5] Generating plots...")
    os.makedirs("plots", exist_ok=True)

    plot_load_profiles(histories, power_cap=args.cap,
                       save_path="plots/load_profiles.png")
    plot_metrics_comparison(df_metrics,
                            save_path="plots/metrics_comparison.png")
    if q_agent and q_agent.episode_rewards:
        plot_training_curve(q_agent,
                            save_path="plots/q_training_curve.png")
    if dqn_agent and dqn_agent.episode_rewards:
        plot_training_curve(dqn_agent,
                            save_path="plots/dqn_training_curve.png")

    print("\nDone! All outputs saved.")


if __name__ == "__main__":
    main()

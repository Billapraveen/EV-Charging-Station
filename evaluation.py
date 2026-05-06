"""
evaluation.py
Run all scheduling algorithms on the same session trace
and return a comparison table + plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from copy import deepcopy

from simulator  import ChargingSimulator
from schedulers import get_scheduler
from rl_agent   import QLearningAgent


# ── Run a single algorithm ────────────────────────────────────────────────────
def run_algorithm(sessions, name, action_fn,
                  n_ports=54, power_cap=150.0, v2g_enabled=False):
    """
    Simulate the given action_fn on sessions and return metrics + history.
    """
    sim  = ChargingSimulator(sessions, n_ports=n_ports, power_cap=power_cap, v2g_enabled=v2g_enabled)
    obs  = sim.reset()
    done = False
    while not done:
        _, _, done, _ = sim.step(action_fn)

    metrics             = sim.get_metrics()
    metrics["Algorithm"] = name
    return metrics, sim.history_load, sim.history_queue


# ── Compare all algorithms ────────────────────────────────────────────────────
def compare_all(sessions, q_agent=None, dqn_agent=None, n_ports=54, power_cap=150.0, v2g_enabled=False):
    """
    Run FIFO, SJF, EDF, LLF, Priority and (optionally) Q-learning/DQN on the same sessions.
    Returns a summary DataFrame and per-algorithm load histories.
    """
    results  = []
    histories = {}

    for name in ["FIFO", "SJF", "EDF", "LLF", "Priority"]:
        try:
            fn = get_scheduler(name)
            m, load_h, queue_h = run_algorithm(
                sessions, name, fn, n_ports, power_cap, v2g_enabled)
            results.append(m)
            histories[name] = {"load": load_h, "queue": queue_h}
            print(f"  [{name:8s}] Wait={m['avg_waiting_time_min']:6.1f} min | "
                  f"CR={m['completion_rate_pct']:5.1f}% | "
                  f"Var={m['load_variance_kw2']:7.1f} kW²")
        except ValueError:
            pass # In case priority or LLF aren't defined

    if q_agent is not None:
        fn = q_agent.greedy_policy()
        m, load_h, queue_h = run_algorithm(
            sessions, "Tabular Q-learning", fn, n_ports, power_cap, v2g_enabled)
        results.append(m)
        histories["Tabular Q-learning"] = {"load": load_h, "queue": queue_h}
        print(f"  [Q-Lr]     Wait={m['avg_waiting_time_min']:6.1f} min | "
              f"CR={m['completion_rate_pct']:5.1f}% | "
              f"Var={m['load_variance_kw2']:7.1f} kW²")

    if dqn_agent is not None:
        fn = dqn_agent.greedy_policy()
        m, load_h, queue_h = run_algorithm(
            sessions, "DQN", fn, n_ports, power_cap, v2g_enabled)
        results.append(m)
        histories["DQN"] = {"load": load_h, "queue": queue_h}
        print(f"  [DQN]      Wait={m['avg_waiting_time_min']:6.1f} min | "
              f"CR={m['completion_rate_pct']:5.1f}% | "
              f"Var={m['load_variance_kw2']:7.1f} kW²")

    df = pd.DataFrame(results).set_index("Algorithm")
    cols = [
        "avg_waiting_time_min", "completion_rate_pct",
        "utilisation_pct", "load_variance_kw2",
        "energy_cost_inr", "n_completed", "n_dropped",
    ]
    return df[cols], histories


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_load_profiles(histories, power_cap=150.0,
                       step_minutes=5, save_path=None):
    """Plot site load over time for each algorithm."""
    fig, axes = plt.subplots(len(histories), 1,
                             figsize=(12, 3 * len(histories)),
                             sharex=True)
    if len(histories) == 1:
        axes = [axes]

    colors = {"FIFO": "#4C72B0", "SJF": "#DD8452",
              "EDF": "#55A868", "Q-learning": "#C44E52"}

    for ax, (name, hist) in zip(axes, histories.items()):
        load = np.array(hist["load"])
        t_hours = np.arange(len(load)) * step_minutes / 60
        color = colors.get(name, "#8172B2")

        ax.fill_between(t_hours, load, alpha=0.3, color=color)
        ax.plot(t_hours, load, color=color, linewidth=1.5, label=name)
        ax.axhline(power_cap, color="red", linestyle="--",
                   linewidth=1.2, label=f"Cap ({power_cap} kW)")
        ax.set_ylabel("Load (kW)", fontsize=9)
        ax.set_title(f"{name} — Load Profile", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, power_cap * 1.2)

    axes[-1].set_xlabel("Time (hours)", fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def plot_metrics_comparison(df_metrics, save_path=None):
    """Bar chart comparing key metrics across algorithms."""
    metrics = {
        "avg_waiting_time_min": "Avg Waiting Time (min)",
        "completion_rate_pct":  "Completion Rate (%)",
        "load_variance_kw2":    "Load Variance (kW²)",
        "energy_cost_inr":      "Energy Cost (Rs.)",
    }
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for ax, (col, title) in zip(axes, metrics.items()):
        vals  = df_metrics[col]
        bars  = ax.bar(vals.index, vals.values,
                       color=colors[:len(vals)], edgecolor="black",
                       linewidth=0.7)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(title.split("(")[-1].rstrip(")") if "(" in title else "")
        ax.set_xticklabels(vals.index, rotation=15, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Algorithm Comparison — EV Charging Scheduling",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def plot_training_curve(agent, save_path=None):
    """Plot Q-learning training reward and waiting time curves."""
    if not agent.episode_rewards:
        print("No training data to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    eps = list(range(1, len(agent.episode_rewards) + 1))

    # Smooth with rolling average
    window = 20
    def smooth(arr):
        return pd.Series(arr).rolling(window, min_periods=1).mean()

    ax1.plot(eps, smooth(agent.episode_rewards),
             color="#4C72B0", linewidth=1.5)
    ax1.set_ylabel("Episode Reward", fontsize=10)
    ax1.set_title("Q-Learning Training Progress", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    ax2.plot(eps, smooth(agent.episode_wait_times),
             color="#DD8452", linewidth=1.5)
    ax2.set_ylabel("Avg Waiting Time (min)", fontsize=10)
    ax2.set_xlabel("Episode", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def print_summary_table(df_metrics):
    """Pretty-print the comparison table."""
    rename = {
        "avg_waiting_time_min": "Avg Wait (min)",
        "completion_rate_pct":  "Completion (%)",
        "utilisation_pct":      "Utilisation (%)",
        "load_variance_kw2":    "Load Var (kW²)",
        "energy_cost_inr":      "Cost (Rs.)",
        "n_completed":          "Completed",
        "n_dropped":            "Dropped",
    }
    display = df_metrics.rename(columns=rename)
    print("\n" + "="*70)
    print("  ALGORITHM COMPARISON")
    print("="*70)
    print(display.to_string())
    print("="*70 + "\n")

"""
dashboard.py
Streamlit dashboard for EV Charging Scheduling project.

Run with:
    streamlit run dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

# Allow import of sibling modules
sys.path.insert(0, os.path.dirname(__file__))

from simulator   import ChargingSimulator, make_synthetic_sessions
from schedulers  import get_scheduler
from rl_agent    import QLearningAgent
from dqn_agent   import DQNAgent
from evaluation  import compare_all
from data_loader import load_sessions
from ml_prediction import run_ml_pipeline

@st.cache_data(show_spinner=False)
def _cached_synthetic(n_sessions, seed):
    return make_synthetic_sessions(n=n_sessions, n_steps=288, seed=seed)

@st.cache_data(show_spinner=False)
def _cached_local(local_file, site_choice):
    return load_sessions(local_file, site=site_choice, verbose=False)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EV Charging Scheduler",
    page_icon="⚡",
    layout="wide",
)

COLORS = {
    "FIFO":               "#4C72B0",
    "SJF":                "#DD8452",
    "EDF":                "#55A868",
    "LLF":                "#8C564B",
    "Priority":           "#E377C2",
    "Tabular Q-learning": "#C44E52",
    "DQN":                "#9467BD",
}


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚡ EV Charging RL Scheduler")
st.sidebar.markdown("---")

st.sidebar.subheader("Data Source")
data_source = st.sidebar.selectbox("Source", ["Synthetic", "Local File"])
if data_source == "Synthetic":
    n_sessions  = st.sidebar.slider("Number of sessions",  100, 1000, 400, 50)
else:
    local_file = st.sidebar.text_input("Path to JSON/CSV", "sessions.json")
    site_choice = st.sidebar.selectbox("Site", ["caltech", "jpl"])

st.sidebar.subheader("Simulation Parameters")
n_ports     = st.sidebar.slider("Number of ports (EVSEs)", 10, 100, 54, 2)
power_cap   = st.sidebar.slider("Power cap (kW)", 50, 300, 150, 10)
v2g_enabled = st.sidebar.checkbox("Enable V2G Discharging", value=False)

st.sidebar.subheader("RL Training")
n_episodes  = st.sidebar.slider("Training episodes", 10, 300, 30, 10)
run_tabular = st.sidebar.checkbox("Include Tabular Q-learning", value=False)
run_dqn     = st.sidebar.checkbox("Include DQN Agent", value=False)
seed        = st.sidebar.number_input("Random seed", value=42, step=1)

run_btn = st.sidebar.button("▶ Run Simulation", type="primary")


# ── Main content ──────────────────────────────────────────────────────────────
st.title("⚡ EV Charging Scheduling Dashboard")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Algorithm Comparison",
    "📈 Load Profiles",
    "🤖 RL Training",
    "🧠 ML Predictions",
    "🔬 Ablation Study",
    "ℹ️ About",
])

if "results" not in st.session_state:
    st.info("👈 Configure settings in the sidebar, then click **▶ Run Simulation** to start.")

if run_btn:
    progress_bar = st.progress(0, text="Loading data...")
    
    # 1. Data loading
    if data_source == "Synthetic":
        sessions, n_steps = _cached_synthetic(n_sessions, int(seed))
    else:
        try:
            sessions, n_steps = _cached_local(local_file, site_choice)
        except Exception as e:
            st.error(f"Failed to load local file: {e}")
            st.stop()
    
    progress_bar.progress(15, text="Data loaded. Setting up simulator...")

    # 2. Train agents
    q_agent = None
    dqn_agent = None
    
    def sim_factory():
        return ChargingSimulator(sessions, n_ports=n_ports,
                                 power_cap=power_cap, v2g_enabled=v2g_enabled)

    if run_tabular:
        progress_bar.progress(20, text="Training Tabular Q-learning agent...")
        q_agent = QLearningAgent(alpha=0.1, gamma=0.95,
                               epsilon=1.0, eps_min=0.01,
                               eps_decay=0.995, seed=int(seed))
        q_agent.train(sim_factory, n_episodes=n_episodes, verbose=False)

    if run_dqn:
        progress_bar.progress(30, text="Training DQN agent (this may take a moment)...")
        dqn_agent = DQNAgent(seed=int(seed))
        dqn_agent.train(sim_factory, n_episodes=n_episodes, verbose=False)

    # 3. Evaluate algorithms
    progress_bar.progress(70, text="Evaluating all algorithms...")
    df_metrics, histories = compare_all(
        sessions, q_agent=q_agent, dqn_agent=dqn_agent,
        n_ports=n_ports, power_cap=power_cap, v2g_enabled=v2g_enabled)

    # 4. ML Predictions
    progress_bar.progress(80, text="Running ML prediction pipeline...")
    ml_results = run_ml_pipeline(sessions)

    # 5. Ablation Study
    progress_bar.progress(90, text="Running ablation study...")
    base_fn = get_scheduler("FIFO")
    
    sim_unconstrained = ChargingSimulator(sessions, n_ports=n_ports, power_cap=9999.0, v2g_enabled=False)
    obs = sim_unconstrained.reset()
    done = False
    while not done:
        _, _, done, _ = sim_unconstrained.step(base_fn)
    ablation_metrics_unconstrained = sim_unconstrained.get_metrics()
    
    sim_no_v2g = ChargingSimulator(sessions, n_ports=n_ports, power_cap=power_cap, v2g_enabled=False)
    obs = sim_no_v2g.reset()
    done = False
    while not done:
        _, _, done, _ = sim_no_v2g.step(base_fn)
    ablation_metrics_nov2g = sim_no_v2g.get_metrics()
    
    sim_v2g = ChargingSimulator(sessions, n_ports=n_ports, power_cap=power_cap, v2g_enabled=True)
    obs = sim_v2g.reset()
    done = False
    while not done:
        _, _, done, _ = sim_v2g.step(base_fn)
    ablation_metrics_v2g = sim_v2g.get_metrics()

    ablation_df = pd.DataFrame([
        {"Variant": "No Power Cap", **ablation_metrics_unconstrained},
        {"Variant": "With Power Cap (No V2G)", **ablation_metrics_nov2g},
        {"Variant": "With Power Cap & V2G", **ablation_metrics_v2g},
    ]).set_index("Variant")

    # Save to session state
    st.session_state["sessions"]         = sessions
    st.session_state["results"]          = df_metrics
    st.session_state["histories"]        = histories
    st.session_state["q_agent"]          = q_agent
    st.session_state["dqn_agent"]        = dqn_agent
    st.session_state["power_cap"]        = power_cap
    st.session_state["ml_results"]       = ml_results
    st.session_state["ablation_df"]      = ablation_df

    progress_bar.progress(100, text="Done! ✅")
    st.success("Simulation complete! View results in the tabs above.")


# ── Tab 1: Metrics comparison ─────────────────────────────────────────────────
with tab1:
    st.subheader("Algorithm Comparison Metrics")
    df = st.session_state.get("results")
    if df is not None:
        rename = {
            "avg_waiting_time_min": "Avg Wait (min)",
            "completion_rate_pct":  "Completion (%)",
            "utilisation_pct":      "Utilisation (%)",
            "load_variance_kw2":    "Load Var (kW²)",
            "energy_cost_inr":      "Cost (Rs.)",
            "n_completed":          "Completed",
            "n_dropped":            "Dropped",
        }
        st.dataframe(df.rename(columns=rename).style.highlight_min(
            subset=["Avg Wait (min)", "Load Var (kW²)", "Cost (Rs.)"],
            color="lightgreen"
        ).highlight_max(
            subset=["Completion (%)"],
            color="lightgreen"
        ), use_container_width=True)

        # Bar charts
        col1, col2 = st.columns(2)
        metrics_to_plot = [
            ("avg_waiting_time_min", "Avg Waiting Time (min)"),
            ("completion_rate_pct",  "Completion Rate (%)"),
            ("load_variance_kw2",    "Load Variance (kW²)"),
            ("energy_cost_inr",      "Energy Cost (Rs.)"),
        ]
        for i, (col_name, title) in enumerate(metrics_to_plot):
            target = col1 if i % 2 == 0 else col2
            with target:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                vals  = df[col_name]
                bars  = ax.bar(
                    range(len(vals)), vals.values,
                    color=[COLORS.get(n, "#888") for n in vals.index],
                    edgecolor="black", linewidth=0.6)
                ax.set_title(title, fontsize=10, fontweight="bold")
                ax.set_xticks(range(len(vals)))
                ax.set_xticklabels(vals.index, rotation=45, ha='right', fontsize=8)
                ax.grid(axis="y", alpha=0.3)
                for bar, v in zip(bars, vals.values):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height()*1.01,
                            f"{v:.1f}", ha="center", va="bottom", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

# ── Tab 2: Load profiles ──────────────────────────────────────────────────────
with tab2:
    st.subheader("Site Load Profiles vs Power Cap")
    histories = st.session_state.get("histories", {})
    pcap      = st.session_state.get("power_cap", 150)

    if histories:
        for name, hist in histories.items():
            load    = np.array(hist["load"])
            t_hours = np.arange(len(load)) * 5 / 60
            color   = COLORS.get(name, "#888")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))

            # Load profile
            ax1.fill_between(t_hours, load, alpha=0.25, color=color)
            ax1.plot(t_hours, load, color=color, linewidth=1.5)
            ax1.axhline(pcap, color="red", linestyle="--",
                        linewidth=1.2, label=f"Cap ({pcap} kW)")
            ax1.axhline(0, color="black", linewidth=0.5)
            ax1.set_title(f"{name} — Load Profile", fontsize=10)
            ax1.set_xlabel("Time (hr)"); ax1.set_ylabel("Power (kW)")
            ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
            # Allow negative y-limits for V2G
            min_y = min(-10, np.min(load) * 1.1)
            ax1.set_ylim(min_y, pcap * 1.25)

            # Queue length
            queue = np.array(hist["queue"])
            ax2.fill_between(t_hours[:len(queue)], queue,
                             alpha=0.25, color=color)
            ax2.plot(t_hours[:len(queue)], queue,
                     color=color, linewidth=1.5)
            ax2.set_title(f"{name} — Queue Length", fontsize=10)
            ax2.set_xlabel("Time (hr)"); ax2.set_ylabel("Waiting EVs")
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ── Tab 3: RL Training ────────────────────────────────────────────────────────
with tab3:
    st.subheader("Reinforcement Learning Training Progress")
    q_agent = st.session_state.get("q_agent")
    dqn_agent = st.session_state.get("dqn_agent")

    def plot_agent_curve(agent, name, color):
        if not agent or not agent.episode_rewards:
            return
        
        rewards = agent.episode_rewards
        waits = agent.episode_wait_times
        window = max(1, len(rewards) // 20)
        def smooth(arr):
            return pd.Series(arr).rolling(window, min_periods=1).mean()

        st.markdown(f"**{name} Agent**")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.5))
        eps = list(range(1, len(rewards)+1))

        ax1.plot(eps, smooth(rewards), color=color, linewidth=1.8)
        ax1.set_ylabel("Episode Reward", fontsize=10)
        ax1.set_title("Cumulative Reward per Episode", fontsize=11)
        ax1.grid(True, alpha=0.3)

        ax2.plot(eps, smooth(waits), color=color, linewidth=1.8)
        ax2.set_ylabel("Avg Waiting Time (min)", fontsize=10)
        ax2.set_xlabel("Episode", fontsize=10)
        ax2.set_title("Average Waiting Time per Episode", fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    if q_agent:
        plot_agent_curve(q_agent, "Tabular Q-Learning", COLORS["Tabular Q-learning"])
    if dqn_agent:
        plot_agent_curve(dqn_agent, "Deep Q-Network (DQN)", COLORS["DQN"])
        
    if not q_agent and not dqn_agent:
        st.info("Enable an RL agent in the sidebar and click Run Simulation.")

# ── Tab 4: ML Predictions ─────────────────────────────────────────────────────
with tab4:
    st.subheader("Energy Demand Prediction")
    st.markdown("Predicting EV energy demand (`energy_kwh`) using Random Forest, XGBoost, and SVM.")
    
    ml_results = st.session_state.get("ml_results")
    if ml_results:
        metrics_df = pd.DataFrame({
            model: {"RMSE": res["RMSE"], "MAE": res["MAE"]}
            for model, res in ml_results.items()
        }).T
        
        st.table(metrics_df.style.highlight_min(color='lightgreen'))
        
        col1, col2 = st.columns(2)
        
        for name in ['Random Forest', 'XGBoost']:
            if name in ml_results and ml_results[name]['importance']:
                with (col1 if name == 'Random Forest' else col2):
                    fig, ax = plt.subplots(figsize=(5, 4))
                    imp = ml_results[name]['importance']
                    sorted_imp = sorted(imp.items(), key=lambda x: x[1])
                    
                    features = [x[0].replace('_', ' ').title() for x in sorted_imp]
                    importances = [x[1] for x in sorted_imp]
                    
                    ax.barh(features, importances, color='steelblue')
                    ax.set_title(f"{name} Feature Importance")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
    else:
        st.warning("Not enough sessions for ML prediction.")

# ── Tab 5: Ablation Study ─────────────────────────────────────────────────────
with tab5:
    st.subheader("Ablation Study: Impact of System Constraints (using FIFO)")
    st.markdown("Evaluating the system by enabling/disabling the Power Cap and Vehicle-to-Grid (V2G) functionality.")
    
    ablation_df = st.session_state.get("ablation_df")
    if ablation_df is not None:
        rename = {
            "avg_waiting_time_min": "Avg Wait (min)",
            "completion_rate_pct":  "Completion (%)",
            "load_variance_kw2":    "Load Var (kW²)",
            "energy_cost_inr":      "Cost (Rs.)",
        }
        st.dataframe(ablation_df[["avg_waiting_time_min", "completion_rate_pct", "load_variance_kw2", "energy_cost_inr"]].rename(columns=rename))
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot Cost
        vals = ablation_df["energy_cost_inr"]
        axes[0].bar(range(len(vals)), vals.values, color=["#E24A33", "#348ABD", "#988ED5"])
        axes[0].set_title("Energy Cost Comparison (Rs.)")
        axes[0].set_xticks(range(len(vals)))
        axes[0].set_xticklabels(vals.index, rotation=15)
        
        # Plot Variance
        vals = ablation_df["load_variance_kw2"]
        axes[1].bar(range(len(vals)), vals.values, color=["#E24A33", "#348ABD", "#988ED5"])
        axes[1].set_title("Load Variance (kW²)")
        axes[1].set_xticks(range(len(vals)))
        axes[1].set_xticklabels(vals.index, rotation=15)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Run the simulation to generate the ablation study.")

# ── Tab 6: About ──────────────────────────────────────────────────────────────
with tab6:
    st.subheader("About This Project")
    st.markdown("""
    **Title:** Reinforcement Learning-Based Scheduling of Electric Vehicle Charging
    to Minimize Waiting Time and Grid Overload

    ---
    ### System Overview
    - **Dataset:** ACN-Data (Caltech/JPL)
    - **Simulator:** Discrete-time (5 min steps), hard power cap enforcement, Optional V2G
    - **Baselines:** FIFO, SJF, EDF, LLF, Priority
    - **RL Agents:** Tabular Q-learning, Deep Q-Network (PyTorch)
    - **ML Prediction:** RF, XGBoost, SVM
    """)

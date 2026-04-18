import streamlit as st
import numpy as np
import json
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


@st.cache_data
def load_metrics():
    rows = []
    for algo, folder in [("DQN", "dqn"), ("Double DQN", "ddqn"), ("TD3", "td3")]:
        config_path = os.path.join(RESULTS_DIR, folder, "config.json")
        rewards_path = os.path.join(RESULTS_DIR, folder, "rewards.npy")

        if not os.path.exists(config_path):
            continue

        with open(config_path) as f:
            config = json.load(f)
        rewards = np.load(rewards_path)

        rows.append({
            "Algorithm": algo,
            "Episodes Trained": len(rewards),
            "Training Time": f'{config.get("training_time_min", "N/A")} min',
            "Final 100-ep Avg": f"{np.mean(rewards[-100:]):.1f}",
            "Best Episode": f"{np.max(rewards):.1f}",
            "Std (last 100)": f"{np.std(rewards[-100:]):.1f}",
        })
    return pd.DataFrame(rows)


def show_image(path, caption=""):
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Image not found: {os.path.basename(path)}")


def render():
    st.title("Results & Analysis")
    st.markdown("Training performance, comparisons, and hyperparameter analysis from CA1.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Performance Summary",
        "Training Curves",
        "Hyperparameter Sensitivity",
        "Evaluation"
    ])

    with tab1:
        st.markdown("### Algorithm Performance Comparison")
        df = load_metrics()
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown("### All Algorithms Comparison")
        show_image(os.path.join(RESULTS_DIR, "all_algorithms_comparison.png"),
                   "Training reward curves for DQN, Double DQN, and TD3")

    with tab2:
        st.markdown("### Individual Training Curves")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**DQN**")
            show_image(os.path.join(RESULTS_DIR, "dqn", "training_rewards.png"))
        with col2:
            st.markdown("**Double DQN**")
            show_image(os.path.join(RESULTS_DIR, "ddqn", "training_rewards.png"))
        with col3:
            st.markdown("**TD3**")
            show_image(os.path.join(RESULTS_DIR, "td3", "training_rewards.png"))

        st.markdown("### DQN vs Double DQN")
        col1, col2 = st.columns(2)
        with col1:
            show_image(os.path.join(RESULTS_DIR, "dqn_vs_ddqn_comparison.png"),
                       "Standard comparison")
        with col2:
            show_image(os.path.join(RESULTS_DIR, "dqn_vs_ddqn_detailed.png"),
                       "Detailed comparison")

        st.markdown("### Epsilon Decay")
        show_image(os.path.join(RESULTS_DIR, "epsilon_decay_comparison.png"),
                   "Exploration rate over training episodes")

    with tab3:
        st.markdown("### Discount Factor (Gamma) Sensitivity")
        st.markdown("""
        The discount factor determines how far-sighted the agent is.
        We tested three values: 0.95, 0.99, and 0.999.
        """)
        show_image(os.path.join(RESULTS_DIR, "gamma_sensitivity.png"),
                   "Gamma = 0.99 is optimal; 0.95 fails entirely")

        st.markdown("### Learning Rate Sensitivity")
        st.markdown("""
        Learning rate controls how fast the network updates.
        Too high causes divergence, too low prevents convergence.
        """)
        show_image(os.path.join(RESULTS_DIR, "lr_sensitivity.png"),
                   "5e-4 optimal for DQN, 1e-3 for TD3")

    with tab4:
        st.markdown("### Evaluation Distributions")
        st.markdown("100 greedy episodes per algorithm with no exploration.")
        show_image(os.path.join(RESULTS_DIR, "evaluation_histograms.png"),
                   "Reward distributions across 100 evaluation episodes")

        st.markdown("### Individual Evaluation Histograms")
        col1, col2, col3 = st.columns(3)
        with col1:
            show_image(os.path.join(RESULTS_DIR, "dqn", "eval_histogram.png"), "DQN")
        with col2:
            show_image(os.path.join(RESULTS_DIR, "ddqn", "eval_histogram.png"), "Double DQN")
        with col3:
            show_image(os.path.join(RESULTS_DIR, "td3", "eval_histogram.png"), "TD3")

        st.markdown("### Random Baseline")
        show_image(os.path.join(RESULTS_DIR, "random_baseline.png"),
                   "Random policy performance (no learning)")

    with st.expander("Hyperparameter Configuration Details"):
        for algo, folder in [("DQN", "dqn"), ("Double DQN", "ddqn"), ("TD3", "td3")]:
            config_path = os.path.join(RESULTS_DIR, folder, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
                st.markdown(f"**{algo}**")
                st.json(config)

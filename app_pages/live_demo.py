import streamlit as st
import gymnasium as gym
import numpy as np
import io
import os
import time
from PIL import Image

from dqn_agent import DQNAgent
from td3_agent import TD3Agent

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


@st.cache_resource
def load_agent(algo_name):
    if algo_name == "DQN":
        agent = DQNAgent(double_dqn=False)
        agent.load(os.path.join(RESULTS_DIR, "dqn", "dqn_model.pth"))
        agent.epsilon = 0.0
        return agent, "LunarLander-v3"
    elif algo_name == "Double DQN":
        agent = DQNAgent(double_dqn=True)
        agent.load(os.path.join(RESULTS_DIR, "ddqn", "ddqn_model.pth"))
        agent.epsilon = 0.0
        return agent, "LunarLander-v3"
    elif algo_name == "TD3":
        agent = TD3Agent(hidden_dim=128)
        agent.load(os.path.join(RESULTS_DIR, "td3", "td3_model.pth"))
        return agent, "LunarLanderContinuous-v3"


def run_episode(agent, env_name, seed=None):
    env = gym.make(env_name, render_mode='rgb_array')
    reset_kwargs = {"seed": seed} if seed is not None else {}
    state, _ = env.reset(**reset_kwargs)

    frames = []
    cumulative_rewards = []
    total_reward = 0.0
    steps = 0

    for step in range(1000):
        frame = env.render()
        frames.append(frame)

        action = agent.select_action(state, training=False)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        cumulative_rewards.append(total_reward)
        steps = step + 1

        if terminated or truncated:
            frames.append(env.render())
            break

    env.close()

    if total_reward >= 200:
        status = "LANDED"
    elif total_reward < 0:
        status = "CRASHED"
    else:
        status = "PARTIAL"

    return frames, cumulative_rewards, total_reward, steps, status


def frames_to_gif(frames, duration=50):
    pil_frames = [Image.fromarray(f) for f in frames]
    buf = io.BytesIO()
    pil_frames[0].save(
        buf, format='GIF', save_all=True,
        append_images=pil_frames[1:],
        duration=duration, loop=0
    )
    buf.seek(0)
    return buf.getvalue()


def render():
    st.title("Live Demo")
    st.markdown("Watch the trained RL agents land on the moon in real-time.")

    if "episode_history" not in st.session_state:
        st.session_state.episode_history = []

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        algo = st.selectbox("Select Algorithm", ["DQN", "Double DQN", "TD3"])
    with col2:
        speed = st.slider("Playback Speed", 0.5, 3.0, 1.0, 0.5)
    with col3:
        seed = st.number_input("Seed (0 = random)", min_value=0, value=0, step=1)

    seed_val = seed if seed > 0 else None

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_single = st.button("Run 1 Episode", type="primary", use_container_width=True)
    with col_btn2:
        run_batch = st.button("Run 10 Episodes (Stats)", use_container_width=True)

    agent, env_name = load_agent(algo)

    if run_single:
        with st.spinner(f"Running {algo} episode..."):
            frames, rewards, total, steps, status = run_episode(agent, env_name, seed_val)
            duration = int(50 / speed)
            gif_bytes = frames_to_gif(frames, duration)

        col_gif, col_stats = st.columns([2, 1])

        with col_gif:
            st.image(gif_bytes, caption=f"{algo} Landing Attempt", use_container_width=True)

        with col_stats:
            if status == "LANDED":
                st.success(f"Status: {status}")
            elif status == "CRASHED":
                st.error(f"Status: {status}")
            else:
                st.warning(f"Status: {status}")

            st.metric("Total Reward", f"{total:.1f}")
            st.metric("Steps", steps)
            st.metric("Algorithm", algo)

            st.markdown("**Cumulative Reward**")
            st.line_chart(rewards)

        st.session_state.episode_history.append({
            "Algorithm": algo,
            "Reward": round(total, 1),
            "Steps": steps,
            "Status": status
        })

    if run_batch:
        progress = st.progress(0, text=f"Running 10 {algo} episodes...")
        batch_rewards = []
        batch_steps = []
        batch_status = []

        for i in range(10):
            _, _, total, steps, status = run_episode(agent, env_name)
            batch_rewards.append(total)
            batch_steps.append(steps)
            batch_status.append(status)
            progress.progress((i + 1) / 10, text=f"Episode {i+1}/10 — Reward: {total:.1f}")

        progress.empty()

        successes = sum(1 for s in batch_status if s == "LANDED")
        mean_reward = np.mean(batch_rewards)
        std_reward = np.std(batch_rewards)

        st.markdown(f"### Batch Results — {algo}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean Reward", f"{mean_reward:.1f}")
        col2.metric("Std Dev", f"{std_reward:.1f}")
        col3.metric("Success Rate", f"{successes}/10 ({successes*10}%)")
        col4.metric("Avg Steps", f"{np.mean(batch_steps):.0f}")

        st.bar_chart({"Episode Rewards": batch_rewards})

        for i, (r, s) in enumerate(zip(batch_rewards, batch_status)):
            st.session_state.episode_history.append({
                "Algorithm": algo,
                "Reward": round(r, 1),
                "Steps": batch_steps[i],
                "Status": s
            })

    if st.session_state.episode_history:
        st.markdown("---")
        st.markdown("### Episode History")
        import pandas as pd
        df = pd.DataFrame(st.session_state.episode_history)
        st.dataframe(df, use_container_width=True)

        if st.button("Clear History"):
            st.session_state.episode_history = []
            st.rerun()

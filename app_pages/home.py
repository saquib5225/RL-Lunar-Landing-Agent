import streamlit as st
import gymnasium as gym
import numpy as np
from PIL import Image


def render():
    st.title("Reinforcement Learning for Autonomous Lunar Landing")
    st.markdown("**A deep RL approach to solving the LunarLander problem from scratch**")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### What is Reinforcement Learning?

        Reinforcement Learning is a branch of AI where an **agent** learns to make decisions
        by interacting with an **environment**. The agent takes **actions**, receives **rewards**,
        and learns a **policy** that maximises its total reward over time.

        Unlike supervised learning, RL requires no labelled data. The agent discovers
        optimal behaviour through trial and error, much like how a child learns to walk
        by repeatedly trying and falling.

        ### The Problem: Lunar Landing

        A spacecraft spawns at the top of the screen and must land safely between two flags
        on the moon's surface. The agent controls the engines to slow down, steer, and
        touch down gently. A successful landing earns +100 to +140 reward, while crashing
        costs -100.
        """)

    with col2:
        try:
            env = gym.make('LunarLander-v3', render_mode='rgb_array')
            env.reset(seed=42)
            frame = env.render()
            env.close()
            st.image(frame, caption="LunarLander-v3 Environment", use_container_width=True)
        except Exception:
            st.info("LunarLander environment preview unavailable.")

    st.markdown("---")

    st.markdown("### Environment Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### State Space (8 variables)")
        st.markdown("""
        | Variable | Meaning |
        |---|---|
        | x, y | Position |
        | vx, vy | Velocity |
        | theta | Body angle |
        | omega | Angular velocity |
        | left leg | Ground contact |
        | right leg | Ground contact |
        """)

    with col2:
        st.markdown("#### Action Space")
        st.markdown("""
        **Discrete (DQN / DDQN):**
        - 0: Do nothing
        - 1: Fire left engine
        - 2: Fire main engine
        - 3: Fire right engine

        **Continuous (TD3):**
        - Main engine power [-1, 1]
        - Side engine power [-1, 1]
        """)

    with col3:
        st.markdown("#### Reward Signal")
        st.markdown("""
        | Event | Reward |
        |---|---|
        | Safe landing | +100 to +140 |
        | Leg contact | +10 each |
        | Crashing | -100 |
        | Main engine | -0.3 / step |
        | Side engine | -0.03 / step |

        **Solved:** avg reward >= 200
        """)

    st.markdown("---")

    st.markdown("### Algorithms Implemented")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        #### DQN
        **Deep Q-Network**

        Neural network learns Q-values for
        each discrete action. Uses experience
        replay and a target network for
        stable training.
        """)

    with col2:
        st.markdown("""
        #### Double DQN
        **Reduced Overestimation**

        Fixes DQN's tendency to overestimate
        Q-values by decoupling action selection
        (online net) from evaluation (target net).
        """)

    with col3:
        st.markdown("""
        #### TD3
        **Twin Delayed DDPG**

        Actor-critic for continuous control.
        Twin critics prevent overestimation,
        delayed updates stabilise training,
        target smoothing improves robustness.
        """)

    st.markdown("---")
    st.info("Navigate to **Live Demo** in the sidebar to watch the trained agents land.")

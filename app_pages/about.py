import streamlit as st


def render():
    st.title("About This Project")

    st.markdown("---")

    st.markdown("### Team Members")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **Afaq Ahmad**
        - Student ID: 20063247
        """)

    with col2:
        st.markdown("""
        **Saquib Zakir Husein Pirjade**
        - Student ID: 20079780
        """)

    with col3:
        st.markdown("""
        **Habib Ullah**
        - Student ID: 20073274
        """)

    st.markdown("---")

    st.markdown("### Module Information")
    st.markdown("""
    | | |
    |---|---|
    | **Module** | B9AI105 - Reinforcement Learning |
    | **Programme** | MSc Artificial Intelligence |
    | **Institution** | Dublin Business School |
    | **Lecturer** | Dr. Anesu Nyabadza |
    | **Assessment** | CA2 - Artifact Demonstration (60%) |
    """)

    st.markdown("---")

    st.markdown("### Project Overview")
    st.markdown("""
    This project applies reinforcement learning to the problem of autonomous lunar landing
    using the Gymnasium LunarLander environment. Three deep RL algorithms were implemented
    entirely from scratch using PyTorch:

    1. **DQN (Deep Q-Network)** for discrete control
    2. **Double DQN** for discrete control with reduced overestimation
    3. **TD3 (Twin Delayed DDPG)** for continuous control

    All algorithms successfully solved their respective environments, achieving average
    rewards exceeding 200 over 100 evaluation episodes.

    This web application serves as the CA2 deployment, demonstrating the trained agents
    in an interactive format.
    """)

    st.markdown("---")

    st.markdown("### Technology Stack")
    st.markdown("""
    | Component | Technology |
    |---|---|
    | RL Environment | Gymnasium (LunarLander-v3) |
    | Deep Learning | PyTorch |
    | Deployment | Streamlit |
    | Language | Python 3.11 |
    | Visualisation | Matplotlib, PIL |
    """)

    st.markdown("---")

    st.markdown("### How to Run")
    st.code("""
cd submission
source venv/bin/activate
pip install -r requirements_app.txt
streamlit run app.py
    """, language="bash")

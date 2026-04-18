import streamlit as st


def render():
    st.title("Ethical Considerations")
    st.markdown("""
    Deploying reinforcement learning in real-world systems raises important ethical questions
    that must be carefully considered before moving from simulation to production.
    """)

    st.markdown("---")

    st.markdown("### 1. Safety in Autonomous Systems")
    st.markdown("""
    RL agents learn through trial and error, which inherently means they will fail many times
    before succeeding. In our project, the agent crashed hundreds of times during training before
    learning to land safely. In simulation, this is harmless. In the real world, every crash has
    consequences.

    Autonomous landing systems for drones, rockets, or spacecraft cannot afford trial-and-error
    learning in deployment. A single failure could result in destruction of expensive equipment,
    environmental damage, or loss of life. This means RL-trained policies must be extensively
    validated in simulation before any real-world deployment, and safety constraints must be
    hard-coded as fallbacks that override the learned policy when dangerous states are detected.
    """)

    st.markdown("---")

    st.markdown("### 2. Bias in Reward Design")
    st.markdown("""
    The reward function is designed by humans and encodes specific values and priorities.
    In our LunarLander environment:

    - **Fuel penalty (-0.3 per engine fire):** This encodes a preference for fuel efficiency.
      A different reward could prioritise speed of landing over fuel conservation.
    - **Landing bonus (+100):** This values safe landings equally regardless of precision.
      A real system might need to weight landing accuracy differently.
    - **Crash penalty (-100):** This treats all crashes equally, but in reality a gentle
      crash is very different from a high-velocity impact.

    The reward function is a simplification of human values. Poorly designed rewards can lead
    to **reward hacking**, where the agent finds unexpected ways to maximise reward without
    achieving the intended goal. For example, an agent might learn to hover indefinitely
    to avoid the crash penalty rather than attempting to land.
    """)

    st.markdown("---")

    st.markdown("### 3. Accountability and Responsibility")
    st.markdown("""
    When a traditionally programmed controller fails, the cause can be traced to specific
    lines of code written by an identifiable engineer. When an RL agent fails, the situation
    is more complex:

    - **Who is responsible?** The developer who wrote the algorithm, the engineer who designed
      the reward function, the team that validated the training, or the organisation that
      deployed it?
    - **Neural network policies are opaque.** Unlike rule-based systems, you cannot easily
      explain *why* the agent took a specific action that led to failure.
    - **Regulatory gaps exist.** Current aviation and space regulations were not designed for
      learned controllers. There is no established framework for certifying RL-based systems.

    Clear chains of accountability must be established before RL systems are deployed in
    safety-critical applications.
    """)

    st.markdown("---")

    st.markdown("### 4. Sim-to-Real Transfer")
    st.markdown("""
    Our agent was trained entirely in a 2D physics simulation with perfect sensor readings
    and deterministic dynamics. Real-world deployment involves:

    - **Sensor noise:** Real accelerometers, gyroscopes, and altimeters have measurement error.
    - **Actuator lag:** Real engines do not respond instantaneously to commands.
    - **Environmental variability:** Wind, turbulence, temperature, and gravity variations.
    - **3D dynamics:** Real landing involves six degrees of freedom, not two.

    Deploying a simulation-trained policy without addressing these gaps is ethically problematic
    because the system may appear validated but could fail in conditions it was never exposed to.
    Domain randomisation and sim-to-real transfer techniques are essential but not sufficient
    to guarantee safe deployment.
    """)

    st.markdown("---")

    st.markdown("### 5. Dual-Use Concerns")
    st.markdown("""
    The reinforcement learning techniques used in this project are general-purpose. The same
    algorithms that learn to land a spacecraft safely could be applied to:

    - **Autonomous weapons:** Drone strike planning, missile guidance, target tracking.
    - **Surveillance:** Learning optimal patrol patterns or tracking strategies.
    - **Manipulation:** Learning to influence human behaviour through adaptive interfaces.

    The RL research community has a responsibility to consider how published algorithms and
    trained models might be repurposed. While our LunarLander project is educational, the
    underlying methods (DQN, TD3) are the same ones used in military and surveillance research.
    """)

    st.markdown("---")

    st.markdown("### 6. Transparency and Interpretability")
    st.markdown("""
    Neural network policies are black boxes. When our DQN agent decides to fire the main engine,
    we cannot easily explain *why* it made that decision in terms a human can audit.

    For safety-critical systems, regulators and the public may require:

    - **Explainability:** Why did the agent take this action in this state?
    - **Predictability:** Will the agent always behave the same way in similar situations?
    - **Verifiability:** Can we formally prove the agent will never enter a dangerous state?

    Current RL methods cannot provide these guarantees. This is a fundamental limitation
    that must be communicated honestly when proposing RL for real-world deployment.
    """)

    st.markdown("---")

    st.markdown("### 7. Environmental Cost of Training")
    st.markdown("""
    Training RL agents requires significant computational resources. Our project involved:

    - Training 3 algorithms across ~800 episodes each.
    - Hyperparameter sensitivity experiments (gamma, learning rate, network size).
    - Multiple failed experiments that consumed compute without producing useful models.

    At scale, RL research contributes to energy consumption and carbon emissions. Responsible
    development should consider computational efficiency, report training costs, and avoid
    unnecessary hyperparameter sweeps when prior knowledge can guide choices.
    """)

    st.markdown("---")

    st.info("""
    **Our Approach:** Throughout this project, we prioritised understanding over brute-force
    optimisation. We documented failed experiments honestly, used modest computational resources
    (CPU-only, ~10 minutes total training), and focused on educational value rather than
    achieving state-of-the-art performance.
    """)

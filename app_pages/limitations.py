import streamlit as st


def render():
    st.title("Limitations")
    st.markdown("""
    Understanding the limitations of our solution is essential for honest evaluation
    and for identifying areas where real-world deployment would require further work.
    """)

    st.markdown("---")

    st.markdown("### 1. Simulation vs Reality Gap")
    st.markdown("""
    Our entire solution was developed and tested in the Gymnasium LunarLander simulation.
    This environment is a significant simplification of real-world landing:

    | Simulation | Reality |
    |---|---|
    | 2D physics | 3D dynamics with 6 degrees of freedom |
    | Perfect sensors | Noisy IMU, altimeter, GPS readings |
    | Instant engine response | Actuator delays and thrust curves |
    | No wind or weather | Turbulence, crosswinds, dust |
    | Deterministic resets | Variable initial conditions |
    | Flat terrain | Uneven, rocky landing surfaces |

    A policy trained in this simulation would not transfer directly to a real lander
    without substantial additional work (domain randomisation, robust control, hardware-in-the-loop testing).
    """)

    st.markdown("---")

    st.markdown("### 2. Statistical Limitations")
    st.markdown("""
    - **Single random seed (42):** All experiments used one seed. Results may vary with different
      seeds. Proper scientific evaluation would require multiple seeds with confidence intervals.
    - **Limited evaluation episodes (100):** While standard for this benchmark, 100 episodes
      may not capture rare failure modes.
    - **No cross-validation:** We did not test whether hyperparameters generalise across
      different environment configurations.
    """)

    st.markdown("---")

    st.markdown("### 3. No Safety Guarantees")
    st.markdown("""
    Our trained agents have no formal safety guarantees:

    - No worst-case performance bounds.
    - No formal verification that the agent will never enter a dangerous state.
    - No constraint-based learning (e.g., constrained MDPs or safety layers).
    - The agent could behave unpredictably on out-of-distribution states it never encountered during training.

    For safety-critical deployment, techniques like constrained optimisation, safety shields,
    or formal verification would be necessary.
    """)

    st.markdown("---")

    st.markdown("### 4. Algorithm Scope")
    st.markdown("""
    We implemented 3 algorithms (DQN, Double DQN, TD3). Many other RL methods exist that
    we did not explore:

    - **PPO (Proximal Policy Optimization):** On-policy method, often more stable.
    - **SAC (Soft Actor-Critic):** Entropy-regularised, better exploration.
    - **A3C / A2C:** Parallel training for faster convergence.
    - **Model-based methods:** Could be more sample-efficient.
    - **REINFORCE / Monte Carlo policy gradient:** Simpler baselines.

    Our comparison is not exhaustive, and different algorithms might perform better
    on this or related tasks.
    """)

    st.markdown("---")

    st.markdown("### 5. Computational Constraints")
    st.markdown("""
    - **CPU-only training:** No GPU acceleration was used. Training times were 1.5-7 minutes,
      which limited the scope of hyperparameter search.
    - **No exhaustive grid search:** We tested a few key hyperparameter values rather than
      performing a comprehensive search.
    - **No Bayesian optimisation:** More sophisticated hyperparameter tuning could yield
      better results.
    - **Network size limited:** We used small networks (64-128 hidden units). Larger networks
      with GPU training might perform better but were impractical for our setup.
    """)

    st.markdown("---")

    st.markdown("### 6. Discrete vs Continuous Comparison")
    st.markdown("""
    DQN/DDQN and TD3 operate on different environment variants (discrete vs continuous action spaces).
    Directly comparing their performance is inherently limited because:

    - The action spaces are fundamentally different (4 discrete choices vs 2 continuous values).
    - The environments, while similar, have different dynamics.
    - Episode lengths and reward distributions differ between variants.

    A fairer comparison would require running all algorithms on the same environment variant,
    which is not possible since DQN cannot handle continuous actions and TD3 is designed
    for continuous control.
    """)

    st.markdown("---")

    st.markdown("### 7. Deployment Limitations")
    st.markdown("""
    This Streamlit application has practical deployment constraints:

    - **Local execution only:** Requires Python, PyTorch, and Gymnasium installed locally.
    - **Box2D dependency:** The physics engine requires compilation tools (swig, gcc),
      making cloud deployment complex.
    - **Single user:** Streamlit runs as a single-user local server.
    - **No persistent storage:** Episode results are lost when the app restarts.
    - **No API:** The trained models are not exposed as a service that other applications
      could consume.
    """)

    st.markdown("---")

    st.markdown("### 8. Theory vs Deployed Solution")
    st.warning("""
    **Key differences between our theoretical solution (CA1) and this deployment (CA2):**

    - **CA1** focused on algorithm correctness, training curves, and hyperparameter analysis
      in a research context. Results were evaluated offline via scripts.
    - **CA2** wraps the same trained models in an interactive application. The underlying
      algorithms are identical, but the deployment adds a user interface layer.
    - **No retraining occurs** in the deployed app. The models are frozen snapshots from CA1.
    - **Real-time rendering** introduces latency not present during training evaluation.
    - **User interaction** means episodes are run on-demand rather than in batch,
      which changes the performance profile.
    """)

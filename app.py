import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import matplotlib
matplotlib.use('Agg')

import streamlit as st

st.set_page_config(
    page_title="RL Lunar Lander",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 1rem;}
    h1 {font-size: 2.2rem !important;}
    h2 {font-size: 1.6rem !important;}
    h3 {font-size: 1.3rem !important;}
    .stMetric label {font-size: 0.85rem !important;}
</style>
""", unsafe_allow_html=True)

from app_pages import home, live_demo, results_analysis, ethics, limitations, about

st.sidebar.title("RL Lunar Lander")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "Home",
    "Live Demo",
    "Results & Analysis",
    "Ethics",
    "Limitations",
    "About"
])

st.sidebar.markdown("---")
st.sidebar.caption("B9AI105 - Reinforcement Learning")
st.sidebar.caption("MSc AI - Dublin Business School")

if page == "Home":
    home.render()
elif page == "Live Demo":
    live_demo.render()
elif page == "Results & Analysis":
    results_analysis.render()
elif page == "Ethics":
    ethics.render()
elif page == "Limitations":
    limitations.render()
elif page == "About":
    about.render()

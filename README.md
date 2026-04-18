# RL Lunar Landing Agent

Reinforcement learning project for autonomous lunar landing using Gymnasium and PyTorch. The repository includes from-scratch implementations of DQN, Double DQN, and TD3, along with saved training outputs, trained model checkpoints, a Streamlit demo app, and the project report.

## Project Scope

- `DQN` for discrete control on `LunarLander-v3`
- `Double DQN` for reduced Q-value overestimation on `LunarLander-v3`
- `TD3` for continuous control on `LunarLanderContinuous-v3`
- Streamlit app for interactive walkthrough, live demo, and results analysis
- Training artifacts, plots, and pretrained checkpoints under `results/`

## Repository Structure

```text
.
|-- app.py
|-- app_pages/
|-- src/
|-- results/
|-- demo_assets/
|-- requirements_app.txt
|-- ca2_report.pdf
`-- demo_presentation.html
```

## Requirements

- Python 3.11 recommended
- Box2D-compatible environment for Gymnasium LunarLander
- Dependencies listed in `requirements_app.txt`

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements_app.txt
```

## Run The Streamlit App

```bash
streamlit run app.py
```

The app includes:

- project overview
- live demo pages
- results and comparison plots
- ethics and limitations sections
- team and module information

## Train The Agents

Train DQN and Double DQN:

```bash
cd src
python train_dqn.py
```

Train only one discrete-control variant:

```bash
cd src
python train_dqn.py --dqn
python train_dqn.py --ddqn
```

Train TD3:

```bash
cd src
python train_td3.py
```

Generated checkpoints and plots are written to `results/`.

## Run The Local Demo

```bash
cd src
python demo.py
```

Optional flags:

- `--dqn`
- `--ddqn`
- `--td3`
- `--episodes 5`

## Included Outputs

- pretrained model files for DQN, Double DQN, and TD3
- training reward curves and comparison plots
- evaluation histograms and sensitivity plots
- Streamlit presentation assets
- final report in PDF and LaTeX

## Tech Stack

- Python
- Gymnasium
- PyTorch
- Streamlit
- NumPy
- Matplotlib
- Pillow
- Pandas

## Authors

- Saquib Pirjade

Module: `B9AI105 - Reinforcement Learning`  
Programme: `MSc Artificial Intelligence`  
Institution: `Dublin Business School`

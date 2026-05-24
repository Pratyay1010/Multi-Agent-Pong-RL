# Multi-Agent-Pong-RL

Multi-agent reinforcement learning experiments for competitive Pong environments using PettingZoo and PyTorch.

<p align="center">
  <img src="assets/gifs/training-demo.gif" width="700"/>
</p>

---

## Motivation

This project explores competitive multi-agent reinforcement learning through Atari Pong using self-play Deep Q-Network (DQN) agents.

The goal was to investigate how reinforcement learning agents can learn reactive control policies directly from raw visual observations in a competitive environment.

Key areas explored:
- visual perception for control
- self-play reinforcement learning
- policy learning from RGB observations
- competitive agent dynamics
- lightweight RL experimentation pipelines

The repository was designed with a minimal research-oriented structure focused on readability, modularity, and reproducible experimentation.

---

## Overview

The repository includes:
- CNN-based DQN agents
- PettingZoo Pong environment wrapper
- Experience replay training
- Target network synchronization
- Self-play multi-agent training
- Model checkpointing utilities

---

## Architecture

```text
src/
├── agents/
│   └── policy_agent.py
│
├── envs/
│   └── pong_env.py
│
└── training/
    └── train_agent.py
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Multi-Agent-Pong-RL.git
cd Multi-Agent-Pong-RL
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Windows
```bash
venv\Scripts\activate
```

### Linux / macOS
```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

Example `requirements.txt`:

```text
torch
numpy
opencv-python
gymnasium
pettingzoo[atari]
ale-py
pygame
```

---

## Method

### Observation Pipeline
- Raw RGB Atari frames
- Frame preprocessing and resizing
- CNN-based feature extraction

### Learning Strategy
- Deep Q-Network (DQN)
- Experience replay
- Target network synchronization
- Epsilon-greedy exploration

### Environment
- PettingZoo Atari Pong
- Two-agent competitive setting
- Self-play training loop

---

## Training

Run training with:

```bash
python -m src.training.train_agent
```

---

## Model Checkpoints

Saved models are stored in:

```text
checkpoints/
```

---

## Tech Stack

- Python
- PyTorch
- PettingZoo
- Gymnasium
- OpenCV
- NumPy

---

## Results

| Metric | Value |
|---|---|
| Environment | Atari Pong |
| Observation Type | RGB Frames |
| Action Space | Discrete (6) |
| Learning Method | DQN |
| Training Mode | Self-Play |

Add:
- gameplay GIFs
- reward curves
- qualitative behavior examples

inside:

```text
assets/
```

---

## Repository Structure

```text
Multi-Agent-Pong-RL/
│
├── assets/
├── checkpoints/
│
├── src/
│   ├── agents/
│   ├── envs/
│   └── training/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Future Improvements

- Double DQN
- PPO baseline comparison
- Frame stacking
- Curriculum self-play
- Evaluation benchmarking

---
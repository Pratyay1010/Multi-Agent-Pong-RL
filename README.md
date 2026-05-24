# Multi-Agent-Pong-RL

Multi-agent reinforcement learning experiments for competitive Pong environments using PettingZoo and PyTorch.



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



## Overview

The repository includes:
- CNN-based DQN agents
- PettingZoo Pong environment wrapper
- Experience replay training
- Target network synchronization
- Self-play multi-agent training
- Model checkpointing utilities



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



## Installation

Clone the repository:

```bash
git clone https://github.com/Pratyay1010/Multi-Agent-Pong-RL.git
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

Install Atari ROMs:

```bash
AutoROM --accept-license
```



## Requirements

```text
torch
numpy
opencv-python
gymnasium
pettingzoo[atari]
ale-py
pygame
autorom
```



## Methodology

### Environment Setup

The project uses the PettingZoo Atari Pong environment as a competitive multi-agent benchmark. Two independent agents interact within the same environment and learn through repeated self-play episodes.

Observations are received as raw RGB image frames with shape `(210, 160, 3)` directly from the Atari simulator. Each agent selects actions from a discrete six-action control space corresponding to standard Pong movement and firing operations.



### Observation Processing

Raw observations are preprocessed before being passed to the policy network.

The preprocessing pipeline includes:
- cropping irrelevant frame regions
- resizing frames to `84 × 84`
- preserving RGB channel information
- converting image layouts from HWC to CHW tensor format

This reduces computational complexity while retaining sufficient spatial information for policy learning.



### Deep Q-Network Architecture

The policy network is implemented using a convolutional Deep Q-Network (DQN).

The architecture consists of:
- stacked convolutional layers for visual feature extraction
- ReLU activation functions
- fully connected layers for Q-value estimation

The convolutional encoder enables the agent to learn spatial motion patterns directly from visual observations without handcrafted features.

The network outputs Q-values for all valid actions:
- NOOP
- FIRE
- RIGHT
- LEFT
- FIRE + RIGHT
- FIRE + LEFT



### Experience Replay

Training transitions are stored inside a replay buffer.

Each transition contains:
- current observation
- selected action
- reward
- next observation
- terminal state indicator

Mini-batches are sampled uniformly from replay memory during optimization. This reduces temporal correlation between consecutive samples and improves training stability.



### Target Network Synchronization

A separate target network is maintained alongside the online policy network.

The target network parameters are periodically synchronized with the policy network to stabilize temporal difference learning and reduce oscillatory Q-value updates during optimization.



### Exploration Strategy

The agents use epsilon-greedy exploration during training.

At the beginning of training:
- actions are sampled randomly with high probability

As training progresses:
- epsilon gradually decays
- the policy increasingly exploits learned Q-values

This balances:
- exploration of unseen state-action pairs
- exploitation of learned control strategies



### Optimization

The model is optimized using:
- Mean Squared Error (MSE) loss
- Adam optimizer
- gradient clipping for training stability

The temporal difference target is computed using the Bellman update equation.

For Q-learning targets:

:contentReference[oaicite:0]{index=0}

where:
- `r` is the observed reward
- `γ` is the discount factor
- `Q_target` is the target network estimate



### Self-Play Training

Both agents are trained within the same competitive environment through repeated self-play interactions.

This setup enables:
- adaptive policy evolution
- competitive behavior emergence
- dynamic strategy learning

without requiring manually scripted opponents or expert demonstrations.



### Model Persistence

The repository includes checkpoint saving and loading utilities for:
- policy network weights
- target network weights
- optimizer state
- exploration parameters

This allows interrupted training sessions to be resumed and enables reproducible evaluation experiments.



## Training

Run training with:

```bash
python -m src.training.train_agent
```



## Model Checkpoints

Saved models are stored in:

```text
checkpoints/
```



## Tech Stack

- Python
- PyTorch
- PettingZoo
- Gymnasium
- OpenCV
- NumPy



## Results

| Metric | Value |
|||
| Environment | Atari Pong |
| Observation Type | RGB Frames |
| Action Space | Discrete (6) |
| Learning Method | DQN |
| Training Mode | Self-Play |



## Repository Structure

```text
Multi-Agent-Pong-RL/
│
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



## Future Improvements

- Double DQN
- PPO baseline comparison
- Frame stacking
- Curriculum self-play
- Evaluation benchmarking

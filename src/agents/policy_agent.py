import random
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 84, 84)
            feature_dim = self.feature_extractor(dummy_input).shape[1]

        self.q_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = x / 255.0
        x = self.feature_extractor(x)
        return self.q_head(x)


class ReplayBuffer:
    def __init__(self, capacity=30000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.stack(states),
            actions,
            rewards,
            np.stack(next_states),
            dones,
        )

    def __len__(self):
        return len(self.buffer)


class PolicyAgent:
    def __init__(self, agent_id, action_space):
        self.agent_id = agent_id
        self.action_space = action_space
        self.n_actions = action_space.n

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.policy_net = QNetwork(self.n_actions).to(self.device)
        self.target_net = QNetwork(self.n_actions).to(self.device)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=1e-4,
        )

        self.replay_buffer = ReplayBuffer(capacity=10000)

        self.gamma = 0.99
        self.batch_size = 32

        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

        self.target_update_frequency = 1000
        self.train_step = 0

        self.current_state = None

    def preprocess_observation(self, obs):
        if obs is None:
            return np.zeros((84, 84, 3), dtype=np.uint8)

        cropped = obs[20:, :, :]

        resized = cv2.resize(
            cropped,
            (84, 84),
            interpolation=cv2.INTER_AREA,
        )

        return resized

    def act(self, obs):
        state = self.preprocess_observation(obs)

        self.current_state = state

        if random.random() < self.epsilon:
            return self.action_space.sample()

        state_tensor = (
            torch.from_numpy(state)
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        return q_values.argmax(dim=1).item()

    def update(self, obs, action, reward, next_obs, done):
        next_state = self.preprocess_observation(next_obs)

        self.replay_buffer.push(
            self.current_state,
            action,
            reward,
            next_state,
            done,
        )

        if len(self.replay_buffer) >= self.batch_size:
            self._train_step()

    def _train_step(self):
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.replay_buffer.sample(self.batch_size)

        states = (
            torch.from_numpy(states)
            .float()
            .permute(0, 3, 1, 2)
            .to(self.device)
        )

        next_states = (
            torch.from_numpy(next_states)
            .float()
            .permute(0, 3, 1, 2)
            .to(self.device)
        )

        actions = torch.tensor(actions).long().to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        dones = torch.tensor(dones).float().to(self.device)

        current_q_values = self.policy_net(states).gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]

            target_q_values = (
                rewards
                + self.gamma * next_q_values * (1 - dones)
            )

        loss = nn.MSELoss()(
            current_q_values,
            target_q_values,
        )

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            1.0,
        )

        self.optimizer.step()

        self.train_step += 1

        if self.train_step % self.target_update_frequency == 0:
            self.target_net.load_state_dict(
                self.policy_net.state_dict()
            )

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    def save_model(self, filepath):
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "train_step": self.train_step,
            },
            filepath,
        )

    def load_model(self, model_data):
        checkpoint = (
            model_data
            if isinstance(model_data, dict)
            else torch.load(model_data, map_location=self.device)
        )

        self.policy_net.load_state_dict(
            checkpoint["policy_net_state_dict"]
        )

        self.target_net.load_state_dict(
            checkpoint["target_net_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        self.epsilon = checkpoint.get(
            "epsilon",
            self.epsilon,
        )

        self.train_step = checkpoint.get(
            "train_step",
            self.train_step,
        )
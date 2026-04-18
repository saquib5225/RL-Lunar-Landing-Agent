import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from networks import QNetwork
from replay_buffer import ReplayBuffer


class DQNAgent:
    """
    DQN Agent with optional Double DQN mode.
    Implements epsilon-greedy exploration, experience replay, and target network.
    """

    def __init__(self, state_dim=8, action_dim=4, seed=42,
                 lr=5e-4, gamma=0.99, tau=1e-3,
                 buffer_size=100000, batch_size=64,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 update_every=4, double_dqn=False):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every
        self.double_dqn = double_dqn

        # epsilon schedule
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # local and target Q-networks
        self.q_local = QNetwork(state_dim, action_dim).to(self.device)
        self.q_target = QNetwork(state_dim, action_dim).to(self.device)

        # copy weights to target
        self.q_target.load_state_dict(self.q_local.state_dict())

        self.optimizer = optim.Adam(self.q_local.parameters(), lr=lr)

        # replay memory
        self.memory = ReplayBuffer(capacity=buffer_size)

        self.step_count = 0

    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.q_local.eval()
        with torch.no_grad():
            q_values = self.q_local(state_t)
        self.q_local.train()

        return q_values.argmax(dim=1).item()

    def step(self, state, action, reward, next_state, done):
        """Store transition and learn if it's time. Returns loss if learning happened."""
        self.memory.push(state, action, reward, next_state, done)

        self.step_count += 1

        # only learn every update_every steps and when we have enough samples
        if self.step_count % self.update_every == 0 and len(self.memory) >= self.batch_size:
            return self.learn()
        return None

    def learn(self):
        """Update Q-network using a batch from replay buffer."""
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions.astype(np.int64)).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # current Q values
        q_values = self.q_local(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # compute targets
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: select action with local, evaluate with target
                best_actions = self.q_local(next_states_t).argmax(dim=1)
                q_next = self.q_target(next_states_t).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            else:
                # vanilla DQN: just take max from target network
                q_next = self.q_target(next_states_t).max(dim=1)[0]

            targets = rewards_t + self.gamma * q_next * (1 - dones_t)

        # Huber loss instead of MSE — more stable with large TD errors
        loss = F.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # clip gradients to prevent instability
        torch.nn.utils.clip_grad_norm_(self.q_local.parameters(), 1.0)
        self.optimizer.step()

        # soft update target network
        self.soft_update()

        return loss.item()

    def soft_update(self):
        """Polyak averaging to slowly update target network."""
        for target_param, local_param in zip(self.q_target.parameters(), self.q_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)

    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        torch.save({
            'q_local': self.q_local.state_dict(),
            'q_target': self.q_target.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.q_local.load_state_dict(checkpoint['q_local'])
        self.q_target.load_state_dict(checkpoint['q_target'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']

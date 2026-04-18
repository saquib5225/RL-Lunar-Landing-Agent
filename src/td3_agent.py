import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from replay_buffer import ReplayBuffer


class Actor(nn.Module):
    """Deterministic policy network for TD3. Maps state -> action."""

    def __init__(self, state_dim=8, action_dim=2, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # tanh squashes to [-1, 1] which matches the action space
        return torch.tanh(self.out(x))


class Critic(nn.Module):
    """Q-network for TD3. Takes (state, action) -> Q-value.
    TD3 uses twin critics so we define two Q-networks in one module.
    """

    def __init__(self, state_dim=8, action_dim=2, hidden_dim=256):
        super(Critic, self).__init__()
        # Q1
        self.fc1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.out_q1 = nn.Linear(hidden_dim, 1)

        # Q2
        self.fc1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.out_q2 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)

        q1 = F.relu(self.fc1_q1(sa))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.out_q1(q1)

        q2 = F.relu(self.fc1_q2(sa))
        q2 = F.relu(self.fc2_q2(q2))
        q2 = self.out_q2(q2)

        return q1, q2

    def q1_forward(self, state, action):
        """Only compute Q1 -- used for actor update."""
        sa = torch.cat([state, action], dim=-1)
        q1 = F.relu(self.fc1_q1(sa))
        q1 = F.relu(self.fc2_q1(q1))
        return self.out_q1(q1)


class TD3Agent:
    """
    Twin Delayed DDPG (TD3) agent for continuous control.

    Key improvements over DDPG:
    1. Twin critics -- take the min of two Q estimates to reduce overestimation
    2. Delayed actor updates -- update actor less frequently than critic
    3. Target policy smoothing -- add noise to target actions
    """

    def __init__(self, state_dim=8, action_dim=2, seed=42,
                 actor_lr=1e-3, critic_lr=1e-3,
                 gamma=0.99, tau=0.005,
                 buffer_size=200000, batch_size=256,
                 policy_noise=0.2, noise_clip=0.5,
                 exploration_noise=0.1,
                 policy_delay=2, hidden_dim=256,
                 learning_starts=10000):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise
        self.policy_delay = policy_delay
        self.learning_starts = learning_starts

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # actor and target
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # twin critics and targets
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.memory = ReplayBuffer(capacity=buffer_size)
        self.total_steps = 0
        self.learn_step = 0

    def select_action(self, state, training=True):
        """Select action. Add Gaussian noise during training for exploration."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy().flatten()

        if training:
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action = action + noise
            action = np.clip(action, -1.0, 1.0)

        return action

    def step(self, state, action, reward, next_state, done):
        """Store transition and learn every 2 steps (to save compute)."""
        self.memory.push(state, action, reward, next_state, done)
        self.total_steps += 1

        if len(self.memory) >= self.learning_starts and self.total_steps % 2 == 0:
            self.learn()

    def learn(self):
        """Update critic and (delayed) actor using TD3 algorithm."""
        self.learn_step += 1
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        # actions from buffer are stored as int64 by default in our replay buffer
        # but for continuous they should be float
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            # target policy smoothing: add clipped noise to target actions
            noise = (torch.randn_like(actions_t) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states_t) + noise).clamp(-1.0, 1.0)

            # twin critics: take the minimum
            target_q1, target_q2 = self.critic_target(next_states_t, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target = rewards_t + self.gamma * target_q * (1 - dones_t)

        # update both critics
        current_q1, current_q2 = self.critic(states_t, actions_t)
        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # delayed actor update
        if self.learn_step % self.policy_delay == 0:
            # actor loss: maximise Q1(s, actor(s))
            actor_loss = -self.critic.q1_forward(states_t, self.actor(states_t)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # soft update targets
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic, self.critic_target)

        return critic_loss.item()

    def soft_update(self, source, target):
        """Polyak averaging for target networks."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data)

    def save(self, filepath):
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

import torch
import torch.nn.functional as F

from config import Config
from ddpg_agent import Agent
from replay_buffer import ReplayBuffer


class MultiAgentDDPG():
    """Manage multi agents while interacting with the environment."""

    def __init__(self):
        super(MultiAgentDDPG, self).__init__()
        self.config = Config()
        self.agents = [Agent() for _ in range(self.config.num_agents)]
        self.buffer = ReplayBuffer()

    def act(self, state):
        """Obtain actions for given state of all agents."""
        actions = [agent.act(obs) for agent, obs in zip(self.agents, state)]
        return actions

    def actions_target(self, states):
        """Actors' target model. Frozen copy of agent, updated infrequently."""
        batch_size = self.config.batch_size
        num_agents = self.config.num_agents
        action_size = self.config.action_size

        with torch.no_grad():
            actions = torch.empty(
                (batch_size, num_agents, action_size), device=self.config.device)
            for idx, agent in enumerate(self.agents):
                actions[:, idx] = agent.actor_target(states[:, idx])
        return actions

    def actions_local(self, states, agent_id):
        """Actors' local model. Local copy of agent."""
        batch_size = self.config.batch_size
        num_agents = self.config.num_agents
        action_size = self.config.action_size

        actions = torch.empty(
            (batch_size, num_agents, action_size), device=self.config.device)
        for idx, agent in enumerate(self.agents):
            action = agent.actor_local(states[:, idx])
            if not idx == agent_id:
                action.detach()
            actions[:, idx] = action
        return actions

    def store(self, state, actions, rewards, next_state):
        """Store experience in replay buffer and use random sample from buffer to learn."""
        self.buffer.store(state, actions, rewards, next_state)

        # Learn, if enough samples are available in memory
        if len(self.buffer) >= self.config.batch_size:
            self.learn()

    def learn(self):
        """Update policy and value parameters using sample of experience tuples.
        Q_targets / y = r + γ * critic_target(next_state, actions_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) --> Q-value
        """
        batch_size = self.config.batch_size
        for agent_id, agent in enumerate(self.agents):
            # sample a batch of experiences
            states, actions, rewards, next_states = self.buffer.sample()
            # stack the agents' variables to feed the networks
            obs = states.view(batch_size, -1)
            actions = actions.view(batch_size, -1)
            next_obs = next_states.view(batch_size, -1)
            # Consider only the rewards for this agent
            r = rewards[:, agent_id].unsqueeze_(1)

            ##### Train The Critic Network #####

            with torch.no_grad():
                # Get predicted next-state actions (for all agents) from actor_target model
                next_actions = self.actions_target(next_states)
                # stack the agents' variables to feed the networks
                next_actions = next_actions.view(batch_size, -1)
                # Get predicted next-state Q-Values (for all agents) from critic_target model
                next_q_val = agent.critic_target(next_obs, next_actions)
                # Compute Q targets for current states (y_i)
                y = r + self.config.gamma * next_q_val
            # Compute critic loss
            agent.critic_optimizer.zero_grad()
            q_value_predicted = agent.critic_local(obs, actions)
            loss = F.mse_loss(q_value_predicted, y)
            # Minimise the loss
            loss.backward()
            agent.critic_optimizer.step()

            ##### Train The Actor Network #####

            agent.actor_optimizer.zero_grad()
            # Compute actor loss
            actions_local = self.actions_local(states, agent_id)
            actions_local = actions_local.view(batch_size, -1)
            q_value_predicted = agent.critic_local(obs, actions_local)
            loss = -q_value_predicted.mean()
            # Minimize the loss
            loss.backward()
            agent.actor_optimizer.step()

        ###### Update Target Networks #####
        for agent in self.agents:
            agent.soft_update()

    def reset_noise(self):
        """Reset the noise to mean."""
        for agent in self.agents:
            agent.reset_noise()

    def state_dict(self):
        return [agent.actor_local.state_dict() for agent in self.agents]

    def load_state_dict(self, state_dicts):
        for agent, state_dict in zip(self.agents, state_dicts):
            agent.actor_local.load_state_dict(state_dict)

    def lr_step(self):
        """Decay the learning rate of agents."""
        for agent in self.agents:
            agent.lr_step()

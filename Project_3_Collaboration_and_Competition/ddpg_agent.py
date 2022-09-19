import copy
import random
import numpy as np
import torch
import torch.optim as optim
from config import Config
from model import Actor, Critic


class Agent():
    """Interacts with and learns from the environment."""
    def __init__(self):
        self.config = Config()
        random.seed(self.config.seed)

        # Actor Network
        self.actor_local = Actor()
        self.actor_target = Actor()
        # Returns a dictionary containing a whole state of the module. Both parameters and persistent buffers (e.g. running averages) are included.
        local_state_dict = self.actor_local.state_dict()
        # Copies parameters and buffers from state_dict into this module and its descendants.
        self.actor_target.load_state_dict(local_state_dict)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(),
            lr=self.config.actor_lr)
        self.actor_lr_scheduler = optim.lr_scheduler.StepLR(
            self.actor_optimizer,
            step_size=self.config.lr_sched_step,
            gamma=self.config.lr_sched_gamma)

        # Critic Network
        self.critic_local = Critic()
        self.critic_target = Critic()
        # Returns a dictionary containing a whole state of the module. Both parameters and persistent buffers (e.g. running averages) are included.
        local_state_dict = self.critic_local.state_dict()
        # Copies parameters and buffers from state_dict into this module and its descendants.
        self.critic_target.load_state_dict(local_state_dict)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(),
            lr=self.config.critic_lr)
        self.critic_lr_scheduler = optim.lr_scheduler.StepLR(
            self.critic_optimizer,
            step_size=self.config.lr_sched_step,
            gamma=self.config.lr_sched_gamma)

        # Initialize a noise process
        self.noise = OUNoise()

    def soft_update(self):
        """Soft update actor and critic parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        tau = self.config.tau
        for target_param, local_param \
                in zip(self.actor_target.parameters(),
                       self.actor_local.parameters()):
            target_param.data.copy_(
                tau * local_param.data
                + (1.0 - tau) * target_param.data)
        for target_param, local_param \
                in zip(self.critic_target.parameters(),
                       self.critic_local.parameters()):
            target_param.data.copy_(
                tau * local_param.data
                + (1.0 - tau) * target_param.data)

    def act(self, state):
        """Returns actions for given state as per current policy."""
        # deactivate autograd engine
        with torch.no_grad():
            # obtain the best-believed action
            self.actor_local.eval()
            # create a tensor from numpy array
            state = torch.from_numpy(state).float()
            state.to(self.config.device)
            # obtain action by passing state to Actor
            action = self.actor_local(state).data.cpu().numpy()
            # train actor local
            self.actor_local.train()

        if self.config.noise:
            # Add noise to action to explore the environment
            action += self.noise.sample()
            np.clip(action, a_min=-1, a_max=1, out=action)

        return action

    def lr_step(self):
        """ Perform decay of learning rate """
        self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()

    def reset_noise(self):
        """ Reset the noise to mean. """
        self.noise.reset()


class OUNoise:
    """Ornstein-Uhlenbeck process."""
    
    def __init__(self, mu=0.):
        """Initialize parameters and noise process."""
        self.config = Config()
        random.seed(self.config.seed)
        self.mu = mu * np.ones(self.config.action_size)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        random_array = [random.random() for i in range(len(x))]
        dx = self.config.noise_theta * (self.mu - x) \
             + self.config.noise_sigma * np.array(random_array)
        self.state = x + dx
        return self.state

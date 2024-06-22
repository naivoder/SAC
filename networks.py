import torch
import numpy as np


class ValueNetwork(torch.nn.Module):
    def __init__(
        self,
        input_shape,
        h1_size,
        h2_size,
        learning_rate=3e-4,
        chkpt_path="weights/value.pt",
    ):
        super(ValueNetwork, self).__init__()
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.lr = learning_rate
        self.checkpoint_path = chkpt_path

        self.fc1 = torch.nn.Linear(*input_shape, h1_size)
        self.fc2 = torch.nn.Linear(h1_size, h2_size)
        self.V = torch.nn.Linear(h2_size, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = torch.nn.functional.relu(self.fc1(state))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.V(x)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_path)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_path))


class CriticNetwork(torch.nn.Module):
    def __init__(
        self,
        input_shape,
        n_actions,
        h1_size,
        h2_size,
        learning_rate=3e-4,
        chkpt_path="weights/critic.pt",
    ):
        super(CriticNetwork, self).__init__()
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.lr = learning_rate
        self.checkpoint_path = chkpt_path

        self.fc1 = torch.nn.Linear(np.prod(input_shape) + n_actions, h1_size)
        self.fc2 = torch.nn.Linear(h1_size, h2_size)
        self.Q = torch.nn.Linear(h2_size, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, actions):
        x = torch.concatenate((state, actions), dim=1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.Q(x)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_path)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_path))


class ActorNetwork(torch.nn.Module):
    def __init__(
        self,
        input_shape,
        n_actions,
        h1_size,
        h2_size,
        min_action,
        max_action,
        learning_rate=3e-5,
        reparam_noise=1e-6,
        chkpt_path="weights/actor.pt",
    ):
        super(ActorNetwork, self).__init__()
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.lr = learning_rate
        self.min_action = min_action
        self.max_action = max_action
        self.reparam_noise = reparam_noise
        self.checkpoint_path = chkpt_path

        self.fc1 = torch.nn.Linear(*input_shape, self.h1_size)
        self.fc2 = torch.nn.Linear(self.h1_size, self.h2_size)
        self.mean = torch.nn.Linear(self.h2_size, n_actions)
        self.std = torch.nn.Linear(self.h2_size, n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self.action_scale = torch.FloatTensor((self.max_action - self.min_action) / 2.).to(self.device)
        self.action_bias = torch.FloatTensor((self.max_action + self.min_action) / 2.).to(self.device)

    def forward(self, state):
        x = torch.nn.functional.relu(self.fc1(state))
        x = torch.nn.functional.relu(self.fc2(x))
        mean = self.mean(x)
        std = self.std(x)
        std = torch.clamp(std, -20, 2)
        return mean, std

    def sample_normal(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        probs = torch.distributions.Normal(mu, std)
        
        x_t = probs.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)

        action = y_t * self.action_scale + self.action_bias

        log_probs = probs.log_prob(x_t)
        log_probs -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        # for deterministic policy return mu instead of action
        return action, log_probs

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_path)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_path))

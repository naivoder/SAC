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
        max_action,
        learning_rate=3e-5,
        reparam_noise=1e-6,
        chkpt_path="weights/actor.pt",
    ):
        super(ActorNetwork, self).__init__()
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.lr = learning_rate
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

    def forward(self, state):
        x = torch.nn.functional.relu(self.fc1(state))
        x = torch.nn.functional.relu(self.fc2(x))
        mean = self.mean(x)
        std = self.std(x)
        std = torch.clamp(std, self.reparam_noise, 1)
        return mean, std

    def sample_normal(self, state, reparam=False):
        # could also experiment with multivariate normal
        mu, sigma = self.forward(state)
        # print(state, mu)
        action_probs = torch.distributions.Normal(mu, sigma)
        actions = action_probs.rsample() if reparam else action_probs.sample()
        action = torch.tanh(actions) * torch.tensor(self.max_action).to(self.device)

        # add reparam noise since squared action can = 1 (can't take log of 0)
        log_probs = action_probs.log_prob(actions)
        log_probs -= torch.log(1 - action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(-1, keepdim=True)

        # for deterministic policy return mu instead of action
        return action, log_probs

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_path)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_path))

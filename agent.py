import torch
import numpy as np
import networks
import memory


class SACAgent(torch.nn.Module):
    def __init__(
        self,
        input_dims,
        action_space,
        tau=5e-3,
        reward_scale=2,
        batch_size=256,
        lr=3e-4,
        gamma=0.99,
        h1_size=256,
        h2_size=256,
        mem_size=int(1e6),
    ):
        super(SACAgent, self).__init__()
        self.input_dims = input_dims
        self.n_actions = action_space.shape[0]
        self.min_action = action_space.low
        self.max_action = action_space.high
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.tau = tau
        self.lr = lr
        self.gamma = gamma
        self.h1_size = h1_size
        self.h2_size = h2_size
        self.mem_size = mem_size

        self.memory = memory.ReplayBuffer(
            self.input_dims, self.n_actions, self.mem_size
        )

        self.Q1 = networks.CriticNetwork(
            self.input_dims,
            self.n_actions,
            self.h1_size,
            self.h2_size,
            learning_rate=self.lr,
            chkpt_path="weights/critic_1.pt",
        )
        self.Q2 = networks.CriticNetwork(
            self.input_dims,
            self.n_actions,
            self.h1_size,
            self.h2_size,
            learning_rate=self.lr,
            chkpt_path="weights/critic_2.pt",
        )
        self.V = networks.ValueNetwork(
            self.input_dims,
            self.h1_size,
            self.h2_size,
            learning_rate=self.lr,
            chkpt_path="weights/value.pt",
        )
        self.V_target = networks.ValueNetwork(
            self.input_dims,
            self.h1_size,
            self.h2_size,
            learning_rate=self.lr,
            chkpt_path="weights/value_target.pt",
        )
        self.Actor = networks.ActorNetwork(
            self.input_dims,
            self.n_actions,
            self.h1_size,
            self.h2_size,
            learning_rate=self.lr,
            max_action=self.max_action,
            chkpt_path="weights/actor.pt",
        )

        self.update_network_params(tau=1)

    def choose_action(self, state):
        self.Actor.eval()
        state = torch.tensor(np.array(state)).to(torch.float32).to(self.Actor.device)
        action, _ = self.Actor.sample_normal(state)
        self.Actor.train()
        return action.cpu().detach().numpy()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        states, actions, rewards, next_states, done = self.memory.sample(
            self.batch_size
        )
        states = torch.tensor(states).to(torch.float32).to(self.Actor.device)
        next_states = torch.tensor(next_states).to(torch.float32).to(self.Actor.device)
        actions = torch.tensor(actions).to(torch.float32).to(self.Actor.device)
        rewards = torch.tensor(rewards).to(torch.float32).to(self.Actor.device)
        done = torch.tensor(done).to(torch.bool).to(self.Actor.device)

        self._value_loss(states)
        self._actor_loss(states)

        # get value of next state according to target network
        next_state_values = self.V_target(next_states).view(-1)
        next_state_values[done] = 0.0

        # get critic values of states with old policy (from memory)
        q1 = self.Q1(states, actions).view(-1)
        q2 = self.Q2(states, actions).view(-1)

        # calculate q_hat term i.e. scaled discounted returns
        q_hat = self.reward_scale * rewards + self.gamma * (next_state_values)

        # calculate critic loss
        self.Q1.optimizer.zero_grad()
        self.Q2.optimizer.zero_grad()
        q1_loss = 0.5 * torch.nn.functional.mse_loss(q1, q_hat)
        q2_loss = 0.5 * torch.nn.functional.mse_loss(q2, q_hat)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.Q1.optimizer.step()
        self.Q2.optimizer.step()

        self.update_network_params()

    def _value_loss(self, states):
        values = self.V(states).view(-1)

        # get min critic value of states with current policy
        current_policy, log_probs = self.Actor.sample_normal(states)
        log_probs = log_probs.view(-1)
        q1 = self.Q1(states, current_policy)
        q2 = self.Q2(states, current_policy)
        critic_value = torch.min(q1, q2).view(-1)

        # calculate value loss
        self.V.optimizer.zero_grad()
        value_targets = critic_value - log_probs
        value_loss = 0.5 * torch.nn.functional.mse_loss(values, value_targets)
        value_loss.backward(retain_graph=True)  # avoid coupling between loss functions
        self.V.optimizer.step()

    def _actor_loss(self, states):
        # get min critic value of states with new policy (reparameterized)
        new_policy, log_probs = self.Actor.sample_normal(states)
        log_probs = log_probs.view(-1)
        q1 = self.Q1(states, new_policy)
        q2 = self.Q2(states, new_policy)
        critic_value = torch.min(q1, q2).view(-1)

        # calculate actor loss
        actor_loss = torch.mean(log_probs - critic_value)
        self.Actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.Actor.optimizer.step()

    def update_network_params(self, tau=None):
        if tau is None:
            tau = self.tau

        value_params = dict(self.V.named_parameters())
        target_value_params = dict(self.V_target.named_parameters())
        for name in value_params:
            value_params[name] = (
                tau * value_params[name].clone()
                + (1 - tau) * target_value_params[name].clone()
            )
        self.V_target.load_state_dict(value_params)

    def save_checkpoints(self):
        self.V.save_checkpoint()
        self.V_target.save_checkpoint()
        self.Q1.save_checkpoint()
        self.Q2.save_checkpoint()
        self.Actor.save_checkpoint()

    def load_checkpoints(self):
        self.V.load_checkpoint()
        self.V_target.load_checkpoint()
        self.Q1.load_checkpoint()
        self.Q2.load_checkpoint()
        self.Actor.load_checkpoint()

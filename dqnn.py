import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from memory import Memory, Transition


class DeepQNetwork(nn.Module):
    def __init__(self, n_observations: int, n_actions: int, lr: float):
        super(DeepQNetwork, self).__init__()
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.n_observations, 32)
        self.fc2 = nn.Linear(32, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Agent:
    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        batch_size: int = 128,
        lr: float = 0.001,
        max_mem_size: int = 100000,
        eps_start: float = 0.95,
        eps_end: float = 0.05,
        eps_dec: float = 1000,
        gamma: float = 0.99,
        tau: float = 0.005,
    ):
        self.batch_size = batch_size
        self.lr = lr
        self.mem_size = max_mem_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.gamma = gamma
        self.tau = tau

        # Action indeces
        self.action_space = [i for i in range(n_actions)]

        # Init Networks
        self.policy_net = DeepQNetwork(n_observations, n_actions, lr=lr)
        self.target_net = DeepQNetwork(n_observations, n_actions, lr=lr)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Init memory
        self.memory = Memory(max_mem_size)

        # Init step counter
        self.steps_done = 0

    def choose_action(self, observation):
        # NOTE: for this env an observation is an array [x_pos, vel]

        # NOTE: this could be any decrement (in this case is exp)
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_dec
        )
        # Linear interpolation
        # eps = np.interp(
        #     self.steps_done, [0, self.eps_dec], [self.eps_start, self.eps_end]
        # )
        self.steps_done += 1

        if np.random.random() > eps:
            with torch.no_grad():
                return self.policy_net(observation).max(1).indices.view(1, 1)

        return torch.tensor(
            [[np.random.choice(self.action_space)]],
            device=self.policy_net.device,
            dtype=torch.long,
        )

    def learn(self):
        # Just return if memory is not filled with at least batch_size transitions
        if len(self.memory) < self.batch_size:
            return

        # Take random sample from memory
        transitions = self.memory.sample(self.batch_size)

        # Convert batch-array of Transitions to Transition of batch-arrays
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.policy_net.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # print(self.policy_net(state_batch).shape, action_batch.shape, sep="\n\n\n")

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken. These are the actions which would've been taken for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.policy_net.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute loss
        loss = self.policy_net.criterion(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        # Optimize the model
        self.policy_net.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        # torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.policy_net.optimizer.step()

    def update_target_net(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_v = target_net_state_dict[key]
            policy_v = policy_net_state_dict[key]
            updated_target_t = policy_v * self.tau + target_v * (1 - self.tau)
            target_net_state_dict[key] = updated_target_t
        self.target_net.load_state_dict(target_net_state_dict)

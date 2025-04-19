import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import MSELoss
from torch.optim.adam import Adam

from replay_memory import ExperienceReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Q_value(nn.Module):
    """ Class for the DQN model aproximating the value function. """

    def __init__(self, n_inputs, n_outputs):
        super(Q_value, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.hidden = nn.Linear(self.n_inputs, 256)
        #nn.init.kaiming_normal_(self.hidden.weight)
        self.hidden.bias.data.zero_()

        self.output = nn.Linear(256, self.n_outputs)
        nn.init.xavier_uniform_(self.output.weight)
        self.output.bias.data.zero_()

    def forward(self, states, goals):
        x = torch.cat([states, goals], dim=-1)
        x = F.relu(self.hidden(x))
        return self.output(x)



class Agent:
    """ Class for a standar DQN agent. """

    def __init__(self, n_bits, lr, memory_size, batch_size, gamma):
        """
        Initialize the agent.

        Parameters:
        n_bits (int): Number of bits in the state and goal.
        lr (float): Learning rate for the optimizer.
        memory_size (int): Size of the experience replay buffer.
        batch_size (int): Batch size for training.
        gamma (float): Discount factor for future rewards.
        """



        # Set initial epsilon and decay after each trial parameters for the epsilon-greedy policy 
        self.epsilon = 1.0
        self.epsilon_decay = 0.999

        # Store parameters as member variables

        # Theres 2^n possible states, where we can choose from n to flip
        self.obs_dim, self.act_dim = 2 * n_bits, n_bits  # 2*nbits as we concatenate states with goals
        self.batch_size = batch_size
        self.gamma = gamma

        # Initialize Replay Buffer
        self.replay_buffer = ExperienceReplayBuffer(memory_size)

        # Create the DQN network as a member variable
        self.dqn_net = Q_value(self.obs_dim, self.act_dim).to(device)

        # Create the target DQN network and copy weights
        self.dqn_target_net = Q_value(self.obs_dim, self.act_dim).to(device)

        # Copy parameters from DQN to target DQN
        for target_param, param in zip(self.dqn_target_net.parameters(), self.dqn_net.parameters()):
            target_param.data.copy_(param.data)

        # Set up optimizer
        self.opt = Adam(self.dqn_net.parameters(), lr=lr)

    def choose_action(self, states, goals):
        """
        Choose an action based on epsilon-greedy policy.

        Parameters:
        states (array): Current state of the environment.
        goals (array): Target state of the environment.

        Returns:
        int: The chosen action.
        """

        # Compute Q_values
        Q_values = self.dqn_net(
            torch.tensor(states, dtype=torch.float32).to(device),
            torch.tensor(goals, dtype=torch.float32).to(device)
        ).cpu().detach().numpy()

        # Return according to epsilon-greedy policy
        return np.random.choice(np.arange(self.act_dim)) if (np.random.random() <= self.epsilon) else np.argmax(Q_values)


    def update_epsilon(self):
        """
        Update the epsilon value for the epsilon-greedy policy.
        """
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0)

    def store(self, state, action, reward, terminated, truncated, next_state, goal):
        """
        Store a transition in the replay memory.

        Parameters:
        state (array): Current state.
        action (int): Action taken.
        reward (float): Reward received.
        terminated (bool): Whether the episode has terminated.
        truncated (bool): Whether the episode has truncated.
        next_state (array): Next state.
        goal (array): Target state.
        """

        self.replay_buffer.store(state.copy(), action, reward, terminated, truncated, next_state.copy(), goal.copy())


    def learn(self):
        """
        Perform a learning step by sampling a batch from the replay memory and updating the model.

        Returns:
        float: The loss value.
        """


        # Get the data
        state_batch, action_batch, reward_batch, terminated_batch, truncated_batch, next_state_batch, goal_batch = self.replay_buffer.sample(
            self.batch_size)

        # Move data to Tensor and also to device to take profit of GPU if available
        state_batch = torch.FloatTensor(state_batch).to(device)
        action_batch = torch.Tensor(action_batch).to(dtype=torch.long).to(device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        reward_batch = torch.FloatTensor(reward_batch).to(device).unsqueeze(1)
        terminated_batch = torch.FloatTensor(terminated_batch).to(dtype=torch.long).to(device).unsqueeze(1)
        truncated_batch = torch.FloatTensor(truncated_batch).to(dtype=torch.long).to(device).unsqueeze(1)
        goal_batch = torch.FloatTensor(goal_batch).to(dtype=torch.long).to(device)

        # Compute the Q-values for the next_state_batch to compute the target
        q_targets_next = torch.max(self.dqn_target_net(next_state_batch, goal_batch).detach(), dim=1, keepdim=True)[
            0]

        # Q(s, a) is standard but when episode terminates target should be only the reward.
        target = reward_batch + (1 - terminated_batch) * self.gamma * q_targets_next

        # Compute the Q-values for the state_batch according to the DQN network
        q_expected = torch.gather(self.dqn_net(state_batch, goal_batch), 1, action_batch)

        #Compute loss
        criterion = MSELoss()
        loss = criterion(q_expected, target)


        # Update the model according to the loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # Update the target network
        self.soft_update_of_target_network(self.dqn_net, self.dqn_target_net)

        return loss.item()  # Return the loss value just to store it and print it in the main loop

    @staticmethod
    def soft_update_of_target_network(local_model, target_model, tau=0.05):
        """
        Soft update of the target network parameters.

        Parameters:
        local_model (torch.nn.Module): The local model.
        target_model (torch.nn.Module): The target model.
        tau (float): Interpolation parameter for soft update.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

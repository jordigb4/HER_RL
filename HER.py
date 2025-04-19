from pygments.formatters import terminal
import numpy as np
from bit_flip_env import Env
from agent import Agent


# n_bits is the length of the bit string. You should try different values to reproduce
# the results of figure 1 of the paper.
n_bits = 40

# Standard hyperparameters
lr = 1e-3
gamma = 0.98
MAX_EPISODE_NUM = 20000
memory_size = 1e+6
batch_size = 128

# HER specific hyperparameters
k_future = 4
OPT_STEPS = 4

# Print the number of bits of the string
print(f"Number of bits:{n_bits}")

# Create the environment and the agent
env = Env(n_bits)
agent = Agent(n_bits=n_bits, lr=lr, memory_size=memory_size, batch_size=batch_size, gamma=gamma)

# This variable is used to calculate the number of times the agent reaches the goal using optimal actions
optimal=0
# This variable is used to calculate the number of times the agent achieves the goal state
solved=0

for episode_num in range(1,MAX_EPISODE_NUM+1):
    # state is inital state and goal is the target state for the episode.
    state, goal = env.reset()
    # mincost is the number of bits that are different between the state and the goal, so the minimum cost to reach the goal
    mincost = state.shape[0]-sum(goal == state) # Minimum cost to reach the goal (number of different bits between the state and the goal)
    episode_reward = 0 # Should maintain for statistics the sum of the rewards obtained in the episode
    episode = [] # Should maintain the transitions of the episode
    done = False

    # Collect an episode (until done)
    while not done:


        #Sample an action using agent A
        action = agent.choose_action(state, goal)

        # Execute action and observe new state
        next_state, reward, terminated, truncated, _ = env.step(action)


        episode.append((
            state.copy(),
            action,
            reward,
            terminated,
            truncated,
            next_state.copy(),
            goal.copy()
        ))

        # Sum the reward of the episode
        episode_reward += reward

        # Check if the episode has finished
        done = terminated or truncated

        # Now state is next_state (iterate)
        state = next_state.copy()

    # Check if the episode was solved optimally
    if mincost+episode_reward >=1: # When the episode ends, we check if the agent reached the goal state optimally (notice episode_reward is negative and the sum of the rewards collected)
        optimal=optimal+1

    # Loop to Store the episode in the agent's memory
    for i, transition in enumerate(episode):
        # Store transitions in Experience Buffer
        agent.store(*transition)

        # Transition is of the form (state, action, reward, terminated, truncated, next_state, goal)


        # Sample future states
        future_samples = min(k_future, len(episode) - i - 1)

        if future_samples > 0:
            # Sample indices from future transitions
            future_indices = np.random.choice(np.arange(i + 1, len(episode)), size=future_samples, replace=False)
            for idx in future_indices:
                # Get the state from the future transition
                future_state = episode[idx][0]
                new_goal = future_state.copy()
                # See if next_state matches the new goal
                new_reward = 0.0 if np.array_equal(transition[5], new_goal) else -1.0
                new_terminated = np.array_equal(transition[5], new_goal)
                new_transition = (
                    transition[0].copy(),
                    transition[1].copy(),
                    new_reward,
                    new_terminated,
                    False,
                    transition[5].copy(),
                    new_goal
                )
                agent.store(*new_transition)


    # Learn from the episode. All work is done in the agent.learn() method you have implemented earlier
    for _ in range(OPT_STEPS):
        loss = agent.learn()


    # Update epsilon
    agent.update_epsilon()

    # Update and print the results each 500 episodes
    if episode_num == 1:
        global_running_r = episode_reward
    else:
        global_running_r = 0.99 * global_running_r + 0.01 * episode_reward
    if episode_num % 500 == 0:
        print(f"Ep:{episode_num}| "
                f"Ep_r:{episode_reward:3.3f}| "
                f"Ep_running_r:{global_running_r:3.3f}| "
                f"Epsilon:{agent.epsilon:3.3f}| "
                f"Mem_size:{len(agent.replay_buffer)}| "
                f"Optimal:{100*optimal/500:.2f}%")
        optimal=0


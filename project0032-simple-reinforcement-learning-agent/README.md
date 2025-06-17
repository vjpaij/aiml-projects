### Description:

Reinforcement Learning (RL) is a learning paradigm where an agent learns to make decisions by interacting with an environment to maximize cumulative rewards. In this project, we implement a basic Q-learning agent to solve the classic FrozenLake environment using OpenAI Gym.

- How an agent learns by interacting with an environment
- Balancing exploration vs. exploitation
- Storing and updating Q-values for state-action pairs

## Q-Learning on FrozenLake-v1 (OpenAI Gym)

This example demonstrates how to apply Q-learning to the classic reinforcement learning problem, **FrozenLake-v1**, using the OpenAI Gym toolkit.

### Environment Setup

FrozenLake is a 4x4 grid world where the agent must navigate from the start (S) to the goal (G), avoiding holes (H). The surface is slippery, making it a stochastic environment.

```bash
# Install OpenAI Gym
pip install gym==0.26.2
```

### Code Explanation

```python
import gym
import numpy as np
import random

# Create the FrozenLake environment (4x4, slippery by default)
env = gym.make("FrozenLake-v1", is_slippery=True)

# Initialize the Q-table: rows = states, columns = actions
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Hyperparameters
alpha = 0.8        # Learning rate
gamma = 0.95       # Discount factor
epsilon = 1.0      # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 2000    # Total training episodes
max_steps = 100    # Max steps per episode

# Training loop
for episode in range(episodes):
    state = env.reset()[0]
    done = False

    for _ in range(max_steps):
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore: random action
        else:
            action = np.argmax(q_table[state])  # Exploit: best known action

        # Take action and observe result
        new_state, reward, done, truncated, info = env.step(action)

        # Q-learning update rule
        q_table[state, action] += alpha * (
            reward + gamma * np.max(q_table[new_state]) - q_table[state, action]
        )

        state = new_state

        if done:
            break

    # Reduce exploration rate
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# Print the trained Q-table
print("Trained Q-table:\n", q_table)

# Evaluate the trained policy
successes = 0
for episode in range(100):
    state = env.reset()[0]
    done = False

    for _ in range(max_steps):
        action = np.argmax(q_table[state])  # Always exploit best action
        new_state, reward, done, truncated, info = env.step(action)
        state = new_state

        if done:
            successes += reward  # reward is 1 if goal is reached
            break

print(f"\nSuccess rate over 100 evaluation episodes: {successes}%")
```

### Key Concepts

* **Q-table**: Stores estimated future rewards for state-action pairs.
* **Exploration vs Exploitation**: Balances trying new actions (explore) vs using known best (exploit).
* **Learning Rate (alpha)**: Controls how much new information overrides old.
* **Discount Factor (gamma)**: Weighs future rewards compared to immediate rewards.
* **Epsilon Decay**: Gradually reduces exploration over time.

This basic Q-learning implementation enables an agent to learn an optimal policy for reaching the goal in FrozenLake-v1, even in a stochastic environment.

import gymnasium as gym
import numpy as np
import random
import time

# -------------------------
# Create Environment
# -------------------------
env = gym.make("CliffWalking-v1")

# Hyperparameters
gamma = 0.99
alpha = 0.5
epsilon = 0.1
episodes = 500

# Q Table
Q = np.zeros((env.observation_space.n, env.action_space.n))

# -------------------------
# Epsilon Greedy Policy
# -------------------------
def epsilon_greedy(state):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])


# -------------------------
# SARSA Training
# -------------------------
print("Training Started...\n")

for episode in range(episodes):

    state, _ = env.reset()
    action = epsilon_greedy(state)

    done = False
    total_reward = 0
    episode_len = 0

    while not done:

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        next_action = epsilon_greedy(next_state)

        # SARSA Update
        Q[state, action] += alpha * (
            reward
            + gamma * Q[next_state, next_action]
            - Q[state, action]
        )

        state = next_state
        action = next_action

        total_reward += reward
        episode_len += 1

    print(
        f"Episode {episode+1:3d} | Reward = {total_reward:4d} | Steps = {episode_len}"
    )

print("\nTraining Finished!")

# ---------------------------------------------------
# Testing with Human Rendering
# ---------------------------------------------------

print("\nTesting Learned Policy...\n")

env = gym.make("CliffWalking-v1", render_mode="human")

state, _ = env.reset()

done = False

total_reward = 0
episode_len = 0

while not done:

    action = np.argmax(Q[state])

    state, reward, terminated, truncated, _ = env.step(action)

    done = terminated or truncated

    total_reward += reward
    episode_len += 1

env.close()

print("\n==========================")
print("Testing Complete")
print("==========================")
print("Total Reward :", total_reward)
print("Episode Length :", episode_len)
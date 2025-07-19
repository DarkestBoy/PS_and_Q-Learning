import numpy as np
import random
import gymnasium as gym

def profit_sharing(
    env,
    num_episodes=10000,
    alpha=0.1,
    epsilon=0.5,
    max_steps=100,
    beta=0.9
):

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    PS = np.zeros((n_states, n_actions))

    reward_list = []

    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        trajectory = []

        for step in range(max_steps):
            # ε-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(PS[state])

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            trajectory.append((state, action))
            state = next_state
            total_reward += reward

            if done:
                break

        current_profit = total_reward
        for s, a in reversed(trajectory):
            PS[s, a] += alpha * (current_profit - PS[s, a])
            current_profit *= beta    # 奖励递减

        reward_list.append(total_reward)

    return PS, reward_list

# # test
# if __name__ == "__main__":
#     env = gym.make('FrozenLake-v1', is_slippery=False)
#     PS, rewards = profit_sharing(env)
#     print("final：")
#     print(PS)
#     print("last 10：", np.mean(rewards[-10:]))


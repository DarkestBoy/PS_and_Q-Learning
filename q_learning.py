import numpy as np
import random
import gymnasium as gym

def q_learning(
    env,
    num_episodes=10000,
    alpha=0.1,
    gamma=0.99,
    epsilon=0.5,
    max_steps=100
):

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    reward_list = []

    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        total_reward = 0

        for step in range(max_steps):
            # ε-greedy
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result

            if isinstance(next_state, tuple):
                next_state = next_state[0]

            # Q
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

            state = next_state
            total_reward += reward

            if done:
                break

        reward_list.append(total_reward)

    return Q, reward_list

# # test
# if __name__ == "__main__":
#     env = gym.make('FrozenLake-v1',is_slippery=False)
#     Q, rewards = q_learning(env)
#     print("final：")
#     print(Q)
#     print("last 10：", np.mean(rewards[-10:]))

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
    """
    Q-learning 算法实现（适用于FrozenLake-v1等离散型环境，兼容新版gym接口）

    参数:
        env          : gym环境对象，需支持reset()和step()接口
        num_episodes : 训练回合数
        alpha        : 学习率
        gamma        : 折扣因子
        epsilon      : ε-贪婪动作选择中的探索概率
        max_steps    : 每回合最大步数

    返回:
        Q            : 训练好的Q表 (np.ndarray)
        reward_list  : 每回合获得的reward序列
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))    # 初始化Q表

    reward_list = []

    for episode in range(num_episodes):
        state = env.reset()
        # 兼容新版gym reset返回(state, info)
        if isinstance(state, tuple):
            state = state[0]
        total_reward = 0

        for step in range(max_steps):
            # ε-贪婪策略选动作
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            step_result = env.step(action)
            if len(step_result) == 5:
                # 新版 Gym: (obs, reward, terminated, truncated, info)
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                # 旧版 Gym
                next_state, reward, done, info = step_result

            if isinstance(next_state, tuple):
                next_state = next_state[0]

            # Q表更新
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

            state = next_state
            total_reward += reward

            if done:
                break

        reward_list.append(total_reward)

    return Q, reward_list

# 测试代码（可选，提交时可删除或注释）
if __name__ == "__main__":
    env = gym.make('FrozenLake-v1',is_slippery=False)
    Q, rewards = q_learning(env)
    print("训练结束，最终Q表：")
    print(Q)
    print("最近10回合平均奖励：", np.mean(rewards[-10:]))

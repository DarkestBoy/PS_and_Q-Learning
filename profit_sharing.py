import numpy as np
import random
import gymnasium as gym

def profit_sharing(
    env,
    num_episodes=10000,
    alpha=0.1,
    epsilon=0.5,
    max_steps=100,
    beta=0.9   # 奖励回溯折扣
):
    """
    Profit Sharing 算法实现（适用于FrozenLake-v1等离散型环境，兼容新版gymnasium接口）

    参数:
        env          : gymnasium环境对象，需支持reset()和step()接口
        num_episodes : 训练回合数
        alpha        : 学习率
        epsilon      : ε-贪婪动作选择中的探索概率
        max_steps    : 每回合最大步数
        beta         : 奖励回溯递减系数（0~1）

    返回:
        PS           : 利益表（与Q表同结构）
        reward_list  : 每回合获得的reward序列
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    PS = np.zeros((n_states, n_actions))    # 初始化利益表

    reward_list = []

    for episode in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        trajectory = []

        for step in range(max_steps):
            # ε-贪婪动作选择
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

        # 回溯分配最终奖励
        current_profit = total_reward
        for s, a in reversed(trajectory):
            PS[s, a] += alpha * (current_profit - PS[s, a])
            current_profit *= beta    # 奖励递减

        reward_list.append(total_reward)

    return PS, reward_list

# 测试代码（可选，提交时可删除或注释）
if __name__ == "__main__":
    env = gym.make('FrozenLake-v1', is_slippery=False)
    PS, rewards = profit_sharing(env)
    print("训练结束，最终PS表：")
    print(PS)
    print("最近10回合平均奖励：", np.mean(rewards[-10:]))


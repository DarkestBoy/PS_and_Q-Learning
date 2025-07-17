import gymnasium as gym
import numpy as np
from q_learning import q_learning
from profit_sharing import profit_sharing
from util import plot_rewards_only, plot_final_success_bar, plot_policy_arrows

def main():
    # ----- Experiment Parameters -----
    num_episodes = 10000
    alpha = 0.1
    epsilon = 0.5
    max_steps = 100
    beta = 0.9
    window = 100
    env_name = 'FrozenLake-v1'
    is_slippery = False   # Deterministic environment for fair comparison

    # ----- Environment -----
    env = gym.make(env_name, map_name="4x4", is_slippery=is_slippery)

    # ----- Q-learning -----
    Q, q_rewards = q_learning(
        env,
        num_episodes=num_episodes,
        alpha=alpha,
        gamma=0.99,
        epsilon=epsilon,
        max_steps=max_steps
    )
    print("Q-learning: Average reward in last 10 episodes:", np.mean(q_rewards[-10:]))

    # ----- Profit Sharing -----
    PS, ps_rewards = profit_sharing(
        env,
        num_episodes=num_episodes,
        alpha=alpha,
        epsilon=epsilon,
        max_steps=max_steps,
        beta=beta
    )
    print("Profit Sharing: Average reward in last 10 episodes:", np.mean(ps_rewards[-10:]))

    # ----- Visualization -----
    plot_rewards_only(
        [q_rewards, ps_rewards],
        labels=["Q-learning", "Profit Sharing"],
        window=100,
        downsample_step=20
    )

    plot_final_success_bar(
        [q_rewards, ps_rewards],
        labels=["Q-learning", "Profit Sharing"],
        last_n=1000,
        title="Final Success Rate"
    )

    plot_policy_arrows(Q, title="Q-learning Policy (Optimal Actions)")
    plot_policy_arrows(PS, title="Profit Sharing Policy (Optimal Actions)")


if __name__ == "__main__":
    main()

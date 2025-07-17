import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def downsample(data, step):
    return np.array(data)[::step]

def plot_rewards_only(
    reward_lists,
    labels,
    window=100,
    downsample_step=1,
    title="Smoothed Reward vs Episode",
    save_path=None
):

    plt.figure(figsize=(14, 5))
    color_map = {"Q-learning": "blue", "Profit Sharing": "red"}
    for rewards, label in zip(reward_lists, labels):
        smoothed = moving_average(rewards, window)
        if downsample_step > 1:
            smoothed = downsample(smoothed, downsample_step)
            x = np.arange(len(smoothed)) * downsample_step
        else:
            x = np.arange(len(smoothed))
        color = color_map.get(label, None)
        plt.plot(x, smoothed, label=label, color=color, linewidth=2)
    plt.xlabel("Episode", fontsize=18)
    plt.ylabel("Reward", fontsize=18)
    plt.title(title, fontsize=20)
    plt.legend(fontsize=16)
    plt.grid()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()

def plot_final_success_bar(
    reward_lists,
    labels,
    last_n=1000,
    title="Final Success Rate (Last N Episodes)",
    save_path=None
):

    color_map = {"Q-learning": "blue", "Profit Sharing": "red"}
    success_rates = []
    for rewards in reward_lists:
        last_rewards = rewards[-last_n:] if last_n < len(rewards) else rewards
        successes = np.array(last_rewards) == 1
        rate = np.mean(successes) * 100  # Percentage
        success_rates.append(rate)
    bar_colors = [color_map.get(label, None) for label in labels]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, success_rates, color=bar_colors, width=0.6)
    plt.ylim(0, 100)
    plt.ylabel("Success Rate (%)", fontsize=18)
    plt.title(title + f" (Last {last_n} Episodes)", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=15)
    for bar, rate in zip(bars, success_rates):
        plt.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f"{rate:.1f}%", ha='center', va='bottom', fontsize=17
        )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()

def plot_policy_arrows(
    Q_or_PS,
    map_shape=(4, 4),
    title="Learned Policy (Optimal Action per State)",
    save_path=None
):

    # Action direction mapping: 0=Left, 1=Down, 2=Right, 3=Up (FrozenLake convention)
    arrow_dict = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    arrow_symbol = {0: '←', 1: '↓', 2: '→', 3: '↑'}

    n_states, n_actions = Q_or_PS.shape
    rows, cols = map_shape
    policy = np.argmax(Q_or_PS, axis=1).reshape(rows, cols)

    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    ax.set_xlim(-0.5, cols-0.5)
    ax.set_ylim(-0.5, rows-0.5)
    ax.set_xticks(np.arange(-0.5, cols, 1))
    ax.set_yticks(np.arange(-0.5, rows, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.grid(True, which='both', color='gray', linewidth=1, linestyle='--', alpha=0.4)
    plt.title(title, fontsize=18)

    # Draw arrows
    for i in range(rows):
        for j in range(cols):
            action = policy[i, j]
            dx, dy = arrow_dict[action]
            # The +0.4 is for visual centering of arrow length
            ax.arrow(
                j, i, dx*0.4, dy*0.4,
                head_width=0.2, head_length=0.2,
                fc="red", ec="red", linewidth=2, length_includes_head=True
            )
            # add action symbol for clarity
            ax.text(j, i, arrow_symbol[action], fontsize=22, ha='center', va='center', color='blue', alpha=0.7)

    # Optionally, mark start/goal positions (for standard FrozenLake 4x4)
    ax.text(0, 0, "S", fontsize=20, ha='center', va='center', color='green')
    ax.text(cols-1, rows-1, "G", fontsize=20, ha='center', va='center', color='orange')

    plt.gca().invert_yaxis()  # so that (0,0) is at the bottom-left
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()
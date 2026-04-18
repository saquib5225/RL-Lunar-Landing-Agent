import numpy as np
import matplotlib.pyplot as plt
import os
import json


def plot_training_rewards(rewards, window=100, title="Training Rewards", save_path=None):
    """Plot episode rewards with a rolling average line."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(rewards, alpha=0.3, color='steelblue', label='Episode Reward')

    if len(rewards) >= window:
        rolling = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), rolling, color='darkblue',
                linewidth=2, label=f'{window}-Episode Rolling Avg')

    ax.axhline(y=200, color='green', linestyle='--', alpha=0.7, label='Solved Threshold (200)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.close()


def plot_comparison(rewards_dict, window=100, title="Algorithm Comparison", save_path=None):
    """Overlay rolling average reward curves for multiple algorithms."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ['steelblue', 'coral', 'seagreen', 'orchid']

    for i, (name, rewards) in enumerate(rewards_dict.items()):
        color = colors[i % len(colors)]
        if len(rewards) >= window:
            rolling = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), rolling,
                    linewidth=2, color=color, label=name)

    ax.axhline(y=200, color='green', linestyle='--', alpha=0.7, label='Solved (200)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (Rolling Average)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    plt.close()


def plot_losses(losses, title="Training Loss", save_path=None):
    """Plot the training loss over steps."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses, alpha=0.5, color='tomato')

    # smoothed version
    if len(losses) > 50:
        window = min(50, len(losses) // 5)
        smooth = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(losses)), smooth, color='darkred', linewidth=2)

    ax.set_xlabel('Update Step')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_epsilon_decay(epsilons, save_path=None):
    """Plot epsilon values over episodes."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epsilons, color='purple')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.set_title('Exploration Rate (Epsilon) Decay')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_eval_histogram(eval_rewards, title="Evaluation Rewards Distribution", save_path=None):
    """Histogram of evaluation episode rewards."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(eval_rewards, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=np.mean(eval_rewards), color='red', linestyle='--',
               label=f'Mean: {np.mean(eval_rewards):.1f}')
    ax.axvline(x=200, color='green', linestyle='--', alpha=0.7, label='Solved (200)')
    ax.set_xlabel('Episode Reward')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_results(rewards, losses, epsilons, config, save_dir):
    """Save training logs to disk."""
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'rewards.npy'), np.array(rewards))
    if losses:
        np.save(os.path.join(save_dir, 'losses.npy'), np.array(losses))
    if epsilons:
        np.save(os.path.join(save_dir, 'epsilons.npy'), np.array(epsilons))
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Results saved to {save_dir}")


def print_training_summary(rewards, algorithm_name):
    """Print a summary of training performance."""
    print(f"\n{'='*50}")
    print(f"  {algorithm_name} Training Summary")
    print(f"{'='*50}")
    print(f"  Total episodes: {len(rewards)}")
    print(f"  Final 100-episode avg: {np.mean(rewards[-100:]):.2f}")
    print(f"  Best episode reward: {max(rewards):.2f}")
    print(f"  Worst episode reward: {min(rewards):.2f}")

    # check when/if solved
    if len(rewards) >= 100:
        for i in range(100, len(rewards)):
            avg = np.mean(rewards[i-100:i])
            if avg >= 200:
                print(f"  Solved at episode: {i}")
                break
        else:
            print(f"  Not solved (best 100-ep avg: {max(np.convolve(rewards, np.ones(100)/100, mode='valid')):.2f})")
    print(f"{'='*50}\n")

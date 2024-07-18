import numpy as np
import matplotlib.pyplot as plt
from data_process_atari.fixed_replay_buffer import FixedReplayBuffer
import argparse
from tqdm import tqdm
import os
from collections import defaultdict
import cv2
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy
import zlib

def analyze_game_data(game, data_dir_prefix, num_buffers=50, num_steps=5000000, trajectories_per_buffer=100):
    obss_sample = []
    actions = defaultdict(int)
    rewards = []
    done_idxs = []
    frame_differences = []
    total_rewards = []
    trajectory_lengths = []
    first_nonzero_rewards = []

    current_trajectory_reward = 0
    current_trajectory_length = 0
    current_trajectory_first_nonzero = -1
    max_return = float('-inf')  # Initialize max_return

    pbar = tqdm(total=num_steps)
    total_processed = 0

    while total_processed < num_steps:
        buffer_num = np.random.choice(np.arange(50 - num_buffers, 50), 1)[0]
        frb = FixedReplayBuffer(
            data_dir=data_dir_prefix + f'{game}/1/replay_logs',
            replay_suffix=buffer_num,
            observation_shape=(84, 84),
            stack_size=4,
            update_horizon=1,
            gamma=0.99,
            observation_dtype=np.uint8,
            batch_size=32,
            replay_capacity=100000,
        )
        
        if frb._loaded_buffers:
            done = False
            trajectories_to_load = trajectories_per_buffer
            i = 0
            prev_state = None
            while not done:
                states, ac, ret, next_states, next_action, next_reward, terminal, indices = frb.sample_transition_batch(batch_size=1, indices=[i])
                
                states = states.transpose((0, 3, 1, 2))[0]  # (1, 84, 84, 4) --> (4, 84, 84)
                
                if len(obss_sample) < 5000:  # Keep a sample of observations for visualization
                    obss_sample.append(states)
                
                actions[int(ac[0])] += 1
                rewards.append(ret[0])
                current_trajectory_reward += ret[0]
                current_trajectory_length += 1
                
                if current_trajectory_first_nonzero == -1 and ret[0] != 0:
                    current_trajectory_first_nonzero = current_trajectory_length

                if prev_state is not None:
                    frame_differences.append(np.mean(np.abs(states - prev_state)))
                prev_state = states

                total_processed += 1
                pbar.update(1)
                
                if terminal[0]:
                    done_idxs.append(total_processed)
                    total_rewards.append(current_trajectory_reward)
                    trajectory_lengths.append(current_trajectory_length)
                    if current_trajectory_first_nonzero != -1:
                        first_nonzero_rewards.append(current_trajectory_first_nonzero)
                    
                    max_return = max(max_return, current_trajectory_reward)  # Update max_return
                    
                    current_trajectory_reward = 0
                    current_trajectory_length = 0
                    current_trajectory_first_nonzero = -1

                    trajectories_to_load -= 1
                    if trajectories_to_load == 0:
                        done = True

                if total_processed >= num_steps:
                    pbar.close()
                    done = True
                    break
                
                i += 1
                if i >= 100000:
                    done = True

    pbar.close()
    return obss_sample, dict(actions), rewards, done_idxs, frame_differences, total_rewards, trajectory_lengths, first_nonzero_rewards, max_return

def visualize_state(state, game_name):
    # Assuming state shape is (4, 84, 84)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i in range(4):
        axes[i].imshow(state[i], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Frame {i+1}')
    plt.tight_layout()
    plt.savefig(f'dataset_analyze_expand/{game_name}/state_example.png')


def measure_frame_complexity(frames):
    """
    Measure the complexity of a sequence of game frames.
    
    :param frames: numpy array of shape (num_frames, height, width)
    :return: dict of complexity measures
    """
    results = {}
    
    # 1. Image Entropy
    entropies = []
    for frame in frames:
        hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        entropies.append(entropy(hist))
    results['entropy'] = {
        'mean': np.mean(entropies),
        'std': np.std(entropies)
    }
    
    # 2. Edge Detection
    edge_ratios = []
    for frame in frames:
        edges = cv2.Canny(frame, 100, 200)
        edge_ratio = np.sum(edges > 0) / (frame.shape[0] * frame.shape[1])
        edge_ratios.append(edge_ratio)
    results['edge_ratio'] = {
        'mean': np.mean(edge_ratios),
        'std': np.std(edge_ratios)
    }
    
    # 3. Compression Ratio
    compression_ratios = []
    for frame in frames:
        original_size = frame.nbytes
        compressed_size = len(zlib.compress(frame.tobytes()))
        compression_ratios.append(original_size / compressed_size)
    results['compression_ratio'] = {
        'mean': np.mean(compression_ratios),
        'std': np.std(compression_ratios)
    }
    
    # 4. Structural Similarity Index (SSIM)
    ssim_scores = []
    for i in range(len(frames) - 1):
        score = ssim(frames[i], frames[i+1])
        ssim_scores.append(score)
    results['ssim'] = {
        'mean': np.mean(ssim_scores),
        'std': np.std(ssim_scores)
    }
    
    # 5. Feature Detection (using SIFT)
    sift = cv2.SIFT_create()
    feature_counts = []
    for frame in frames:
        keypoints = sift.detect(frame, None)
        feature_counts.append(len(keypoints))
    results['feature_count'] = {
        'mean': np.mean(feature_counts),
        'std': np.std(feature_counts)
    }
    
    return results

# Function to analyze complexity across all frames
def analyze_frame_complexity(obss):
    # obss is a list of numpy arrays, obss[i].shape=(4, 84, 84), len(obss)=3000
    
    # analyse the average complexity of each frame across all observations
    complexities = []
    for obs in obss:
        state = [obs[i] for i in range(4)]
        complexity = measure_frame_complexity(state)
        complexities.append(complexity)
    
    # Print results (average complexity across all observations)
    for key in complexities[0].keys():
        mean = np.mean([c[key]['mean'] for c in complexities])
        std = np.mean([c[key]['std'] for c in complexities])
        print(f"{key}: {mean:.4f} +/- {std:.4f}")


def analyze_action_space(actions, game_name):
    total_actions = sum(actions.values())
    action_percentages = {action: (count / total_actions) * 100 for action, count in actions.items()}
    
    print(f"Total actions: {total_actions}")
    print(f"Unique actions: {sorted(actions.keys())}")
    print(f"Number of unique actions: {len(actions)}")
    
    # Use the original order of actions
    action_items = list(actions.items())
    # sort actions by index
    action_items.sort(key=lambda x: x[0])
    
    # Plot action distribution
    plt.figure(figsize=(12, 6))
    bars = plt.bar([str(action) for action, _ in action_items], 
                   [action_percentages[action] for action, _ in action_items])
    plt.title("Action Distribution")
    plt.xlabel("Action")
    plt.ylabel("Percentage")
    plt.xticks(rotation=45)

    # Add text annotations
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                 f'{height:.2f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'dataset_analyze_expand/{game_name}/action_distribution.png')

    # Print detailed breakdown
    print("\nAction frequency breakdown:")
    for action, count in action_items:
        percentage = action_percentages[action]
        print(f"Action {action}: {count} times ({percentage:.2f}%)")

    # Calculate and print entropy
    entropy = -sum((p/100) * np.log2(p/100) for p in action_percentages.values())
    max_entropy = np.log2(len(actions))
    normalized_entropy = entropy / max_entropy
    print(f"\nAction distribution entropy: {entropy:.4f}")
    print(f"Max possible entropy: {max_entropy:.4f}")
    print(f"Normalized entropy: {normalized_entropy:.4f}")


def analyze_reward_sequence(rewards, done_idxs, total_rewards, trajectory_lengths, first_nonzero_rewards, max_return, game_name):
    avg_trajectory_length = np.mean(trajectory_lengths)
    print(f"Average trajectory length: {avg_trajectory_length:.2f}")

    avg_total_reward = np.mean(total_rewards)
    print(f"Average total reward per trajectory: {avg_total_reward:.2f}")

    print(f"Maximum return: {max_return:.2f}")  # Add this line to print the max return

    if first_nonzero_rewards:
        avg_first_nonzero = np.mean(first_nonzero_rewards)
        print(f"Average steps until first non-zero reward: {avg_first_nonzero:.2f}")
    else:
        print("No non-zero rewards found in the analyzed trajectories.")

    # Visualize reward distribution
    plt.figure(figsize=(10, 5))
    n, bins, patches = plt.hist(rewards, bins=50)
    plt.title("Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")

    # Add text annotations
    for i in range(len(patches)):
        plt.text(patches[i].get_x() + patches[i].get_width() / 2, patches[i].get_height(), str(int(patches[i].get_height())), ha='center', va='bottom')

    plt.savefig(f'dataset_analyze_expand/{game_name}/reward_distribution.png')

    # Visualize cumulative reward distribution
    plt.figure(figsize=(10, 5))
    n, bins, patches = plt.hist(total_rewards, bins=50)
    plt.title("Cumulative Reward Distribution per Trajectory")
    plt.xlabel("Cumulative Reward")
    plt.ylabel("Frequency")

    # Add text annotations
    for i in range(len(patches)):
        plt.text(patches[i].get_x() + patches[i].get_width() / 2, patches[i].get_height(), str(int(patches[i].get_height())), ha='center', va='bottom')

    plt.savefig(f'dataset_analyze_expand/{game_name}/cumulative_reward_distribution.png')
    

def main():
    parser = argparse.ArgumentParser(description="Analyze Atari game data")
    parser.add_argument('--game', type=str, required=True, help='Name of the Atari game')
    parser.add_argument('--data_dir_prefix', type=str, default='./data/data_atari/', help='Path to dataset')
    parser.add_argument('--num_buffers', type=int, default=50, help='Number of buffers to sample from')
    parser.add_argument('--num_steps', type=int, default=5000000, help='Number of steps to analyze (10% of dataset)')
    parser.add_argument('--trajectories_per_buffer', type=int, default=100, help='Number of trajectories to sample per buffer')
    args = parser.parse_args()

    obss_sample, actions, rewards, done_idxs, frame_differences, total_rewards, trajectory_lengths, first_nonzero_rewards, max_return = analyze_game_data(
        args.game, 
        args.data_dir_prefix, 
        args.num_buffers, 
        args.num_steps, 
        args.trajectories_per_buffer
    )

    # Create directory for saving analysis results
    os.makedirs(f'dataset_analyze_expand/{args.game}', exist_ok=True)

    print(f"Analyzing data for game: {args.game}")
    print(f"Total steps analyzed: {len(rewards)}")
    print(f"Number of trajectories: {len(done_idxs)}")

    # Visualize a random state
    random_state_index = np.random.randint(len(obss_sample))
    print("Visualizing a random game state:")
    visualize_state(obss_sample[random_state_index], args.game)

    print("\nAction space analysis:")
    analyze_action_space(actions, args.game)

    print("\nReward sequence analysis:")
    analyze_reward_sequence(rewards, done_idxs, total_rewards, trajectory_lengths, first_nonzero_rewards, max_return, args.game)

    print("\nFrame complexity analysis:")
    analyze_frame_complexity(obss_sample)


if __name__ == "__main__":
    main()
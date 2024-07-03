import numpy as np
import matplotlib.pyplot as plt
from data_process_atari.fixed_replay_buffer import FixedReplayBuffer
import argparse
from tqdm import tqdm
import os
from collections import defaultdict


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
                
                if len(obss_sample) < 1000:  # Keep a sample of observations for visualization
                    obss_sample.append(states)
                
                # actions[ac[0]] += 1
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
    return obss_sample, dict(actions), rewards, done_idxs, frame_differences, total_rewards, trajectory_lengths, first_nonzero_rewards

def visualize_state(state, game_name):
    # Assuming state shape is (4, 84, 84)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i in range(4):
        axes[i].imshow(state[i], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Frame {i+1}')
    plt.tight_layout()
    plt.savefig(f'new_dataset_analyze/{game_name}/state_example.png')

def analyze_frame_differences(obss, game_name):
    # Randomly select 1000 consecutive pairs of states
    indices = np.random.randint(0, len(obss) - 1, 1000)
    differences = []
    for i in indices:
        diff = np.mean(np.abs(obss[i+1] - obss[i]))
        differences.append(diff)
    
    plt.figure(figsize=(10, 5))
    plt.hist(differences, bins=50)
    plt.title("Distribution of Frame Differences")
    plt.xlabel("Average Absolute Difference")
    plt.ylabel("Frequency")
    plt.savefig(f'new_dataset_analyze/{game_name}/frame_difference_distribution.png')

    print(f"Average frame difference: {np.mean(differences):.4f}")
    print(f"Median frame difference: {np.median(differences):.4f}")


# def analyze_action_space(actions, game_name):
#     unique_actions = set(actions)
#     print(f"Unique actions: {sorted(unique_actions)}")
#     print(f"Number of unique actions: {len(unique_actions)}")
    
#     # Count occurrences of each action
#     action_counts = {}
#     for action in actions:
#         if action in action_counts:
#             action_counts[action] += 1
#         else:
#             action_counts[action] = 1
    
#     # Calculate percentages
#     total_actions = len(actions)
#     action_percentages = {action: count / total_actions * 100 for action, count in action_counts.items()}
    
#     # Sort actions by frequency
#     sorted_actions = sorted(action_percentages.items(), key=lambda x: x[1], reverse=True)
    
#     # Plot action distribution
#     plt.figure(figsize=(10, 6))
#     bars = plt.bar([str(action) for action, _ in sorted_actions], [percentage for _, percentage in sorted_actions])
#     plt.title("Action Distribution")
#     plt.xlabel("Action")
#     plt.ylabel("Percentage")
#     plt.xticks(rotation=45)

#     # Add text annotations
#     for bar in bars:
#         plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.2f}%', ha='center', va='bottom')

#     plt.tight_layout()
#     plt.savefig(f'new_dataset_analyze/{game_name}/action_distribution.png')

#     # Print detailed breakdown
#     print("\nAction frequency breakdown:")
#     for action, percentage in sorted_actions:
#         print(f"Action {action}: {percentage:.2f}%")

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
    plt.savefig(f'new_dataset_analyze/{game_name}/action_distribution.png')

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


def analyze_reward_sequence(rewards, done_idxs, total_rewards, trajectory_lengths, first_nonzero_rewards, game_name):
    avg_trajectory_length = np.mean(trajectory_lengths)
    print(f"Average trajectory length: {avg_trajectory_length:.2f}")

    avg_total_reward = np.mean(total_rewards)
    print(f"Average total reward per trajectory: {avg_total_reward:.2f}")

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

    plt.savefig(f'new_dataset_analyze/{game_name}/reward_distribution.png')

    # Visualize cumulative reward distribution
    plt.figure(figsize=(10, 5))
    n, bins, patches = plt.hist(total_rewards, bins=50)
    plt.title("Cumulative Reward Distribution per Trajectory")
    plt.xlabel("Cumulative Reward")
    plt.ylabel("Frequency")

    # Add text annotations
    for i in range(len(patches)):
        plt.text(patches[i].get_x() + patches[i].get_width() / 2, patches[i].get_height(), str(int(patches[i].get_height())), ha='center', va='bottom')

    plt.savefig(f'new_dataset_analyze/{game_name}/cumulative_reward_distribution.png')
    
# def analyze_reward_sequence(rewards, done_idxs, game_name):
#     trajectory_rewards = []
#     start_idx = 0
#     for end_idx in done_idxs:
#         trajectory_rewards.append(rewards[start_idx:end_idx])
#         start_idx = end_idx

#     trajectory_lengths = [len(traj) for traj in trajectory_rewards]
#     avg_trajectory_length = np.mean(trajectory_lengths)
#     print(f"Average trajectory length: {avg_trajectory_length:.2f}")

#     total_rewards = [sum(traj) for traj in trajectory_rewards]
#     avg_total_reward = np.mean(total_rewards)
#     print(f"Average total reward per trajectory: {avg_total_reward:.2f}")

#     # Analyze reward delay
#     first_nonzero_reward = []
#     for traj in trajectory_rewards:
#         try:
#             first_nonzero = next(i for i, r in enumerate(traj) if r != 0)
#             first_nonzero_reward.append(first_nonzero)
#         except StopIteration:
#             pass
    
#     if first_nonzero_reward:
#         avg_first_nonzero = np.mean(first_nonzero_reward)
#         print(f"Average steps until first non-zero reward: {avg_first_nonzero:.2f}")
#     else:
#         print("No non-zero rewards found in the analyzed trajectories.")

#     # Visualize reward distribution
#     plt.figure(figsize=(10, 5))
#     n, bins, patches = plt.hist(rewards, bins=50)
#     plt.title("Reward Distribution")
#     plt.xlabel("Reward")
#     plt.ylabel("Frequency")

#     # Add text annotations
#     for i in range(len(patches)):
#         plt.text(patches[i].get_x() + patches[i].get_width() / 2, patches[i].get_height(), str(int(patches[i].get_height())), ha='center', va='bottom')

#     plt.savefig(f'new_dataset_analyze/{game_name}/reward_distribution.png')

#     # Visualize cumulative reward distribution
#     plt.figure(figsize=(10, 5))
#     n, bins, patches = plt.hist(total_rewards, bins=50)
#     plt.title("Cumulative Reward Distribution per Trajectory")
#     plt.xlabel("Cumulative Reward")
#     plt.ylabel("Frequency")

#     # Add text annotations
#     for i in range(len(patches)):
#         plt.text(patches[i].get_x() + patches[i].get_width() / 2, patches[i].get_height(), str(int(patches[i].get_height())), ha='center', va='bottom')

#     plt.savefig(f'new_dataset_analyze/{game_name}/cumulative_reward_distribution.png')

def main():
    parser = argparse.ArgumentParser(description="Analyze Atari game data")
    parser.add_argument('--game', type=str, required=True, help='Name of the Atari game')
    parser.add_argument('--data_dir_prefix', type=str, default='./data/data_atari/', help='Path to dataset')
    parser.add_argument('--num_buffers', type=int, default=50, help='Number of buffers to sample from')
    parser.add_argument('--num_steps', type=int, default=5000000, help='Number of steps to analyze (10% of dataset)')
    parser.add_argument('--trajectories_per_buffer', type=int, default=100, help='Number of trajectories to sample per buffer')
    args = parser.parse_args()

    obss_sample, actions, rewards, done_idxs, frame_differences, total_rewards, trajectory_lengths, first_nonzero_rewards = analyze_game_data(
        args.game, 
        args.data_dir_prefix, 
        args.num_buffers, 
        args.num_steps, 
        args.trajectories_per_buffer
    )

    # Create directory for saving analysis results
    os.makedirs(f'new_dataset_analyze/{args.game}', exist_ok=True)

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
    analyze_reward_sequence(rewards, done_idxs, total_rewards, trajectory_lengths, first_nonzero_rewards, args.game)

    # print("\nFrame difference analysis:")
    # analyze_frame_differences(frame_differences, args.game)

if __name__ == "__main__":
    main()
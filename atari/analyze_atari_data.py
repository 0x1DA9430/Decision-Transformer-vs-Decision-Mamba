import numpy as np
import matplotlib.pyplot as plt
from data_process_atari.fixed_replay_buffer import FixedReplayBuffer
import argparse
from tqdm import tqdm

def analyze_game_data(game, data_dir_prefix, num_buffers=50, num_steps=500000, trajectories_per_buffer=10):
    obss = []
    actions = []
    rewards = []
    done_idxs = []

    pbar = tqdm(total=num_steps)
    while len(obss) < num_steps:
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
            while not done:
                states, ac, ret, next_states, next_action, next_reward, terminal, indices = frb.sample_transition_batch(batch_size=1, indices=[i])
                
                states = states.transpose((0, 3, 1, 2))[0]  # (1, 84, 84, 4) --> (4, 84, 84)
                
                obss.append(states)
                actions.append(ac[0])
                rewards.append(ret[0])
                
                pbar.update(1)
                if len(obss) >= num_steps:
                    pbar.close()
                    done = True
                    break
                
                if terminal[0]:
                    done_idxs.append(len(obss))
                    if trajectories_to_load == 0:
                        done = True
                    else:
                        trajectories_to_load -= 1
                i += 1
                if i >= 100000:
                    done = True

    pbar.close()
    return obss, actions, rewards, done_idxs

# def visualize_state(state):
#     # Assuming state shape is (84, 84, 4)
#     fig, axes = plt.subplots(1, 4, figsize=(20, 5))
#     for i in range(4):
#         axes[i].imshow(state[:, :, i], cmap='gray')
#         axes[i].axis('off')
#         axes[i].set_title(f'Frame {i+1}')
#     plt.tight_layout()
#     # save the image, don't display it
#     plt.savefig('state.png')

def visualize_state(state):
    # Assuming state shape is (4, 84, 84)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i in range(4):
        axes[i].imshow(state[i], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Frame {i+1}')
    plt.tight_layout()
    plt.savefig('dataset_analyze/state_example.png')

def analyze_frame_differences(obss):
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
    plt.savefig('dataset_analyze/frame_diffs.png')

    print(f"Average frame difference: {np.mean(differences):.4f}")
    print(f"Median frame difference: {np.median(differences):.4f}")

# def analyze_action_space(actions):
#     unique_actions = set([action for trajectory in actions for action in trajectory])
#     print(f"Unique actions: {sorted(unique_actions)}")
#     print(f"Number of unique actions: {len(unique_actions)}")

def analyze_action_space(actions):
    unique_actions = set(actions)
    print(f"Unique actions: {sorted(unique_actions)}")
    print(f"Number of unique actions: {len(unique_actions)}")
    
    # Count occurrences of each action
    action_counts = {}
    for action in actions:
        if action in action_counts:
            action_counts[action] += 1
        else:
            action_counts[action] = 1
    
    # Calculate percentages
    total_actions = len(actions)
    action_percentages = {action: count / total_actions * 100 for action, count in action_counts.items()}
    
    # Sort actions by frequency
    sorted_actions = sorted(action_percentages.items(), key=lambda x: x[1], reverse=True)
    
    # Plot action distribution
    plt.figure(figsize=(10, 6))
    plt.bar([str(action) for action, _ in sorted_actions], [percentage for _, percentage in sorted_actions])
    plt.title("Action Distribution")
    plt.xlabel("Action")
    plt.ylabel("Percentage")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('dataset_analyze/action_distribution.png')

    # Print detailed breakdown
    print("\nAction frequency breakdown:")
    for action, percentage in sorted_actions:
        print(f"Action {action}: {percentage:.2f}%")

# def analyze_reward_sequence(rewards):
#     trajectory_lengths = [len(traj) for traj in rewards]
#     avg_trajectory_length = np.mean(trajectory_lengths)
#     print(f"Average trajectory length: {avg_trajectory_length:.2f}")

#     total_rewards = [sum(traj) for traj in rewards]
#     avg_total_reward = np.mean(total_rewards)
#     print(f"Average total reward per trajectory: {avg_total_reward:.2f}")

#     # Analyze reward delay
#     first_nonzero_reward = []
#     for traj in rewards:
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
#     plt.hist([r for traj in rewards for r in traj], bins=50)
#     plt.title("Reward Distribution")
#     plt.xlabel("Reward")
#     plt.ylabel("Frequency")
#     plt.show()

def analyze_reward_sequence(rewards, done_idxs):
    trajectory_rewards = []
    start_idx = 0
    for end_idx in done_idxs:
        trajectory_rewards.append(rewards[start_idx:end_idx])
        start_idx = end_idx

    trajectory_lengths = [len(traj) for traj in trajectory_rewards]
    avg_trajectory_length = np.mean(trajectory_lengths)
    print(f"Average trajectory length: {avg_trajectory_length:.2f}")

    total_rewards = [sum(traj) for traj in trajectory_rewards]
    avg_total_reward = np.mean(total_rewards)
    print(f"Average total reward per trajectory: {avg_total_reward:.2f}")

    # Analyze reward delay
    first_nonzero_reward = []
    for traj in trajectory_rewards:
        try:
            first_nonzero = next(i for i, r in enumerate(traj) if r != 0)
            first_nonzero_reward.append(first_nonzero)
        except StopIteration:
            pass
    
    if first_nonzero_reward:
        avg_first_nonzero = np.mean(first_nonzero_reward)
        print(f"Average steps until first non-zero reward: {avg_first_nonzero:.2f}")
    else:
        print("No non-zero rewards found in the analyzed trajectories.")

    # Visualize reward distribution
    plt.figure(figsize=(10, 5))
    plt.hist(rewards, bins=50)
    plt.title("Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.savefig('dataset_analyze/reward_distribution.png')

    # Visualize cumulative reward distribution
    plt.figure(figsize=(10, 5))
    plt.hist(total_rewards, bins=50)
    plt.title("Cumulative Reward Distribution per Trajectory")
    plt.xlabel("Cumulative Reward")
    plt.ylabel("Frequency")
    plt.savefig('dataset_analyze/cumulative_reward_distribution.png')

def main():
    parser = argparse.ArgumentParser(description="Analyze Atari game data")
    parser.add_argument('--game', type=str, required=True, help='Name of the Atari game')
    parser.add_argument('--data_dir_prefix', type=str, default='./data/data_atari/', help='Path to dataset')
    parser.add_argument('--num_buffers', type=int, default=50, help='Number of buffers to sample from')
    parser.add_argument('--num_steps', type=int, default=500000, help='Number of steps to analyze')
    parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample per buffer')
    args = parser.parse_args()

    obss, actions, rewards, done_idxs = analyze_game_data(
        args.game, 
        args.data_dir_prefix, 
        args.num_buffers, 
        args.num_steps, 
        args.trajectories_per_buffer
    )

    print(f"Analyzing data for game: {args.game}")
    print(f"Total steps analyzed: {len(obss)}")
    print(f"Number of trajectories: {len(done_idxs)}")

    
    # # Visualize a random state
    # random_trajectory = np.random.choice(len(obss))
    # random_state = np.random.choice(len(obss[random_trajectory]))
    # print("Visualizing a random game state:")
    # visualize_state(obss[random_trajectory][random_state])

    # Visualize a random state
    random_state_index = np.random.randint(len(obss))
    print("Visualizing a random game state:")
    visualize_state(obss[random_state_index])

    print("\nAction space analysis:")
    analyze_action_space(actions)

    print("\nReward sequence analysis:")
    analyze_reward_sequence(rewards, done_idxs)

if __name__ == "__main__":
    main()
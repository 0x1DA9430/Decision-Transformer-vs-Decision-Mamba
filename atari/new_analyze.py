import numpy as np
import matplotlib.pyplot as plt
from data_process_atari.fixed_replay_buffer import FixedReplayBuffer
import argparse
from tqdm import tqdm
import os
from collections import defaultdict

def analyze_game_data_generator(game, data_dir_prefix, num_buffers=50, num_steps=5000000, trajectories_per_buffer=100):
    steps_analyzed = 0
    buffer_nums = np.random.choice(np.arange(50 - num_buffers, 50), num_buffers, replace=False)
    
    for buffer_num in buffer_nums:
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
            trajectories_loaded = 0
            i = 0
            while trajectories_loaded < trajectories_per_buffer and steps_analyzed < num_steps:
                states, ac, ret, next_states, next_action, next_reward, terminal, indices = frb.sample_transition_batch(batch_size=1, indices=[i])
                
                states = states.transpose((0, 3, 1, 2))[0]  # (1, 84, 84, 4) --> (4, 84, 84)
                
                yield states, ac[0], ret[0], terminal[0]
                
                steps_analyzed += 1
                
                if terminal[0]:
                    trajectories_loaded += 1
                
                i += 1
                if i >= 100000:
                    break
    
    print(f"Total steps analyzed: {steps_analyzed}")

def incremental_analyze(game, data_dir_prefix, num_buffers=50, num_steps=5000000, trajectories_per_buffer=100):
    action_counts = defaultdict(int)
    reward_sum = 0
    reward_counts = defaultdict(int)
    total_rewards = []
    current_trajectory_reward = 0
    frame_differences = []
    total_steps = 0
    trajectory_lengths = []
    current_trajectory_length = 0
    first_nonzero_rewards = []
    steps_to_first_nonzero = 0

    data_generator = analyze_game_data_generator(game, data_dir_prefix, num_buffers, num_steps, trajectories_per_buffer)
    
    prev_state = None
    for state, action, reward, done in tqdm(data_generator, total=num_steps):
        # Action analysis
        action_counts[action] += 1
        
        # Reward analysis
        reward_sum += reward
        reward_counts[reward] += 1
        current_trajectory_reward += reward
        current_trajectory_length += 1
        
        if reward != 0 and steps_to_first_nonzero == 0:
            first_nonzero_rewards.append(steps_to_first_nonzero)
            steps_to_first_nonzero = 0
        elif reward == 0:
            steps_to_first_nonzero += 1
        
        # Frame difference analysis
        if prev_state is not None:
            diff = np.mean(np.abs(state - prev_state))
            frame_differences.append(diff)
        prev_state = state
        
        if done:
            total_rewards.append(current_trajectory_reward)
            trajectory_lengths.append(current_trajectory_length)
            current_trajectory_reward = 0
            current_trajectory_length = 0
            steps_to_first_nonzero = 0
        
        total_steps += 1
    
    return action_counts, reward_sum, reward_counts, total_rewards, frame_differences, total_steps, trajectory_lengths, first_nonzero_rewards

def visualize_results(game, results):
    action_counts, reward_sum, reward_counts, total_rewards, frame_differences, total_steps, trajectory_lengths, first_nonzero_rewards = results
    
    os.makedirs(f'dataset_analyze/{game}', exist_ok=True)
    
    # Action distribution
    plt.figure(figsize=(10, 6))
    actions, counts = zip(*sorted(action_counts.items()))
    plt.bar(actions, counts)
    plt.title("Action Distribution")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.savefig(f'dataset_analyze/{game}/action_distribution.png')
    plt.close()
    
    # Reward distribution
    plt.figure(figsize=(10, 6))
    rewards, counts = zip(*sorted(reward_counts.items()))
    plt.bar(rewards, counts)
    plt.title("Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Count")
    plt.savefig(f'dataset_analyze/{game}/reward_distribution.png')
    plt.close()
    
    # Cumulative reward distribution
    plt.figure(figsize=(10, 6))
    plt.hist(total_rewards, bins=50)
    plt.title("Cumulative Reward Distribution per Trajectory")
    plt.xlabel("Cumulative Reward")
    plt.ylabel("Frequency")
    plt.savefig(f'dataset_analyze/{game}/cumulative_reward_distribution.png')
    plt.close()
    
    # Frame difference distribution
    plt.figure(figsize=(10, 6))
    plt.hist(frame_differences, bins=50)
    plt.title("Frame Difference Distribution")
    plt.xlabel("Average Absolute Difference")
    plt.ylabel("Frequency")
    plt.savefig(f'dataset_analyze/{game}/frame_difference_distribution.png')
    plt.close()
    
    # Print statistics
    print(f"Total steps analyzed: {total_steps}")
    print(f"Number of trajectories: {len(total_rewards)}")
    print(f"Average trajectory length: {np.mean(trajectory_lengths):.2f}")
    print(f"Average total reward per trajectory: {np.mean(total_rewards):.2f}")
    print(f"Average frame difference: {np.mean(frame_differences):.4f}")
    print(f"Median frame difference: {np.median(frame_differences):.4f}")
    print(f"Average steps until first non-zero reward: {np.mean(first_nonzero_rewards):.2f}")

def main():
    parser = argparse.ArgumentParser(description="Analyze Atari game data")
    parser.add_argument('--game', type=str, required=True, help='Name of the Atari game')
    parser.add_argument('--data_dir_prefix', type=str, default='./data/data_atari/', help='Path to dataset')
    parser.add_argument('--num_buffers', type=int, default=50, help='Number of buffers to sample from')
    parser.add_argument('--num_steps', type=int, default=5000000, help='Number of steps to analyze')
    parser.add_argument('--trajectories_per_buffer', type=int, default=100, help='Number of trajectories to sample per buffer')
    args = parser.parse_args()

    results = incremental_analyze(
        args.game, 
        args.data_dir_prefix, 
        args.num_buffers, 
        args.num_steps, 
        args.trajectories_per_buffer
    )

    visualize_results(args.game, results)

if __name__ == "__main__":
    main()
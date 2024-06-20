import numpy as np
from tqdm import tqdm
from fixed_replay_buffer import FixedReplayBuffer

def find_max_return(num_buffers, game, data_dir_prefix):
    max_return = 0

    print('Finding maximum return from buffers')
    pbar = tqdm(total=num_buffers, desc="Buffers processed")

    for buffer_num in range(num_buffers):
        print(f'Loading buffer {buffer_num}')
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
            i = 0
            done = False
            current_return = 0
            
            while not done:
                _, _, ret, _, _, _, terminal, _ = frb.sample_transition_batch(batch_size=1, indices=[i])
                
                current_return += ret[0]
                
                if terminal[0]:
                    if current_return > max_return:
                        max_return = current_return
                    current_return = 0
                
                i += 1
                if i >= 100000:
                    done = True
        
        pbar.update(1)  # Update progress bar after processing each buffer
        # log the maximum return found so far
        pbar.set_postfix({'current max return': max_return})

    pbar.close()

    return max_return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_buffers', default=50, type=int, help='Number of buffers to load')
    parser.add_argument('--game', type=str, required=True, help='Name of the game')
    parser.add_argument('--data_dir_prefix', default='../data/data_atari/', type=str, help='Prefix for the data directory')

    args = parser.parse_args()
    max_return = find_max_return(args.num_buffers, args.game, args.data_dir_prefix)
    print(f'Maximum return: {max_return}')

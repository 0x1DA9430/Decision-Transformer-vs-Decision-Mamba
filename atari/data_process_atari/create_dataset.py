import numpy as np
import torch
from torch.utils.data import Dataset
from data_process_atari.fixed_replay_buffer import FixedReplayBuffer
from tqdm import tqdm


# def create_action_fusion_mapping(game):
#     if game == 'KungFuMaster':
#         return {
#             0: 0,  # NOOP
#             1: 1, 2: 1, 3: 1, 4: 1,  # Directional
#             5: 2, 6: 2,  # Diagonal
#             7: 3, 8: 3, 9: 3,  # Fire + Direction
#             10: 4, 11: 4, 12: 4, 13: 4  # Diagonal + Fire
#         }
#     elif game == 'Hero':
#         return {
#             0: 0,  # NOOP
#             1: 1,  # FIRE
#             2: 2, 3: 2, 4: 2, 5: 2,  # Directional
#             6: 3, 7: 3, 8: 3, 9: 3,  # Diagonal
#             10: 4, 11: 4, 12: 4, 13: 4,  # Fire + Direction
#             14: 5, 15: 5, 16: 5, 17: 5  # Diagonal + Fire
#         }
#     else:
#         return None  # No fusion for other games

def create_action_fusion_mapping(game):
    if game == 'Hero':
        return {
            0: 0,  # NOOP
            1: 1,  # FIRE
            2: 2,  # UP -> UP, UPFIRE
            3: 3,  # RIGHT -> RIGHT, RIGHTFIRE
            4: 4,  # LEFT -> LEFT, LEFTFIRE
            5: 5,  # DOWN -> DOWN, DOWNFIRE
            6: 6,  # UPRIGHT -> UPRIGHT, UPRIGHTFIRE
            7: 7,  # UPLEFT -> UPLEFT, UPLEFTFIRE
            8: 8,  # DOWNRIGHT -> DOWNRIGHT, DOWNRIGHTFIRE
            9: 9,  # DOWNLEFT -> DOWNLEFT, DOWNLEFTFIRE
            10: 2, # UPFIRE -> UP, UPFIRE
            11: 3, # RIGHTFIRE -> RIGHT, RIGHTFIRE
            12: 4, # LEFTFIRE -> LEFT, LEFTFIRE
            13: 5, # DOWNFIRE -> DOWN, DOWNFIRE
            14: 6, # UPRIGHTFIRE -> UPRIGHT, UPRIGHTFIRE
            15: 7, # UPLEFTFIRE -> UPLEFT, UPLEFTFIRE
            16: 8, # DOWNRIGHTFIRE -> DOWNRIGHT, DOWNRIGHTFIRE
            17: 9, # DOWNLEFTFIRE -> DOWNLEFT, DOWNLEFTFIRE
        }
    elif game == 'KungFuMaster':
        return {
            0: 0,  # NOOP
            1: 1,  # UP
            2: 2,  # RIGHT -> RIGHT, RIGHTFIRE
            3: 3,  # LEFT -> LEFT, LEFTFIRE
            4: 4,  # DOWN -> DOWN, DOWNFIRE
            5: 5,  # DOWNRIGHT -> DOWNRIGHT, DOWNRIGHTFIRE
            6: 6,  # DOWNLEFT -> DOWNLEFT, DOWNLEFTFIRE
            7: 2,  # RIGHTFIRE -> RIGHT, RIGHTFIRE
            8: 3,  # LEFTFIRE -> LEFT, LEFTFIRE
            9: 4,  # DOWNFIRE -> DOWN, DOWNFIRE
            10: 7, # UPRIGHTFIRE
            11: 8, # UPLEFTFIRE
            12: 5, # DOWNRIGHTFIRE -> DOWNRIGHT, DOWNRIGHTFIRE
            13: 6, # DOWNLEFTFIRE -> DOWNLEFT, DOWNLEFTFIRE
        }
    else:
        return None # No fusion for other games

# def create_action_fusion_mapping(game):
#     if game == 'Hero':
#         return {
#             0: 0,  # NOOP
#             1: 1,  # FIRE
#             2: 2,  # UP
#             3: 3,  # RIGHT
#             4: 4,  # LEFT
#             5: 5,  # DOWN
#             6: 6,  # UPRIGHT
#             7: 7,  # UPLEFT
#             8: 8,  # DOWNRIGHT
#             9: 9,  # DOWNLEFT
#             10: 2, # UPFIRE -> UP
#             11: 3, # RIGHTFIRE -> RIGHT
#             12: 4, # LEFTFIRE -> LEFT
#             13: 5, # DOWNFIRE -> DOWN
#             14: 6, # UPRIGHTFIRE -> UPRIGHT
#             15: 7, # UPLEFTFIRE -> UPLEFT
#             16: 8, # DOWNRIGHTFIRE -> DOWNRIGHT
#             17: 9, # DOWNLEFTFIRE -> DOWNLEFT
#         }
#     elif game == 'KungFuMaster':
#         return {
#             0: 0,  # NOOP
#             1: 1,  # UP
#             2: 2,  # RIGHT
#             3: 3,  # LEFT
#             4: 4,  # DOWN
#             5: 5,  # DOWNRIGHT
#             6: 6,  # DOWNLEFT
#             7: 2,  # RIGHTFIRE -> RIGHT
#             8: 3,  # LEFTFIRE -> LEFT
#             9: 4,  # DOWNFIRE -> DOWN
#             10: 7, # UPRIGHTFIRE
#             11: 8, # UPLEFTFIRE
#             12: 5, # DOWNRIGHTFIRE -> DOWNRIGHT
#             13: 6, # DOWNLEFTFIRE -> DOWNLEFT
#         }
#     else:
#         return None  # No fusion for other games

def get_action_probs(game):
    if game == 'Hero':
        return {
            0: 0.05062, 1: 0.04602, 2: 0.04000, 3: 0.06265, 4: 0.05882, 5: 0.04852, 6: 0.04840,
            7: 0.04644, 8: 0.06300, 9: 0.05618, 10: 0.04030, 11: 0.07317, 12: 0.06553, 13: 0.05024,
            14: 0.05181, 15: 0.05065, 16: 0.07800, 17: 0.06965
        }
    elif game == 'KungFuMaster':
        return {
            0: 0.06515, 1: 0.05810, 2: 0.05423, 3: 0.09354, 4: 0.06801, 5: 0.07249, 6: 0.07120,
            7: 0.07013, 8: 0.09301, 9: 0.06806, 10: 0.06535, 11: 0.07500, 12: 0.06880, 13: 0.07693
        }
    else:
        return None
    
def create_dataset(num_buffers, num_steps, game, data_dir_prefix, trajectories_per_buffer, use_action_fusion):
    # -- load data from memory (make more efficient)
    obss = []
    actions = []
    returns = [0]
    done_idxs = []
    stepwise_returns = []

    transitions_per_buffer = np.zeros(50, dtype=int)
    num_trajectories = 0

    # Create action fusion mapping if needed
    action_fusion_map = create_action_fusion_mapping(game) if use_action_fusion else None
    action_probs = get_action_probs(game) if use_action_fusion else None

    print('loading trajectories from buffers')
    pbar = tqdm(total=num_steps, mininterval=60)  # Initialize the progress bar
    while len(obss) < num_steps:
        buffer_num = np.random.choice(np.arange(50 - num_buffers, 50), 1)[0]
        i = transitions_per_buffer[buffer_num]
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
            curr_num_transitions = len(obss)
            trajectories_to_load = trajectories_per_buffer
            while not done:
                states, ac, ret, next_states, next_action, next_reward, terminal, indices = frb.sample_transition_batch(batch_size=1, indices=[i])

                states = states.transpose((0, 3, 1, 2))[0] # (1, 84, 84, 4) --> (4, 84, 84)

                obss += [states]

                # Apply action fusion if enabled
                if use_action_fusion:
                    ac = action_fusion_map[ac[0]]
                    actions += [ac]
                else:
                    actions += [ac[0]]

                # # print actions during loading
                # print("using action fusion" if use_action_fusion else "not using action fusion")
                # print('-'*50)
                # print(actions)
                # print('-'*50)

                stepwise_returns += [ret[0]]
                
                # Update progress bar
                pbar.update(1)
                if len(obss) >= num_steps:
                    pbar.close()  # Close progress bar if step limit is reached
                    break
                
                if terminal[0]:
                    done_idxs += [len(obss)]
                    returns += [0]
                    if trajectories_to_load == 0:
                        done = True
                    else:
                        trajectories_to_load -= 1
                returns[-1] += ret[0]
                i += 1
                if i >= 100000:
                    obss = obss[:curr_num_transitions]
                    actions = actions[:curr_num_transitions]
                    stepwise_returns = stepwise_returns[:curr_num_transitions]
                    returns[-1] = 0
                    i = transitions_per_buffer[buffer_num]
                    done = True
            num_trajectories += (trajectories_per_buffer - trajectories_to_load)
            transitions_per_buffer[buffer_num] = i
    print('this buffer has %d loaded transitions and there are now %d transitions total divided into %d trajectories' % (i, len(obss), num_trajectories))

    actions = np.array(actions)
    returns = np.array(returns)
    stepwise_returns = np.array(stepwise_returns)
    done_idxs = np.array(done_idxs)

    # -- create reward-to-go dataset
    start_index = 0
    rtg = np.zeros_like(stepwise_returns)
    for i in done_idxs:
        i = int(i)
        curr_traj_returns = stepwise_returns[start_index:i]
        for j in range(i-1, start_index-1, -1): # start from i-1
            rtg_j = curr_traj_returns[j-start_index:i-start_index]
            rtg[j] = sum(rtg_j)
        start_index = i
    print('max rtg is %d' % max(rtg))

    # -- create timestep dataset
    start_index = 0
    timesteps = np.zeros(len(actions)+1, dtype=int)
    for i in done_idxs:
        i = int(i)
        timesteps[start_index:i+1] = np.arange(i+1 - start_index)
        start_index = i+1
    print('max timestep is %d' % max(timesteps))

    return obss, actions, returns, done_idxs, rtg, timesteps, action_fusion_map, action_probs


class StateActionReturnDataset(Dataset):
    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
        states = states / 255.  # normalize pixel values to 0-1
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rtgs, timesteps

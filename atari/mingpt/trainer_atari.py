"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import os
import time
import logging
from collections import deque

import math
import random
import cv2
#from PIL import Image
from tqdm import tqdm
import atari_py

import numpy as np
import torch
#import torch.optim as optim
#from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

from mingpt.utils import sample, update_summary
from data_process_atari.create_dataset import create_action_fusion_mapping, get_action_probs

import csv


try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

try:
    import mlflow
    has_mlflow = True
except ImportError:
    has_mlflow = False


logger = logging.getLogger(__name__)


class TrainerConfig:
    # optimization parameters
    max_epochs = 5
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights

    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)

    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    def __init__(
            self,
            model,
            train_dataset,
            test_dataset,
            config,
            ):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.losses_per_epoch = []  # List to store loss for each epoch
        self.lrs_per_epoch = []  # List to store learning rate for each epoch
        self.metrics_file = "training_metrics.csv"  # File to save the metrics
        self.use_action_fusion = config.use_action_fusion # Use action fusion or not
        self.game = config.game 

        # Action fusion mapping and probabilities
        if self.use_action_fusion:
            self.action_fusion_map = create_action_fusion_mapping(self.game)
            self.action_probs = get_action_probs(self.game)

        # take over whatever gpus are on the system
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        self.start_time = time.time()

    def reverse_map_action(self, reduced_action):
        original_actions = [k for k, v in self.action_fusion_map.items() if v == reduced_action]
        probs = [self.action_probs[a] for a in original_actions]
        total_prob = sum(probs)
        normalized_probs = [p / total_prob for p in probs]
        return np.random.choice(original_actions, p=normalized_probs)
    
    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        # torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def save_metrics_to_file(self, epoch, epoch_losses, epoch_lrs):
        # Open file in append mode and write metrics
        path = os.path.join(self.config.output_dir, self.metrics_file)
        with open(path, 'a', newline='') as file:
            writer = csv.writer(file)
            if epoch == 0:
                writer.writerow(['Epoch', 'Iteration', 'Loss', 'Learning Rate'])
            for it, (loss, lr) in enumerate(zip(epoch_losses, epoch_lrs)):
                writer.writerow([epoch + 1, it + 1, loss, lr])
                
    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
    
        #** train one epoch **
        def run_epoch(split, epoch_num=0, epoch_losses=None, epoch_lrs=None):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data,
                                shuffle=True,
                                pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers,
                                )

            # losses = []
            pbar = tqdm(enumerate(loader), total=len(loader), mininterval=60) if is_train else enumerate(loader)
            for it, (x, y, r, t) in pbar:
                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    # logits, loss = model(x, y, r)
                    logits, loss = model(x, y, y, r, t)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    # losses.append(loss.item())
                    epoch_losses.append(loss.item())

                if is_train:
                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate
                    epoch_lrs.append(lr)
                    
                    # report progress
                    # pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if is_train:
                train_loss = float(np.mean(epoch_losses))
                logger.info("train loss: %f", train_loss)
                return train_loss
            if not is_train:
                test_loss = float(np.mean(epoch_losses))
                logger.info("test loss: %f", test_loss)
                return test_loss
        #*****

        # best_loss = float('inf')
        best_return = -float('inf')

        self.tokens = 0  # counter used for learning rate decay
        for epoch in range(config.max_epochs):

            epoch_losses = []
            epoch_lrs = []

            print(f"epoch {epoch}")
            loss = run_epoch('train', epoch_num=epoch, epoch_losses=epoch_losses, epoch_lrs=epoch_lrs)
            
            self.losses_per_epoch.append(epoch_losses)
            self.lrs_per_epoch.append(epoch_lrs)

            # Save metrics to file
            self.save_metrics_to_file(epoch, epoch_losses, epoch_lrs)
            
            # print(f"epoch {epoch}")
            # loss = run_epoch('train', epoch_num=epoch)

            # if self.test_dataset is not None:
            #     test_loss = run_epoch('test')

            ## supports early stopping based on the test loss, or just save always if no test set is provided
            # good_model = self.test_dataset is None or test_loss < best_loss
            # if self.config.ckpt_path is not None and good_model:
            #     best_loss = test_loss
            #     self.save_checkpoint()

            # -- pass in target returns
            if self.config.model_type == 'naive':
                eval_return, eval_time = self.get_returns(0)
            elif self.config.model_type == 'reward_conditioned':
                if self.config.game == 'Breakout':
                    eval_return, eval_time = self.get_returns(520)  # 5*max return in training data
                elif self.config.game == 'Qbert':
                    eval_return, eval_time = self.get_returns(3200) # 5*max return in training data
                elif self.config.game == 'Pong':
                    eval_return, eval_time = self.get_returns(105)  # 5*max return in training data   
                elif self.config.game == 'Seaquest':
                    eval_return, eval_time = self.get_returns(1570) # 5*max return in training data
                elif self.config.game == 'Hero':
                    eval_return, eval_time = self.get_returns(950)  # 5*max return in training data
                elif self.config.game == 'KungFuMaster':
                    eval_return, eval_time = self.get_returns(1420) # 5*max return in training data
                elif self.config.game == 'Alien':
                    eval_return, eval_time = self.get_returns(1075) # 5*max return in training data
                elif self.config.game == 'RoadRunner':
                    eval_return, eval_time = self.get_returns(1075) # 5*max return in training data
                elif self.config.game == 'BattleZone':
                    eval_return, eval_time = self.get_returns(160)  # 5*max return in training data
                elif self.config.game == 'BankHeist':
                    eval_return, eval_time = self.get_returns(680)  # 5*max return in training data
                elif self.config.game == 'FishingDerby':
                    eval_return, eval_time = self.get_returns(305)  # 5*max return in training data
                elif self.config.game == 'Zaxxon':
                    eval_return, eval_time = self.get_returns(170)  # 5*max return in training data
                elif self.config.game == 'Jamesbond':
                    eval_return, eval_time = self.get_returns(110)  # 5*max return in training data
                elif self.config.game == 'MontezumaRevenge':
                    eval_return, eval_time = self.get_returns(0)    # the max return is 0 in training data
                elif self.config.game == 'MsPacman':
                    eval_return, eval_time = self.get_returns(2350) # 5*max return in training data
                elif self.config.game == 'SpaceInvaders':
                    eval_return, eval_time = self.get_returns(1440) # 5*max return in training data
                else:
                    raise NotImplementedError()

                logs = dict()
                logs['training/train_loss_mean'] = loss
                logs['evaluation/eval_return'] = eval_return
                logs['evaluation/eval_time'] = eval_time  # log the evaluation time
                logs['time/total'] = time.time() - self.start_time
                if self.config.output_dir is not None:
                    update_summary(
                        epoch,
                        logs,
                        filename=os.path.join(config.output_dir, 'summary.csv'),
                        args_dir=self.config.args_dir,
                        write_header=epoch == 0,
                        log_wandb=self.config.log_to_wandb and has_wandb,
                        log_mlflow=self.config.log_to_mlflow and has_mlflow,
                        )

            else:
                raise NotImplementedError()

    def get_returns(self, ret):
        eval_start_time = time.time()  # Start time for evaluation

        self.model.train(False)
        args = Args(self.config.game.lower(), self.config.seed)
        args.use_action_fusion = self.use_action_fusion
        env = Env(args)
        env.eval()

        T_rewards, T_Qs = [], []
        done = True
        for _ in range(10):
            state = env.reset()
            state = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            rtgs = [ret]
            raw_model = self.model.module if hasattr(self.model, "module") else self.model

            # first state is from env, first rtg is target return, and first timestep is 0
            sampled_action = sample(
                raw_model,
                state,
                1,
                temperature=1.0,
                sample=True,
                actions=None,
                rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1),
                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device),
                )

            j = 0
            all_states = state
            actions = []

            while True:
                if done:
                    state, reward_sum, done = env.reset(), 0, False
                
                action = sampled_action.cpu().numpy()[0,-1]

                # print("using action fusion: %s" % self.use_action_fusion)
                # print("action before: ", action)

                """ Reverse mapping """
                # # Reverse map the action if using action fusion
                # if self.use_action_fusion:
                #     action = self.reverse_map_action(action)
            
                # print("action after: ", action)

                actions += [sampled_action]
                state, reward, done = env.step(action)
                reward_sum += reward
                j += 1

                if done:
                    T_rewards.append(reward_sum)
                    break

                state = state.unsqueeze(0).unsqueeze(0).to(self.device)
                all_states = torch.cat([all_states, state], dim=0)
                rtgs += [rtgs[-1] - reward]

                # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                # timestep is just current timestep
                sampled_action = sample(
                    raw_model,
                    all_states.unsqueeze(0),
                    1,
                    temperature=1.0,
                    sample=True,
                    actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0),
                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1),
                    timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)),
                    )

        env.close()
        eval_return = sum(T_rewards)/10.
        print("target return: %d, eval return: %d" % (ret, eval_return))

        eval_time = time.time() - eval_start_time  # Calculate evaluation time

        self.model.train(True)
        return eval_return, eval_time


class Env():
    def __init__(self, args):
        self.device = args.device
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', args.seed)
        self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
        self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.use_action_fusion = args.use_action_fusion

        # Define a dictionary for games with special ROM names
        special_rom_names = {
            'montezumarevenge': 'montezuma_revenge',
            'kungfumaster': 'kung_fu_master',
            'roadrunner': 'road_runner',
            'battlezone': 'battle_zone',
            'bankheist': 'bank_heist',
            'fishingderby': 'fishing_derby',
            'mspacman': 'ms_pacman',
            'spaceinvaders': 'space_invaders',
        }

        # Get the correct ROM name using the dictionary or default to the game name
        rom_name = special_rom_names.get(args.game, args.game)
        self.ale.loadROM(atari_py.get_game_path(rom_name)) # Load the ROM

        # Fused action mapping
        self.fused_action_map = self._create_fused_action_map(args.game) if self.use_action_fusion else None
        
        if self.fused_action_map is not None:
            self.actions = dict([i, e] for i, e in zip(range(len(self.fused_action_map)), self.fused_action_map.keys())) # fused actions
        else:
            actions = self.ale.getMinimalActionSet()
            self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions)) # original actions 
 
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.window = args.history_length  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=args.history_length)
        self.training = True  # Consistent with model training mode
    
    """simple fusion"""
    def _create_fused_action_map(self, game):
        if game.lower() == 'hero':
            return {
                0: [0],    # NOOP
                1: [1],    # FIRE
                2: [2, 10], # UP, UPFIRE
                3: [3, 11], # RIGHT, RIGHTFIRE
                4: [4, 12], # LEFT, LEFTFIRE
                5: [5, 13], # DOWN, DOWNFIRE
                6: [6, 14], # UPRIGHT, UPRIGHTFIRE
                7: [7, 15], # UPLEFT, UPLEFTFIRE
                8: [8, 16], # DOWNRIGHT, DOWNRIGHTFIRE
                9: [9, 17], # DOWNLEFT, DOWNLEFTFIRE
            }
        elif game.lower() == 'kungfumaster':
            return {
                0: [0],     # NOOP
                1: [1],     # UP
                2: [2, 7],  # RIGHT, RIGHTFIRE
                3: [3, 8],  # LEFT, LEFTFIRE
                4: [4, 9],  # DOWN, DOWNFIRE
                5: [5, 12], # DOWNRIGHT, DOWNRIGHTFIRE
                6: [6, 13], # DOWNLEFT, DOWNLEFTFIRE
                7: [10],    # UPRIGHTFIRE
                8: [11],    # UPLEFTFIRE
            }
        else:
            return None  # No reverse mapping for other games

    """fuse according to last 1%"""
    # def _create_fused_action_map(self, game):
    #     if game.lower() == 'hero':
    #         return {
    #             0: [2, 10],
    #             1: [6, 7],
    #             2: [0],
    #             3: [15, 14],
    #             4: [1],
    #             5: [5, 13],
    #             6: [4, 9],
    #             7: [8, 3],
    #             8: [12, 17],
    #             9: [11, 16]
    #         }
    #     elif game.lower() == 'kungfumaster':
    #         return {
    #             0: [2, 1],
    #             1: [0],
    #             2: [4, 10],
    #             3: [12, 6],
    #             4: [9, 7],
    #             5: [13, 5],
    #             6: [3, 11],
    #             7: [8]
    #         }
    #     else:
    #         # No fusion for other games
    #         return None
        
    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
        return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            self.ale.reset_game()
            # Perform up to 30 random no-ops before starting
            for _ in range(random.randrange(30)):
                self.ale.act(0)  # Assumes raw action 0 is always no-op
                if self.ale.game_over():
                    self.ale.reset_game()
        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        return torch.stack(list(self.state_buffer), 0)

    # def step(self, action):
    #     # Repeat action 4 times, max pool over last 2 frames
    #     frame_buffer = torch.zeros(2, 84, 84, device=self.device)
    #     reward, done = 0, False
    #     for t in range(4):
    #         reward += self.ale.act(self.actions.get(action))
    #         if t == 2:
    #             frame_buffer[0] = self._get_state()
    #         elif t == 3:
    #             frame_buffer[1] = self._get_state()
    #         done = self.ale.game_over()
    #         if done:
    #             break
    #     observation = frame_buffer.max(0)[0]
    #     self.state_buffer.append(observation)
    #     # Detect loss of life as terminal in training mode
    #     if self.training:
    #         lives = self.ale.lives()
    #         if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
    #             self.life_termination = not done  # Only set flag when not truly done
    #             done = True
    #         self.lives = lives
    #     # Return state, reward, done
    #     return torch.stack(list(self.state_buffer), 0), reward, done
    
    def step(self, action):
        # Repeat actions 4 times, max pool over last 2 frames
        frame_buffer = torch.zeros(2, 84, 84, device=self.device)
        reward, done = 0, False

        for t in range(4):
            
            if self.use_action_fusion and self.fused_action_map is not None:
                for a in reversed(self.fused_action_map[self.actions.get(action)]):
                    reward += self.ale.act(a) # Use fused actions
            else:
                reward += self.ale.act(self.actions.get(action)) # Use original actions

            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)

        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

    def render(self):
        cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()


class Args:
    def __init__(self, game, seed):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        self.max_episode_length = 108e3
        self.game = game
        self.history_length = 4
        self.use_action_fusion = False
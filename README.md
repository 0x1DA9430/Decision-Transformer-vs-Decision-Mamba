# Transformer vs. Mamba: Analysing the Complexity of Sequential Decision-Making in Atari Games Environments

## Dependencies

```bash
conda create -n [env name] python=3.9
```

### Activate environment and install requirements

```bash
conda activate [env name]
```

```bash
pip install -r requirements.txt
```

> it may take 5 minutes, depends on the device

#### Or PyTorch for CUDA 12.1 (or CUDA 11.8)

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Install mamba-ssm and causal-conv1d manually

```bash
pip install mamba-ssm --no-cache-dir
```

```bash
pip install causal-conv1d --no-cache-dir
```

> To be specific:
> `mamba-ssm==2.0.3`
> `causal-conv1d==1.2.2.post1`
> newer versions may also work.

### Upgrade charset_normalizer manually (if needed)

```bash
pip install --upgrade charset_normalizer
```

## Download dataset

### Download dqn_replay

Create a directory for dataset and download the datasets using [gsutil](https://cloud.google.com/storage/docs/gsutil_install#install) (you may install gsutil first).

```bash
mkdir [DIRECTORY] # create a directory for the dataset
gsutil -m cp -R gs://atari-replay-datasets/dqn/[GAME_NAME] [DIRECTORY]
```

e.g.

```bash
gsutil -m cp -R gs://atari-replay-datasets/dqn/Breakout /msc-project/atari/data/data_atari
```

### Download ROMS (optional, as the ROMS are already included in the repository)

```bash
wget http://www.atarimania.com/roms/Roms.rar
unrar x Roms.rar
```

Load the ROMS

``` bash
python -m atari_py.import_roms ROMS
```

## Run a Minimal Experiment

### Template

```bash
python train_atari.py \
        --game '[GAME NAME]' \
        --data_dir_prefix [PATH TO DATA DIR] \
        --context_length 5 \
        --n_layer 3 \
        --n_embd 8 \
        --token_mixer '[mamba or attn]' \
        --epochs 2 \
        --batch_size 64 \
        --num_steps 2 \
        --num_buffers 1 \
        --trajectories_per_buffer 5 \
        --output [PATH TO OUTPUT DIR] \
        --experiment [experiment name] \
        --seed 123
```

e.g.

```bash
python train_atari.py \
        --game 'Breakout' \
        --data_dir_prefix ./data/data_atari/ \
        --context_length 5 \
        --n_layer 3 \
        --n_embd 8 \
        --token_mixer 'mamba' \
        --epochs 2 \
        --batch_size 64 \
        --num_steps 5000 \
        --num_buffers 50 \
        --trajectories_per_buffer 5 \
        --output ./output/ \
        --experiment test_experiment \
        --seed 123 \
```

<!-- or

```bash
python train_atari.py --game 'Hero' --data_dir_prefix ./data/data_atari/ --context_length 10 --token_mixer 'mamba' --output ./output/ --experiment test_experiment --seed 123 --num_steps 5000 --trajectories_per_buffer 10 --use_action_fusion > ./output/test_experiment.log 2>&1
``` -->

<!-- ### Job submission to Eddie cluster

Experiments for context length 30

```bash
qsub dm_atari_breakout.sh
qsub dt_atari_breakout.sh
qsub dm_atari_qbert.sh
qsub dt_atari_qbert.sh

qsub dm_atari_hero.sh
qsub dt_atari_hero.sh
qsub dm_atari_kungfumaster.sh
qsub dt_atari_kungfumaster.sh

qsub dm_atari_seaquest.sh
qsub dt_atari_seaquest.sh
qsub dm_atari_pong.sh
qsub dt_atari_pong.sh
```

Experiments for context length 10

```bash
qsub 10_dm_atari_breakout.sh
qsub 10_dt_atari_breakout.sh
qsub 10_dm_atari_qbert.sh
qsub 10_dt_atari_qbert.sh

qsub 10_dm_atari_hero.sh
qsub 10_dt_atari_hero.sh
qsub 10_dm_atari_kungfumaster.sh
qsub 10_dt_atari_kungfumaster.sh

qsub 10_dm_atari_seaquest.sh
qsub 10_dt_atari_seaquest.sh
qsub 10_dm_atari_pong.sh
qsub 10_dt_atari_pong.sh

qsub 10_dm_atari_roadrunner.sh
qsub 10_dt_atari_roadrunner.sh
qsub 10_dm_atari_alien.sh
qsub 10_dt_atari_alien.sh

qsub 10_dm_atari_battlezone.sh
qsub 10_dt_atari_battlezone.sh
qsub 10_dm_atari_bankheist.sh
qsub 10_dt_atari_bankheist.sh

qsub 10_dm_atari_fishingderby.sh
qsub 10_dt_atari_fishingderby.sh
qsub 10_dm_atari_zaxxon.sh
qsub 10_dt_atari_zaxxon.sh

qsub 10_dm_atari_mspacman.sh
qsub 10_dt_atari_mspacman.sh
qsub 10_dm_atari_spaceinvaders.sh
qsub 10_dt_atari_spaceinvaders.sh
```

Experiments for context length 50

```bash
qsub 50_dm_atari_breakout.sh
qsub 50_dt_atari_breakout.sh
qsub 50_dm_atari_qbert.sh
qsub 50_dt_atari_qbert.sh

qsub 50_dm_atari_hero.sh
qsub 50_dt_atari_hero.sh
qsub 50_dm_atari_kungfumaster.sh
qsub 50_dt_atari_kungfumaster.sh

qsub 50_dm_atari_seaquest.sh
qsub 50_dt_atari_seaquest.sh
qsub 50_dm_atari_pong.sh
qsub 50_dt_atari_pong.sh
```

Experiments for context length 100

```bash
qsub 100_dm_atari_breakout.sh
qsub 100_dt_atari_breakout.sh
qsub 100_dm_atari_qbert.sh
qsub 100_dt_atari_qbert.sh

qsub 100_dm_atari_hero.sh
qsub 100_dt_atari_hero.sh
qsub 100_dm_atari_kungfumaster.sh
qsub 100_dt_atari_kungfumaster.sh
``` -->

### Run game complexity analysis

```bash
python analyze_atari_data.py --game [GAME NAME] 
```

e.g.

```bash
python analyze_atari_data.py --game Breakout 
```

### Use action fusion for Hero/KungFuMaster

add `--use_action_fusion` as flag at the end of the command

e.g.

```bash
python train_atari.py \
        --game 'Hero' \
        --data_dir_prefix ./data/data_atari/ \
        --context_length 5 \
        --n_layer 3 \
        --n_embd 8 \
        --token_mixer 'mamba' \
        --epochs 2 \
        --batch_size 64 \
        --num_steps 5000 \
        --num_buffers 50 \
        --trajectories_per_buffer 5 \
        --output ./output/ \
        --experiment test_experiment \
        --seed 123 \
        --use_action_fusion
```
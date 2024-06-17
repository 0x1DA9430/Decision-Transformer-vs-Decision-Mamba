# Atari

We build our Atari implementation on top of [minGPT](https://github.com/karpathy/minGPT) and benchmark our results on the [DQN-replay](https://github.com/google-research/batch_rl) dataset.

## Dependencies

```
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

### Install PyTorch 2.3.1 and CUDA 11.8

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### Or PyTorch and CUDA 12.1
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

> For now (10 Jun 2024):
>
> `mamba-ssm==2.0.3` and `causal-conv1d==1.2.2.post1`

## Download dataset

### Download dqn_replay

Create a directory for dataset and load the datasets using [gsutil](https://cloud.google.com/storage/docs/gsutil_install#install). Replace `[DIRECTORY_NAME]` and `[GAME_NAME]` accordingly (e.g., `./dqn_replay` for `[DIRECTORY_NAME]` and `Breakout` for `[GAME_NAME]`)

```bash
mkdir [DIRECTORY_NAME]
gsutil -m cp -R gs://atari-replay-datasets/dqn/[GAME_NAME] [DIRECTORY_NAME]
```

### Download ROMS (needed when using atari-py)

```bash
wget http://www.atarimania.com/roms/Roms.rar
unrar x Roms.rar

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
        --token_mixer 'mamba' \
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
        --game 'Pong' \
        --data_dir_prefix ./data/data_atari/ \
        --context_length 5 \
        --n_layer 3 \
        --n_embd 8 \
        --token_mixer 'mamba' \
        --epochs 2 \
        --batch_size 64 \
        --num_steps 500000 \
        --num_buffers 50 \
        --trajectories_per_buffer 5 \
        --output ./output/ \
        --experiment min_dmamba_pong \
        --seed 123
```



> use single eGPU (reset after every boot)
> `export CUDA_VISIBLE_DEVICES=0`

> Notice: `atari-py` is fully deprecated and no future updates, bug fixes or releases will be made. Please use the official [Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment) Python package (`ale-py`) instead; it is *partially backwards compatible* with `atari-py` code.


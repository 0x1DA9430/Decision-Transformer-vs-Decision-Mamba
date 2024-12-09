# Decision Transformer vs. Decision Mamba: Analysing the Complexity of Sequential Decision Making in Atari Games
[![arXiv](https://img.shields.io/badge/arXiv-2412.00725-b31b1b.svg)](https://arxiv.org/abs/2412.00725)
## Abstract

This work analyses the disparity in performance between [Decision Transformer (DT)](https://github.com/kzl/decision-transformer) and [Decision Mamba (DM)](https://github.com/Toshihiro-Ota/decision-mamba) in sequence modelling reinforcement learning tasks for different Atari games. The study first observed that DM generally outperformed DT in the games Breakout and Qbert, while DT performed better in more complicated games, such as Hero and Kung Fu Master. To understand these differences, we expanded the number of games to 12 and performed a comprehensive analysis of game characteristics, including action space complexity, visual complexity, average trajectory length, and average steps to the first non-zero reward. In order to further analyse the key factors that impact the disparity in performance between DT and DM, we employ various approaches, including quantifying visual complexity, random forest regression, correlation analysis, and action space simplification strategies. The results indicate that the performance gap between DT and DM is affected by the complex interaction of multiple factors, with the complexity of the action space and visual complexity (particularly evaluated by compression ratio) being the primary determining factors. DM performs well in environments with simple action and visual elements, while DT shows an advantage in games with higher action and visual complexity. Our findings contribute to a deeper understanding of how the game characteristics affect the performance difference in sequential modelling reinforcement learning, potentially guiding the development of future model design and applications for diverse and complex environments.

<p align="center">
<img src="https://github.com/0x1DA9430/0x1DA9430/blob/b06d18df56253d8c6bf138c44fa797c5bad1b2ec/assert/dt_dm_arc.png" width="500" height="auto" />
</p>
<p align="center">
The Decision Transformer (Left) and Decision Mamba (Right). $N$ represents normalization layers, activation function $\sigma$ stands for GELU (Gaussian Error Linear Unit), and $+$ are addition operations for skip connections.
</p>


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

## Dataset

### Download dqn_replay

Create a directory for dataset 

```bash
mkdir [DIRECTORY]
```
> or use the existing directory /data/data_atari

Download the datasets using [gsutil](https://cloud.google.com/storage/docs/gsutil_install#install) (you may install gsutil first).


```bash
gsutil -m cp -R gs://atari-replay-datasets/dqn/[GAME_NAME] [DIRECTORY]
```

e.g.

```bash
gsutil -m cp -R gs://atari-replay-datasets/dqn/Breakout ./msc-project/atari/data/data_atari
```

### Download game ROMS 
```bash
wget http://www.atarimania.com/roms/Roms.rar
unrar x Roms.rar
```

Load the ROMS

``` bash
python -m atari_py.import_roms ROMS
```

## Run a Minimal Experiment

#### Command Template

```bash
python train_atari.py \
        --game '[GAME NAME]' \
        --data_dir_prefix [PATH TO DATA DIR] \
        --context_length 5 \
        --n_layer 3 \
        --n_embd 8 \
        --token_mixer '[mamba for DM / attn for DT]' \
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
        --seed 123
```

## Run a Standard Experiment

- `run.sh` is a script to run the standard experiments for Breakout, with 3 random seeds, context length 10.

### Use action fusion for the game Hero/KungFuMaster

add `--use_action_fusion` as the flag at the end of the command

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


## Run complexity analysis for the games in the dataset

Random sampling the data for analysis:

```bash
python dataset_analyze_rand.py --game [GAME NAME] 
```

Only use the last 1% of the data for analysis:

```bash
python dataset_analyze_last_1p.py --game [GAME NAME] 
```

## Analyze the outputs

- **Calculate the Normalized Scores**

  - `atari/output_analyze/analyse_output.ipynb`

- **Visualize the Results**

  - `atari/output_analyze/plot_output.ipynb`

- **Random Forest Regression, Correlation Analysis**

  - `atari/output_analyze/regression_analysis.ipynb`

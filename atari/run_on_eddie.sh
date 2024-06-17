#!/bin/bash

# Grid Engine options (lines prefixed with #$)
#$ -N attari_2
#$ -cwd

#$ -l h_rt=12:00:00

# Request one GPU in the gpu queue:
#$ -q gpu
#$ -pe gpu-a100 1
#$ -l h_vmem=40G

# Combine stdout and stderr into a single log file
#$ -j y
#$ -o output.log

# Initialise the environment
. /etc/profile.d/modules.sh
module load cuda/12
module load anaconda/2024

conda activate ssm
python -m atari_py.import_roms ROMS

# Run the executable
python train_atari.py \
        --game 'Pong' \
        --data_dir_prefix ./data/data_atari/ \
        --context_length 5 \
        --n_layer 3 \
        --n_embd 8 \
        --token_mixer 'mamba' \
        --epochs 2 \
        --batch_size 64 \
        --num_steps 2 \
        --num_buffers 1 \
        --trajectories_per_buffer 5 \
        --output ./output/ \
        --experiment min_dmamba_pong \
        --seed 123
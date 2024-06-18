#!/bin/bash

# Grid Engine options (lines prefixed with #$)
#$ -N min_exp
#$ -cwd

#$ -l h_rt=12:00:00

# Request one GPU in the gpu queue:
#$ -q gpu
#$ -pe gpu-a100 1
#$ -l h_vmem=80G

# Save log
#$ -j y
#$ -o min_output.log

# Initialise the environment
. /etc/profile.d/modules.sh
module load cuda/12
module load anaconda/2024

conda activate ssm
python -m atari_py.import_roms ROMS > /dev/null 2>&1

# Run the executable
# Define data and output directories
DATA_DIR=./data/data_atari/
OUT_DIR=./output/min_atari_breakout_eddie_2/

# Run the min experiments for Breakout with dmamba
EXP_Q=min_dmamba_breakout
for seed in 123 321; do 
    python train_atari.py \
        --game 'Breakout' \
        --data_dir_prefix $DATA_DIR \
        --context_length 30 \
        --n_layer 3 \
        --n_embd 8 \
        --token_mixer 'mamba' \
        --epochs 2 \
        --batch_size 64 \
        --num_steps 2 \
        --num_buffers 1 \
        --trajectories_per_buffer 5 \
        --output $OUT_DIR \
        --experiment $EXP_Q \
        --seed $seed
done

# Run the min experiments for Breakout with attn
EXP_DTQ=min_dtrans_breakout
for seed in 123 321; do 
    python train_atari.py \
        --game 'Breakout' \
        --data_dir_prefix $DATA_DIR \
        --context_length 30 \
        --n_layer 3 \
        --n_embd 8 \
        --token_mixer 'attn' \
        --epochs 2 \
        --batch_size 64 \
        --num_steps 2 \
        --num_buffers 1 \
        --trajectories_per_buffer 5 \
        --output $OUT_DIR \
        --experiment $EXP_DTQ \
        --seed $seed
done
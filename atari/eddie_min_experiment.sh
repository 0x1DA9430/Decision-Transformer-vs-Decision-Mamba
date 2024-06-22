#!/bin/bash

# Grid Engine options (lines prefixed with #$)
#$ -N min_exp
#$ -cwd

#$ -l h_rt=4:00:00

# Request one GPU in the gpu queue:
#$ -q gpu
#$ -pe gpu-a100 1
#$ -l h_vmem=80G

# Send mail at beginning/end of job
#$ -m be
#$ -M s2524927@ed.ac.uk

# Save log
#$ -j y
#$ -o ./output/min_exp/atari_breakout_eddie/min_output.log

# Initialise the environment
. /etc/profile.d/modules.sh
module load cuda/12
module load anaconda/2024

conda activate ssm
python -m atari_py.import_roms ROMS > /dev/null 2>&1

# Run the executable
# Define data and output directories
DATA_DIR=./data/data_atari/
OUT_DIR=./output/min_exp/atari_breakout_eddie/

# Run the min experiments for Breakout with dmamba
EXP_Q=min_dmamba_breakout
for seed in 123 321; do 
    python ../train_atari.py \
        --game 'Breakout' \
        --data_dir_prefix $DATA_DIR \
        --context_length 5 \
        --n_layer 3 \
        --n_embd 8 \
        --token_mixer 'mamba' \
        --epochs 2 \
        --batch_size 64 \
        --num_steps 50 \
        --num_buffers 50 \
        --trajectories_per_buffer 5 \
        --output $OUT_DIR \
        --experiment $EXP_Q \
        --seed $seed
done

# Run the min experiments for Breakout with attn
EXP_DTQ=min_dtrans_breakout
for seed in 123 321; do 
    python ../train_atari.py \
        --game 'Breakout' \
        --data_dir_prefix $DATA_DIR \
        --context_length 5 \
        --n_layer 3 \
        --n_embd 8 \
        --token_mixer 'attn' \
        --epochs 2 \
        --batch_size 64 \
        --num_steps 50 \
        --num_buffers 50 \
        --trajectories_per_buffer 5 \
        --output $OUT_DIR \
        --experiment $EXP_DTQ \
        --seed $seed
done
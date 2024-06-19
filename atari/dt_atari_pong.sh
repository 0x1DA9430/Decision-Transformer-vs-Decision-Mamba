#!/bin/bash

# Grid Engine options (lines prefixed with #$)
#$ -N dm_atari_pong
#$ -cwd

#$ -l h_rt=24:00:00

# Request one GPU in the gpu queue:
#$ -q gpu
#$ -pe gpu-a100 1
#$ -l h_vmem=80G

# Send mail at beginning/end of job
#$ -m be
#$ -M s2524927@ed.ac.uk

# Save log
#$ -j y
#$ -o ./output/atari_pong_eddie/dt_pong_output.log

# Initialise the environment
. /etc/profile.d/modules.sh
module load cuda/12
module load anaconda/2024

conda activate ssm
python -m atari_py.import_roms ROMS > /dev/null 2>&1

# Run the executable
DATA_DIR=./data/data_atari/
OUT_DIR=./output/atari_pong_eddie/

EXP_DTQ=dtrans_pong
for seed in 123 132 213 231 312; do python train_atari.py --game 'Pong' --data_dir_prefix $DATA_DIR --context_length 30 --token_mixer 'attn' --output $OUT_DIR --experiment $EXP_DTQ --seed $seed; done

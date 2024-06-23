#!/bin/bash

# Grid Engine options (lines prefixed with #$)
#$ -N dm_atari_kungfumaster
#$ -cwd

#$ -l h_rt=17:00:00

# Request one GPU in the gpu queue:
#$ -q gpu
#$ -pe gpu-a100 1
#$ -l h_vmem=90G

# Send mail at beginning/end of job
#$ -m be
#$ -M s2524927@ed.ac.uk

# Save log
#$ -j y
#$ -o ./output/context_30_rtg_5max/atari_kungfumaster_eddie/dm_kungfumaster_output.log

# Initialise the environment
. /etc/profile.d/modules.sh
module load cuda/12
module load anaconda/2024

conda activate ssm
python -m atari_py.import_roms ROMS > /dev/null 2>&1

# Run the executable
DATA_DIR=./data/data_atari/
OUT_DIR=./output/context_30_rtg_5max/atari_kungfumaster_eddie/

EXP_Q=dmamba_kungfumaster
for seed in 123 132 321; do python train_atari.py --game 'KungFuMaster' --data_dir_prefix $DATA_DIR --context_length 30 --token_mixer 'mamba' --output $OUT_DIR --experiment $EXP_Q --seed $seed; done
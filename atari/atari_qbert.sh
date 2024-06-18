#!/bin/bash

# Grid Engine options (lines prefixed with #$)
#$ -N attari_qbert
#$ -cwd
#$ -o attari_qbert_output.log 
#$ -e attari_qbert_error.log
#$ -l h_rt=24:00:00


# Request one GPU in the gpu queue:
#$ -q gpu
#$ -pe gpu-a100 1
#$ -l h_vmem=40G

# Save log
#$ -o output_qbert.log
#$ -e error_qbert.log

# Initialise the environment
. /etc/profile.d/modules.sh
module load cuda/12
module load anaconda/2024

conda activate ssm
python -m atari_py.import_roms ROMS > /dev/null 2>&1

# Run the executable
DATA_DIR=./data/data_atari/
OUT_DIR=./output/atari_qbert_eddie/

EXP_Q=dmamba_qbert
for seed in 123 132 213 231 312 321; do python train_atari.py --game 'Qbert' --data_dir_prefix $DATA_DIR --context_length 30 --token_mixer 'mamba' --output $OUT_DIR --experiment $EXP_Q --seed $seed; done

EXP_DTQ=dtrans_qbert
for seed in 123 132 213 231 312 321; do python train_atari.py --game 'Qbert' --data_dir_prefix $DATA_DIR --context_length 30 --token_mixer 'attn' --output $OUT_DIR --experiment $EXP_DTQ --seed $seed; done

#!/bin/bash

# Grid Engine options (lines prefixed with #$)
#$ -N test_run
#$ -cwd
#$ -l h_rt=12:00:00


# Request one GPU in the gpu queue:
#$ -q gpu
#$ -pe gpu-a100 1
#$ -l h_vmem=40G


# Initialise the environment
. /etc/profile.d/modules.sh
module load cuda/12
module load anaconda/2024

conda activate ssm
python -m atari_py.import_roms ROMS

# Run the executable
bash run_atari_2.sh > run_atari_2.log 2>&1
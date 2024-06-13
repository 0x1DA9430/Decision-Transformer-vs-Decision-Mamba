#!/bin/bash

# Grid Engine options (lines prefixed with #$)
$ -N test_run
$ -cwd
$ -l h_rt=12:00:00
$ -l h_vmem=24G


# Request one GPU in the gpu queue:
$ -q gpu&nbsp;
$ -pe gpu-a100 1


# Initialise the environment modules and load CUDA version 11.0.2
. /etc/profile.d/modules.sh
# module load cuda/12.1.1
module load anaconda

conda activate ssm2


# Run the executable
bash run_atari_2.sh
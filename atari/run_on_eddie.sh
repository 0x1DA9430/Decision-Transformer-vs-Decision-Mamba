#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# Runtime limit of 1 hour:
$ -N test_run
$ -cwd
$ -l h_rt=12:00:00
$ -l h_vmem=24G
#
# Set working directory to the directory where the job is submitted from:

#
# Request one GPU in the gpu queue:
$ -q gpu&nbsp;
$ -pe gpu-a100 1
#
# Request 4 GB system RAM&nbsp;
# the total system RAM available to the job is the value specified here multiplied by&nbsp;
# the number of requested GPUs (above)


# Initialise the environment modules and load CUDA version 11.0.2
. /etc/profile.d/modules.sh
# module load cuda/12.1.1
module load anaconda

conda activate ssm2


# Run the executable
./run_atari_2.sh
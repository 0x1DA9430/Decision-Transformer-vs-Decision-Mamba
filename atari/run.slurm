#!/bin/bash
#
#SBATCH --job-name=dm_10_breakout
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=07:00:00

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=tc064-s2524927

# Load the required modules
module load python/3.9.13-gpu
module load module load pytorch/2.2.0-gpu
module load nvidia/nvhpc

# source /work/tc064/tc064/s2524927/myvenv/bin/activate

python -m atari_py.import_roms ROMS > /dev/null 2>&1

# Set directories
DATA_DIR=./data/data_atari/
OUT_DIR=./output/cirrus_exp_test/atari_breakout_cirrus/

EXP_Q=dmamba_breakout

# # Run the script with different seeds
# for seed in 123 132 321; do
#     srun python train_atari.py --game 'Breakout' --data_dir_prefix $DATA_DIR --context_length 10 --token_mixer 'mamba' --output $OUT_DIR --experiment $EXP_Q -$
# done

srun python run_cirrus.py
# Run the executable
DATA_DIR=./data/data_atari/
OUT_DIR=./output/atari_breakout_local/

EXP_Q=dmamba_breakout
for seed in 123; do python train_atari.py --game 'Breakout' --data_dir_prefix $DATA_DIR --context_length 30 --token_mixer 'mamba' --output $OUT_DIR --experiment $EXP_Q --seed $seed; done

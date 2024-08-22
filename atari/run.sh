# Run the executable
DATA_DIR=./data/data_atari/
OUT_DIR=./output/context_10_rtg_5max/atari_breakout_local/

EXP_Q=dmamba_breakout
for seed in 123 132 321; do python train_atari.py --game 'Breakout' --data_dir_prefix $DATA_DIR --context_length 10 --token_mixer 'mamba' --output $OUT_DIR --experiment $EXP_Q --seed $seed; done
# for seed in 231 312; do python train_atari.py --game 'Breakout' --data_dir_prefix $DATA_DIR --context_length 10 --token_mixer 'mamba' --output $OUT_DIR --experiment $EXP_Q --seed $seed; done

EXP_DTQ=dtrans_breakout
for seed in 123 132 321; do python train_atari.py --game 'Breakout' --data_dir_prefix $DATA_DIR --context_length 10 --token_mixer 'attn' --output $OUT_DIR --experiment $EXP_DTQ --seed $seed; done
# for seed in 231 312; do python train_atari.py --game 'Breakout' --data_dir_prefix $DATA_DIR --context_length 10 --token_mixer 'attn' --output $OUT_DIR --experiment $EXP_DTQ --seed $seed; done
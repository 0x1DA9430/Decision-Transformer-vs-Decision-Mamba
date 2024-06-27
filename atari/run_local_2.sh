# Run the executable
DATA_DIR=./data/data_atari/
OUT_DIR=./output/context_50_rtg_5max/atari_kungfumaster_local/


EXP_DTQ=dtrans_kungfumaster
for seed in 123 132 321; do python train_atari.py --game 'KungFuMaster' --data_dir_prefix $DATA_DIR --context_length 50 --token_mixer 'attn' --output $OUT_DIR --experiment $EXP_DTQ --seed $seed; done

EXP_Q=dmamba_kungfumaster
for seed in 123 132 321; do python train_atari.py --game 'KungFuMaster' --data_dir_prefix $DATA_DIR --context_length 50 --token_mixer 'mamba' --output $OUT_DIR --experiment $EXP_Q --seed $seed; done

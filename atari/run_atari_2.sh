#!/bin/bash

DATA_DIR=./data/data_atari/
OUT_DIR=./output/atari_2_eddie/


# EXP_P=dmamba_pong
# for seed in 123 321 213; do python train_atari.py --game 'Pong' --data_dir_prefix $DATA_DIR --context_length 50 --token_mixer 'mamba' --output $OUT_DIR --experiment $EXP_P --seed $seed; done

# EXP_DTP=dtrans_pong
# for seed in 123 321 213; do python train_atari.py --game 'Pong' --data_dir_prefix $DATA_DIR --context_length 50 --token_mixer 'attn' --output $OUT_DIR --experiment $EXP_DTP --seed $seed; done


EXP_S=dmamba_seaquest
for seed in 123 132 213 231 312 321; do python train_atari.py --game 'Seaquest' --data_dir_prefix $DATA_DIR --context_length 30 --token_mixer 'mamba' --output $OUT_DIR --experiment $EXP_S --seed $seed; done

EXP_DTS=dtrans_seaquest
for seed in 123 132 213 231 312 321; do python train_atari.py --game 'Seaquest' --data_dir_prefix $DATA_DIR --context_length 30 --token_mixer 'attn' --output $OUT_DIR --experiment $EXP_DTS --seed $seed; done




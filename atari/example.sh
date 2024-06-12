#!/bin/bash

DATA_DIR=./data/data_atari/
OUT_DIR=./output/atari/

# Minimal experiment settings
CONTEXT_LENGTH=5
N_LAYER=3
N_EMBD=8
EPOCHS=1
BATCH_SIZE=128
NUM_STEPS=2
NUM_BUFFERS=1
TRAJECTORIES_PER_BUFFER=5

# Breakout Game Minimal Experiment
EXP_B=dmamba_breakout_minimal
for seed in 123 231 312; do
    python train_atari.py \
        --game 'Breakout' \
        --data_dir_prefix $DATA_DIR \
        --context_length $CONTEXT_LENGTH \
        --n_layer $N_LAYER \
        --n_embd $N_EMBD \
        --token_mixer 'attn' \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --num_steps $NUM_STEPS \
        --num_buffers $NUM_BUFFERS \
        --trajectories_per_buffer $TRAJECTORIES_PER_BUFFER \
        --output $OUT_DIR \
        --experiment $EXP_B \
        --seed $seed
done

# Qbert Game Minimal Experiment
EXP_Q=dmamba_qbert_minimal
for seed in 123 231 312; do
    python train_atari.py \
        --game 'Qbert' \
        --data_dir_prefix $DATA_DIR \
        --context_length $CONTEXT_LENGTH \
        --n_layer $N_LAYER \
        --n_embd $N_EMBD \
        --token_mixer 'attn' \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --num_steps $NUM_STEPS \
        --num_buffers $NUM_BUFFERS \
        --trajectories_per_buffer $TRAJECTORIES_PER_BUFFER \
        --output $OUT_DIR \
        --experiment $EXP_Q \
        --seed $seed
done

# Pong Game Minimal Experiment
EXP_P=dmamba_pong_minimal
for seed in 123 231 312; do
    python train_atari.py \
        --game 'Pong' \
        --data_dir_prefix $DATA_DIR \
        --context_length $CONTEXT_LENGTH \
        --n_layer $N_LAYER \
        --n_embd $N_EMBD \
        --token_mixer 'attn' \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --num_steps $NUM_STEPS \
        --num_buffers $NUM_BUFFERS \
        --trajectories_per_buffer $TRAJECTORIES_PER_BUFFER \
        --output $OUT_DIR \
        --experiment $EXP_P \
        --seed $seed
done

# Seaquest Game Minimal Experiment
EXP_S=dmamba_seaquest_minimal
for seed in 123 231 312; do
    python train_atari.py \
        --game 'Seaquest' \
        --data_dir_prefix $DATA_DIR \
        --context_length $CONTEXT_LENGTH \
        --n_layer $N_LAYER \
        --n_embd $N_EMBD \
        --token_mixer 'attn' \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --num_steps $NUM_STEPS \
        --num_buffers $NUM_BUFFERS \
        --trajectories_per_buffer $TRAJECTORIES_PER_BUFFER \
        --output $OUT_DIR \
        --experiment $EXP_S \
        --seed $seed
done
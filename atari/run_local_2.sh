# Run the executable
DATA_DIR=./data/data_atari/
OUT_DIR=./output/context_10_rtg_5max_action_fusion/atari_kungfumaster_local/

EXP_Q=dmamba_kungfumaster
# for seed in 123 132 321; do python train_atari.py --game 'KungFuMaster' --data_dir_prefix $DATA_DIR --context_length 10 --token_mixer 'mamba' --output $OUT_DIR --experiment $EXP_Q --seed $seed --use_action_fusion; done
for seed in 231 312; do python train_atari.py --game 'KungFuMaster' --data_dir_prefix $DATA_DIR --context_length 10 --token_mixer 'mamba' --output $OUT_DIR --experiment $EXP_Q --seed $seed --use_action_fusion; done

EXP_DTQ=dtrans_kungfumaster
# for seed in 123 132 321; do python train_atari.py --game 'KungFuMaster' --data_dir_prefix $DATA_DIR --context_length 10 --token_mixer 'attn' --output $OUT_DIR --experiment $EXP_DTQ --seed $seed --use_action_fusion; done
for seed in 231 312; do python train_atari.py --game 'KungFuMaster' --data_dir_prefix $DATA_DIR --context_length 10 --token_mixer 'attn' --output $OUT_DIR --experiment $EXP_DTQ --seed $seed --use_action_fusion; done
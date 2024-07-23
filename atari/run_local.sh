# Run the executable
DATA_DIR=./data/data_atari/
OUT_DIR=./output/context_10_rtg_5max/atari_roadrunner_eddie/

EXP_Q=dmamba_roadrunner
for seed in 231; do python train_atari.py --game 'RoadRunner' --data_dir_prefix $DATA_DIR --context_length 10 --token_mixer 'mamba' --output $OUT_DIR --experiment $EXP_Q --seed $seed; done

# for seed in 123 132 321; do python train_atari.py --game 'roadrunner' --data_dir_prefix $DATA_DIR --context_length 30 --token_mixer 'mamba' --output $OUT_DIR --experiment $EXP_Q --seed $seed --use_action_fusion; done
# for seed in 231 312; do python train_atari.py --game 'roadrunner' --data_dir_prefix $DATA_DIR --context_length 10 --token_mixer 'mamba' --output $OUT_DIR --experiment $EXP_Q --seed $seed --use_action_fusion; done

EXP_DTQ=dtrans_roadrunner
for seed in 231; do python train_atari.py --game 'RoadRunner' --data_dir_prefix $DATA_DIR --context_length 10 --token_mixer 'attn' --output $OUT_DIR --experiment $EXP_DTQ --seed $seed; done

# for seed in 123 132 321; do python train_atari.py --game 'roadrunner' --data_dir_prefix $DATA_DIR --context_length 30 --token_mixer 'attn' --output $OUT_DIR --experiment $EXP_DTQ --seed $seed --use_action_fusion; done
# for seed in 231 312; do python train_atari.py --game 'roadrunner' --data_dir_prefix $DATA_DIR --context_length 10 --token_mixer 'attn' --output $OUT_DIR --experiment $EXP_DTQ --seed $seed --use_action_fusion; done
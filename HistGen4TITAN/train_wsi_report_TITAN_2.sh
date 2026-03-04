#!/bin/bash

model='histgen_titan'
max_length=100
epochs=40

python /home/woody/iwi5/iwi5204h/HistGen4TITAN/main_train_AllinOne.py \
    --slide_embedding_dir /home/woody/iwi5/iwi5204h/HistGen4TITAN/slide_embeddings_new_script/ \
    --ann_path /home/woody/iwi5/iwi5204h/HistGen4TITAN/train_val_test.json \
    --dataset_name wsi_report \
    --model_name $model \
    --max_seq_length $max_length \
    --num_layers 3 \
    --threshold 1 \
    --batch_size 1 \
    --epochs $epochs \
    --lr_ed 1e-4 \
    --step_size 10 \
    --titan_embedding_dim 768 \
    --projection_dim 768 \
    --embedding_format pt \
    --save_dir /home/woody/iwi5/iwi5204h/HistGen4TITAN/Data/TrainingResult/2 \
    --gamma 0.8 \
    --seed 456789 \
    --log_period 1000 \
    --beam_size 3 \
    --d_model 768 \
    --d_vf 768 \
    --use_titan_embeddings \

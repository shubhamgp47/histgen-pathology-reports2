# TITAN-HistGen Test Configuration
model='histgen_titan'  # Changed from 'histgen' to match your new model
max_length=100
epochs=40

python /home/woody/iwi5/iwi5204h/HistGen4TITAN/main_test_AllinOne.py \
    --slide_embedding_dir /home/woody/iwi5/iwi5204h/HistGen4TITAN/slide_embeddings/ \
    --ann_path /home/woody/iwi5/iwi5204h/HistGen4TITAN/train_val_test.json \
    --dataset_name wsi_report \
    --threshold 1 \
    --step_size 10 \
    --model_name $model \
    --max_seq_length $max_length \
    --batch_size 1 \
    --epochs $epochs \
    --save_dir /home/woody/iwi5/iwi5204h/HistGen4TITAN/Data/TestResults/5_seed46/Best24 \
    --load /home/woody/iwi5/iwi5204h/HistGen4TITAN/Data/TrainingResult/5_seed46/model_best_epoch24.pth \
    --beam_size 3 \
    --embedding_format pt \
    --titan_embedding_dim 768 \
    --projection_dim 1536 \
    --d_model 768 \
    --d_vf 768 \
    --num_layers 3 \
    --d_ff 512 \
    --num_heads 8 \
    --dropout 0.1 \
    --drop_prob_lm 0.1 \
    --seed 46 \
    --log_period 1000

export CUDA_VISIBLE_DEVICES=0

# Change the model name, data name and parameters as needed

python -u main.py \
    --model_name 'DPG_Mixer'\
    --exp_name 'deep_learning'\
    --data_name 'METR-LA' \
    --seq_len 12\
    --pred_len 12\
    --lr 1e-3 \
    --batch_size 32 \
    --resume_dir None\
    --output_dir None\
    --info 'DPG_Mixer METR-LA 12-12'\

python -u main.py \
    --model_name 'PGCN'\
    --exp_name 'deep_learning'\
    --data_name 'METR-LA' \
    --seq_len 12\
    --pred_len 12\
    --lr 1e-3 \
    --batch_size 32 \
    --resume_dir None\
    --output_dir None\
    --info 'PGCN METR-LA 12-12'\

python -u main.py \
    --model_name 'STIDGCN' \
    --exp_name 'deep_learning'\
    --data_name 'METR-LA' \
    --seq_len 12\
    --pred_len 12\
    --lr 1e-3 \
    --batch_size 32 \
    --resume_dir None\
    --output_dir None\
    --info 'STIDGCN METR-LA 12-12'\


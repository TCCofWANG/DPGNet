export CUDA_VISIBLE_DEVICES=4



for rate in     0.0005 0.00001 0.00005
do
python -u main.py \
  --model_name 'WAVGCRN'\
  --exp_name 'deep_learning'\
  --data_name 'METR-LA' \
  --seq_len 24\
  --pred_len 24\
  --lr $rate \
  --batch_size 32 \
  --resume_dir None\
  --output_dir None\
  --info '2.12 long-term WAVGCRN METR-LA 24-24'\

done

for rate in     0.0005 0.00001 0.00005
do
python -u main.py \
  --model_name 'WAVGCRN'\
  --exp_name 'deep_learning'\
  --data_name 'METR-LA' \
  --seq_len 24\
  --pred_len 36\
  --lr $rate \
  --batch_size 16 \
  --resume_dir None\
  --output_dir None\
  --info '2.12 long-term WAVGCRN METR-LA 24-24'\

done


for rate in     0.0005 0.00001 0.00005
do
python -u main.py \
  --model_name 'WAVGCRN'\
  --exp_name 'deep_learning'\
  --data_name 'METR-LA' \
  --seq_len 24\
  --pred_len 48\
  --lr $rate \
  --batch_size 16 \
  --resume_dir None\
  --output_dir None\
  --info '2.12 long-term WAVGCRN METR-LA 24-24'\

done




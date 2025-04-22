export CUDA_VISIBLE_DEVICES=4


# long-term prediction 24-24 36 48
for rate in   0.001 0.0001 0.00001 
do
python -u main.py \
  --model_name 'DPG_Mixer'\
  --exp_name 'deep_learning'\
  --data_name 'PEMS08' \
  --seq_len 24\
  --pred_len 24\
  --lr $rate \
  --batch_size 32 \
  --resume_dir None\
  --output_dir None\
  --info '2.6 long-term DPG_Mixer PEMS08 24'\

done

for rate in 0.001 0.0001 0.00001 
do
python -u main.py \
  --model_name 'DPG_Mixer'\
  --exp_name 'deep_learning'\
  --data_name 'PEMS08' \
  --seq_len 24\
  --pred_len 36\
  --lr $rate \
  --batch_size 32 \
  --resume_dir None\
  --output_dir None\
  --info '2.6 long-term DPG_Mixer_V2 PEMS08 36'\

done


for rate in 0.001 0.0001 0.00001 
do
python -u main.py \
  --model_name 'DPG_Mixer'\
  --exp_name 'deep_learning'\
  --data_name 'PEMS08' \
  --seq_len 24\
  --pred_len 48\
  --lr $rate \
  --batch_size 32 \
  --resume_dir None\
  --output_dir None\
  --info '2.6 long-term DPG_Mixer_V3 PEMS08 48'\

done

# long-term prediction 12-24 36 48
#for rate in   0.001 0.0001 0.00001 
#do
#python -u main.py \
#  --model_name 'DPG_Mixer'\
#  --exp_name 'deep_learning'\
#  --data_name 'PEMS08' \
#  --pred_len 24\
#  --lr $rate \
#  --batch_size 32 \
#  --resume_dir None\
#  --output_dir None\
#  --info '2.4 long-term DPG_Mixer PEMS08 24'\
#
#done
#
#for rate in 0.001 0.0001 0.00001 
#do
#python -u main.py \
#  --model_name 'DPG_Mixer'\
#  --exp_name 'deep_learning'\
#  --data_name 'PEMS08' \
#  --pred_len 36\
#  --lr $rate \
#  --batch_size 32 \
#  --resume_dir None\
#  --output_dir None\
#  --info '2.4 long-term DPG_Mixer_V2 PEMS08 36'\
#
#done
#
#
#for rate in 0.001 0.0001 0.00001 
#do
#python -u main.py \
#  --model_name 'DPG_Mixer'\
#  --exp_name 'deep_learning'\
#  --data_name 'PEMS08' \
#  --pred_len 48\
#  --lr $rate \
#  --batch_size 32 \
#  --resume_dir None\
#  --output_dir None\
#  --info '2.4 long-term DPG_Mixer_V3 PEMS08 48'\
#
#done


# original length prediction
#for rate in   0.001 0.005 0.0001 0.0005 0.00001 0.00005
#do
#python -u main.py \
#  --model_name 'DPG_Mixer'\
#  --exp_name 'deep_learning'\
#  --data_name 'PEMS08' \
#  --pred_len 3\
#  --lr $rate \
#  --batch_size 32 \
#  --resume_dir None\
#  --output_dir None\
#  --info '11.19 DPG_Mixer PEMS08 3'\
#
#done
#
#for rate in 0.001 0.005 0.0001 0.0005 0.00001 0.00005
#do
#python -u main.py \
#  --model_name 'DPG_Mixer_V2'\
#  --exp_name 'deep_learning'\
#  --data_name 'PEMS08' \
#  --pred_len 3\
#  --lr $rate \
#  --batch_size 16 \
#  --resume_dir None\
#  --output_dir None\
#  --info '11.19 DPG_Mixer_V2 PEMS08 '\
#
#done
#
#
#for rate in 0.001 0.005 0.0001 0.0005 0.00001 0.00005
#do
#python -u main.py \
#  --model_name 'DPG_Mixer_V3'\
#  --exp_name 'deep_learning'\
#  --data_name 'PEMS08' \
#  --pred_len 3\
#  --lr $rate \
#  --batch_size 16 \
#  --resume_dir None\
#  --output_dir None\
#  --info '11.19 DPG_Mixer_V3 PEMS08'\
#
#done



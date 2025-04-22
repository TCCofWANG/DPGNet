export CUDA_VISIBLE_DEVICES=0


# long-term prediction  12 - 24 36 48

for rate in 1e-3 1e-4 1e-5
do
python -u main.py \
  --model_name 'DPG_Mixer'\
  --exp_name 'deep_learning'\
  --data_name 'METR-LA' \
  --seq_len 24\
  --pred_len 24\
  --lr $rate \
  --batch_size 32 \
  --resume_dir None\
  --output_dir None\
  --info '2.4 long-term DPG_Mixer METR-LA 24-24'\

done


for rate in 1e-3 1e-4 1e-5
do
python -u main.py \
  --model_name 'DPG_Mixer'\
  --exp_name 'deep_learning'\
  --data_name 'METR-LA' \
  --seq_len 24\
  --pred_len 36\
  --lr $rate \
  --batch_size 32 \
  --resume_dir None\
  --output_dir None\
  --info '2.4 long-term DPG_Mixer METR-LA 24-36'\

done

for rate in 1e-3 1e-4 1e-5
do
python -u main.py \
  --model_name 'DPG_Mixer'\
  --exp_name 'deep_learning'\
  --data_name 'METR-LA' \
  --seq_len 24\
  --pred_len 48\
  --lr $rate \
  --batch_size 32 \
  --resume_dir None\
  --output_dir None\
  --info '2.4 long-term DPG_Mixer METR-LA 24-48'\

done



# original length prediction

#for rate in     1e-3 1e-4 1e-6 5e-6 1e-7 
#do
#python -u main.py \
#  --model_name 'DPG_Mixer'\
#  --exp_name 'deep_learning'\
#  --data_name 'METR-LA' \
#  --pred_len 3\
#  --lr $rate \
#  --batch_size 32 \
#  --resume_dir None\
#  --output_dir None\
#  --info '1.12 DPG_Mixer METR-LA 3'\
#
#done
#
#
#for rate in   1e-3 1e-4 1e-6 5e-6 1e-7 
#do
#python -u main.py \
#  --model_name 'DPG_Mixer'\
#  --exp_name 'deep_learning'\
#  --data_name 'METR-LA' \
#  --pred_len 6\
#  --lr $rate \
#  --batch_size 32 \
#  --resume_dir None\
#  --output_dir None\
#  --info '1.12 DPG_Mixer METR-LA 6'\
#
#done

#for rate in 1e-3 1e-4 1e-6 5e-6 1e-7 
#do
#python -u main.py \
#  --model_name 'DPG_Mixer'\
#  --exp_name 'deep_learning'\
#  --data_name 'METR-LA' \
#  --pred_len 12\
#  --lr $rate \
#  --batch_size 32 \
#  --resume_dir None\
#  --output_dir None\
#  --info '1.12 DPG_Mixer METR-LA 12'\
#
#done
#
#
#
#for rate in   1e-3 1e-4 1e-6 5e-6 1e-7 
#do
#python -u main.py \
#  --model_name 'DPG_Mixer'\
#  --exp_name 'deep_learning'\
#  --data_name 'METR-LA' \
#  --pred_len 3\
#  --lr $rate \
#  --batch_size 32 \
#  --embedding_use False\
#  --resume_dir None\
#  --output_dir None\
#  --info '1.12 DPG_Mixer embedding_use False METR-LA 3'\
#
#done
#
#
#for rate in   1e-3 1e-4 1e-6 5e-6 1e-7 
#do
#python -u main.py \
#  --model_name 'DPG_Mixer'\
#  --exp_name 'deep_learning'\
#  --data_name 'METR-LA' \
#  --pred_len 6\
#  --lr $rate \
#  --batch_size 32 \
#  --embedding_use False\
#  --resume_dir None\
#  --output_dir None\
#  --info '1.12 DPG_Mixer embedding_use False METR-LA 6'\
#
#done
#
#for rate in 1e-3 1e-4 1e-6 5e-6 1e-7 
#do
#python -u main.py \
#  --model_name 'DPG_Mixer'\
#  --exp_name 'deep_learning'\
#  --data_name 'METR-LA' \
#  --pred_len 12\
#  --lr $rate \
#  --batch_size 32 \
#  --embedding_use False\
#  --resume_dir None\
#  --output_dir None\
#  --info '1.12 DPG_Mixer embedding_use False METR-LA 12'\
#
#done
#
#
#
#for rate in   1e-3 1e-4 1e-6 5e-6 1e-7 
#do
#python -u main.py \
#  --model_name 'DPG_Mixer'\
#  --exp_name 'deep_learning'\
#  --data_name 'METR-LA' \
#  --pred_len 3\
#  --lr $rate \
#  --batch_size 32 \
#  --norm_use False\
#  --resume_dir None\
#  --output_dir None\
#  --info '1.12 DPG_Mixer norm_use False METR-LA 3'\
#
#done
#
#
#for rate in   1e-3 1e-4 1e-6 5e-6 1e-7 
#do
#python -u main.py \
#  --model_name 'DPG_Mixer'\
#  --exp_name 'deep_learning'\
#  --data_name 'METR-LA' \
#  --pred_len 6\
#  --lr $rate \
#  --batch_size 32 \
#  --norm_use False\
#  --resume_dir None\
#  --output_dir None\
#  --info '1.12 DPG_Mixer norm_use False METR-LA 6'\
#
#done
#
#for rate in 1e-3 1e-4 1e-6 5e-6 1e-7 
#do
#python -u main.py \
#  --model_name 'DPG_Mixer'\
#  --exp_name 'deep_learning'\
#  --data_name 'METR-LA' \
#  --pred_len 12\
#  --lr $rate \
#  --batch_size 32 \
#  --norm_use False\
#  --resume_dir None\
#  --output_dir None\
#  --info '1.12 DPG_Mixer norm_use False METR-LA 12'\
#
#done



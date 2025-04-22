export CUDA_VISIBLE_DEVICES=2


# long-term prediction 24-24 36 48
#for rate in   0.001 0.0001 0.00001 
#do
#python -u main.py \
#  --model_name 'DPG_Mixer'\
#  --exp_name 'deep_learning'\
#  --data_name 'PeMS-Bay' \
#  --seq_len 24\
#  --pred_len 24\
#  --lr $rate \
#  --batch_size 16 \
#  --resume_dir None\
#  --output_dir None\
#  --info '2.6 long-term DPG_Mixer with trend and season PeMS-Bay 24-24'\
#
#done
#
#
#for rate in   0.001 0.0001 0.00001 
#do
#python -u main.py \
#  --model_name 'DPG_Mixer'\
#  --exp_name 'deep_learning'\
#  --data_name 'PeMS-Bay' \
#  --seq_len 24\
#  --pred_len 36\
#  --lr $rate \
#  --batch_size 16 \
#  --resume_dir None\
#  --output_dir None\
#  --info '2.6 long-term DPG_Mixer with trend and season PeMS-Bay 24-36'\
#
#done

for rate in 0.001 0.0001 0.00001 
do
python -u main.py \
  --model_name 'DPG_Mixer'\
  --exp_name 'deep_learning'\
  --data_name 'PeMS-Bay' \
  --seq_len 24\
  --pred_len 48\
  --lr $rate \
  --batch_size 16 \
  --resume_dir None\
  --output_dir None\
  --info '2.6 long-term DPG_Mixer with trend and season PeMS-Bay 24-48'\

done





# long-term prediction 12-24 36 48
#for rate in   0.001 0.0001 0.00001 
#do
#python -u main.py \
#  --model_name 'DPG_Mixer'\
#  --exp_name 'deep_learning'\
#  --data_name 'PeMS-Bay' \
#  --pred_len 24\
#  --lr $rate \
#  --batch_size 32 \
#  --resume_dir None\
#  --output_dir None\
#  --info '2.4 long-term DPG_Mixer with trend and season PeMS-Bay 24'\
#
#done
#
#
#for rate in   0.001 0.0001 0.00001 
#do
#python -u main.py \
#  --model_name 'DPG_Mixer'\
#  --exp_name 'deep_learning'\
#  --data_name 'PeMS-Bay' \
#  --pred_len 36\
#  --lr $rate \
#  --batch_size 32 \
#  --resume_dir None\
#  --output_dir None\
#  --info '2.4 long-term DPG_Mixer with trend and season PeMS-Bay 36'\
#
#done
#
#for rate in 0.001 0.0001 0.00001 
#do
#python -u main.py \
#  --model_name 'DPG_Mixer'\
#  --exp_name 'deep_learning'\
#  --data_name 'PeMS-Bay' \
#  --pred_len 48\
#  --lr $rate \
#  --batch_size 32 \
#  --resume_dir None\
#  --output_dir None\
#  --info '2.4 long-term DPG_Mixer with trend and season PeMS-Bay 48'\
#
#done






# original length prediction

#for rate in   0.001 0.005 0.0001 0.0005 0.00001 0.00005
#do
#python -u main.py \
#  --model_name 'DPG_Mixer'\
#  --exp_name 'deep_learning'\
#  --data_name 'PeMS-Bay' \
#  --pred_len 3\
#  --lr $rate \
#  --batch_size 32 \
#  --resume_dir None\
#  --output_dir None\
#  --info '12.10 DPG_Mixer with trend and season PeMS-Bay 3'\
#
#done
#
#
#for rate in   0.001 0.005 0.0001 0.0005 0.00001 0.00005
#do
#python -u main.py \
#  --model_name 'DPG_Mixer'\
#  --exp_name 'deep_learning'\
#  --data_name 'PeMS-Bay' \
#  --pred_len 6\
#  --lr $rate \
#  --batch_size 32 \
#  --resume_dir None\
#  --output_dir None\
#  --info '12.10 DPG_Mixer with trend and season PeMS-Bay 6'\
#
#done
#
#for rate in 0.001 0.005 0.0001 0.0005 0.00001 0.00005
#do
#python -u main.py \
#  --model_name 'DPG_Mixer'\
#  --exp_name 'deep_learning'\
#  --data_name 'PeMS-Bay' \
#  --pred_len 12\
#  --lr $rate \
#  --batch_size 32 \
#  --resume_dir None\
#  --output_dir None\
#  --info '12.10 DPG_Mixer with trend and season PeMS-Bay 12'\
#
#done



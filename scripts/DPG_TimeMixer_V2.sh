export CUDA_VISIBLE_DEVICES=6

#for rate in 1e-3 1e-4 1e-6 5e-6 1e-7 
#do
#python -u main.py \
#  --model_name 'DPG_Mixer_V2'\
#  --exp_name 'deep_learning'\
#  --data_name 'METR-LA' \
#  --pred_len 3\
#  --lr $rate \
#  --batch_size 32 \
#  --resume_dir None\
#  --output_dir None\
#  --info '1.12 DPG_Mixer_V2 METR-LA '\
#
#done
#
#for rate in 1e-3 1e-4 1e-6 5e-6 1e-7 
#do
#python -u main.py \
#  --model_name 'DPG_Mixer_V2'\
#  --exp_name 'deep_learning'\
#  --data_name 'METR-LA' \
#  --pred_len 6\
#  --lr $rate \
#  --batch_size 32 \
#  --resume_dir None\
#  --output_dir None\
#  --info '1.12 DPG_Mixer_V2 METR-LA '\
#
#done

for rate in 1e-3 1e-4 1e-6 5e-6 1e-7 
do
python -u main.py \
  --model_name 'DPG_Mixer_V2'\
  --exp_name 'deep_learning'\
  --data_name 'METR-LA' \
  --pred_len 12\
  --lr $rate \
  --batch_size 32 \
  --resume_dir None\
  --output_dir None\
  --info '1.12 DPG_Mixer_V2 METR-LA'\

done




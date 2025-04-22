export CUDA_VISIBLE_DEVICES=5





#for rate in     0.0005 0.00001 0.00005
#do
#python -u main.py \
#  --model_name 'STIDGCN'\
#  --exp_name 'deep_learning'\
#  --data_name 'METR-LA' \
#  --seq_len 24\
#  --pred_len 48\
#  --lr $rate \
#  --batch_size 32 \
#  --resume_dir None\
#  --output_dir None\
#  --info '2.12 long-term TESTAM METR-LA 24-24'\
#
#done



for rate in     0.0005 0.00001 0.00005
do
python -u main.py \
  --model_name 'TESTAM'\
  --exp_name 'deep_learning'\
  --data_name 'electricity' \
  --seq_len 96\
  --pred_len 3\
  --lr $rate \
  --batch_size 8 \
  --resume_dir None\
  --output_dir None\
  --info '2.19 debug long-term TESTAM electricity 12-3'\

done



for rate in     0.0005 0.00001 0.00005
do
python -u main.py \
  --model_name 'TESTAM'\
  --exp_name 'deep_learning'\
  --data_name 'electricity' \
  --seq_len 96\
  --pred_len 6\
  --lr $rate \
  --batch_size 8 \
  --resume_dir None\
  --output_dir None\
  --info '2.19 debug long-term TESTAM electricity 12-6'\

done


for rate in     0.0005 0.00001 0.00005
do
python -u main.py \
  --model_name 'TESTAM'\
  --exp_name 'deep_learning'\
  --data_name 'electricity' \
  --seq_len 96\
  --pred_len 12\
  --lr $rate \
  --batch_size 8 \
  --resume_dir None\
  --output_dir None\
  --info '2.19 debug long-term TESTAM electricity 12-12'\

done


#for rate in     0.0005 0.00001 0.00005
#do
#python -u main.py \
#  --model_name 'TESTAM'\
#  --exp_name 'deep_learning'\
#  --data_name 'weather' \
#  --seq_len 12\
#  --pred_len 3\
#  --lr $rate \
#  --batch_size 32 \
#  --resume_dir None\
#  --output_dir None\
#  --info '2.19 debug long-term TESTAM weather 12-3'\
#
#done
#
#
#
#for rate in     0.0005 0.00001 0.00005
#do
#python -u main.py \
#  --model_name 'TESTAM'\
#  --exp_name 'deep_learning'\
#  --data_name 'weather' \
#  --seq_len 12\
#  --pred_len 6\
#  --lr $rate \
#  --batch_size 32 \
#  --resume_dir None\
#  --output_dir None\
#  --info '2.19 debug long-term TESTAM weather 12-6'\
#
#done
#
#
#for rate in     0.0005 0.00001 0.00005
#do
#python -u main.py \
#  --model_name 'TESTAM'\
#  --exp_name 'deep_learning'\
#  --data_name 'weather' \
#  --seq_len 12\
#  --pred_len 12\
#  --lr $rate \
#  --batch_size 32 \
#  --resume_dir None\
#  --output_dir None\
#  --info '2.19 debug long-term TESTAM weather 12-12'\
#
#done








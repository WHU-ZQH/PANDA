export TASK_NAME=superglue
export DATASET_NAME=cb
export CUDA_VISIBLE_DEVICES=0

bs=16
lr=1e-2
dropout=0.1
psl=20
epoch=100

python3 run.py \
  --model_name_or_path bert-large-cased \
  --task_name $TASK_NAME \
  --dataset_name $DATASET_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --pre_seq_len $psl \
  --output_dir p-tuning-v2/checkpoints-original/$DATASET_NAME/ \
  --overwrite_output_dir \
  --hidden_dropout_prob $dropout \
  --seed 11 \
  --save_strategy epoch \
  --save_total_limit 1 \
  --evaluation_strategy epoch \
  --prefix

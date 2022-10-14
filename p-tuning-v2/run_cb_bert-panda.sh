export TASK_NAME=superglue
export DATASET_NAME=cb
export CUDA_VISIBLE_DEVICES=$1

bs=16
lr=7e-3
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
  --output_dir  p-tuning-v2/checkpoints-transfer-our/$2/$DATASET_NAME-bert/ \
  --overwrite_output_dir \
  --source_prompt p-tuning-v2/checkpoints-transfer/$3/$DATASET_NAME-bert/checkpoint/pytorch_model.bin \
  --prompt_transfer 2 \
  --hidden_dropout_prob $dropout \
  --seed 11 \
--save_strategy no \
  --evaluation_strategy epoch \
  --prefix

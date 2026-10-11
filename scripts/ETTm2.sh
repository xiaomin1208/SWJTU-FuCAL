#!/bin/bash
export PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
for v in $(env | grep -o '^FUCAL_[A-Za-z0-9_]*' || true); do unset "$v"; done
export FUCAL_ACI=0
export FUCAL_CONFORMAL_CALIB=0

data_path=ETTm2
seq_len=336
num_nodes=7
num_fuzzy_levels=30
batch_size=16
seed=2024
log_path=./Results/${data_path}/
mkdir -p $log_path

train() {
  out=${log_path}i${seq_len}_o${pred_len}
  extra=()
  [ -n "$anchor_val" ] && extra+=(--anchor_val "$anchor_val")
  python -u train.py \
    --data_path $data_path --seq_len $seq_len --pred_len $pred_len --num_nodes $num_nodes \
    --batch_size $batch_size --seed $seed --epochs 999 \
    --model_name gpt2 --num_fuzzy_levels $num_fuzzy_levels \
    --channel $channel --dropout_n $dropout_n --lr $learning_rate --d_ff $d_ff \
    --stride $stride --patch_len_list $patch_len_list --mae_weight $mae_weight --accum_steps $accum_steps \
    --warmup_max_patience $warmup_patience --joint_max_patience $joint_patience \
    --no_amp --no_llm_analysis \
    --result_dir $out --ckpt_dir $out/checkpoints --cache_dir $out/tensor_cache \
    "${extra[@]}" "$@" 2>&1 | tee ${out}.log
}

pred_len=96
channel=32
dropout_n=0.4
learning_rate=2e-5
d_ff=256
stride=16
patch_len_list="16 32 48 96"
mae_weight=0.6
accum_steps=1
warmup_patience=20
joint_patience=20
anchor_val=
train "$@"

pred_len=192
channel=32
dropout_n=0.4
learning_rate=2e-5
d_ff=256
stride=16
patch_len_list="16 32 48 96"
mae_weight=0.6
accum_steps=1
warmup_patience=10
joint_patience=10
anchor_val=
train "$@"

pred_len=336
channel=32
dropout_n=0.4
learning_rate=2e-5
d_ff=256
stride=16
patch_len_list="16 32 48 96"
mae_weight=0.6
accum_steps=1
warmup_patience=10
joint_patience=10
anchor_val=2.0
train "$@"

pred_len=720
channel=32
dropout_n=0.4
learning_rate=2e-5
d_ff=256
stride=16
patch_len_list="16 32 48 96"
mae_weight=0.6
accum_steps=1
warmup_patience=10
joint_patience=10
anchor_val=-2.0
train "$@"

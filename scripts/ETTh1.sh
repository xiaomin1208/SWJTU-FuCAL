#!/bin/bash
export PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
for v in $(env | grep -o '^FUCAL_[A-Za-z0-9_]*' || true); do unset "$v"; done
export FUCAL_ACI=0
export FUCAL_CONFORMAL_CALIB=0

data_path=ETTh1
seq_len=96
num_fuzzy_levels=15
model_tag=signed5_gpt2_l6_ETTh1_canonical
batch_size=1024
seed=2024
log_path=./Results/${data_path}/
mkdir -p $log_path

train() {
  out=${log_path}i${seq_len}_o${pred_len}
  python -u train.py \
    --data_path $data_path --seq_len $seq_len --pred_len $pred_len --num_nodes 7 \
    --batch_size $batch_size --test_batch_size $batch_size --seed $seed \
    --model_tag $model_tag --num_fuzzy_levels $num_fuzzy_levels \
    --channel $channel --e_layer $e_layer --head $head --dropout_n $dropout_n \
    --lr $learning_rate --stride $stride --mae_weight $mae_weight --alpha $alpha \
    --epochs 999 --d_ff 512 --patch_len_list 8 16 32 48 --no_amp --model_name gpt2 --d_llm 773 \
    --legacy_d_llm 768 --semantic_d_llm 5 --dual_semantic --semantic_encoder llm \
    --semantic_gate_mode auto --semantic_gate_init_bias 0.0 --semantic_residual_clip 0.30 \
    --semantic_interaction_mode prompt --semantic_basis_rank 12 --semantic_lr 0.005 \
    --lambda_base_hinge 0 --lambda_mae_hinge 0 --lambda_router 0 --lambda_gate_l1 0 \
    --lambda_residual 0 --lambda_semantic_alignment 0 --fusion_mod_init zero --ablation full \
    --train_drop_last false --eval_drop_last false --no_llm_analysis --no_cp_plots \
    --num_workers 0 --warmup_max_patience 40 --joint_max_patience 25 --d_layer 2 \
    --accum_steps 1 --weight_decay 0.005 --lambda_coverage 5.0 \
    --lambda_sharp 0.002 --lambda_calib 0.03 --lambda_nll 0.1 --fredf_alpha 0.0 --no_cache \
    --result_dir $out --ckpt_dir $out/checkpoints --cache_dir $out/tensor_cache \
    "$@" 2>&1 | tee ${out}.log
}

pred_len=96
channel=64
e_layer=1
head=2
dropout_n=0.7
learning_rate=5e-05
stride=8
mae_weight=0.8
alpha=0.1
train "$@"

pred_len=192
channel=64
e_layer=1
head=4
dropout_n=0.6
learning_rate=0.0004
stride=8
mae_weight=0.8
alpha=0.1
train "$@"

pred_len=336
channel=64
e_layer=1
head=2
dropout_n=0.6
learning_rate=0.0004
stride=8
mae_weight=0.8
alpha=0.1
train "$@"

pred_len=720
channel=48
e_layer=2
head=4
dropout_n=0.8
learning_rate=0.0003
stride=8
mae_weight=0.9
alpha=0.05
train "$@"

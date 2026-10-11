#!/bin/bash
set -e
export PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
for v in $(env | grep -o '^FUCAL_[A-Za-z0-9_]*' || true); do unset "$v"; done

model_name=${GPT2_PATH:-gpt2}
for divide in train val test; do
  python storage/store_emb.py --data_path ETTm1 --divide $divide --num_nodes 7 \
    --input_len 336 --output_len 96 --num_fuzzy_levels 20 --model_name $model_name --layer 11 \
    --batch_size 512 --out_tag gpt2
done
for divide in train val test; do
  python storage/store_emb.py --data_path ETTm2 --divide $divide --num_nodes 7 \
    --input_len 336 --output_len 96 --num_fuzzy_levels 30 --model_name $model_name --layer 11 \
    --batch_size 512 --out_tag gpt2
done

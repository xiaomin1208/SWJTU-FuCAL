#!/bin/bash
set -e
export PYTHONPATH=.${PYTHONPATH:+:$PYTHONPATH}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
for v in $(env | grep -o '^FUCAL_[A-Za-z0-9_]*' || true); do unset "$v"; done

model_name=${GPT2_PATH:-gpt2}
input_len=96
output_len=96
num_nodes=7

for data_path in ETTh1 ETTh2; do
  if [ "$data_path" = "ETTh1" ]; then num_fuzzy_levels=15; else num_fuzzy_levels=9; fi
  out_tag=signed5_gpt2_l6_${data_path}_canonical
  for divide in train val test; do
    python storage/store_emb.py --data_path $data_path --divide $divide --num_nodes $num_nodes \
      --input_len $input_len --output_len $output_len --num_fuzzy_levels $num_fuzzy_levels \
      --model_name $model_name --layer 6 --out_tag gpt2
    python storage/store_desc.py --data_path $data_path --divide $divide --num_nodes $num_nodes \
      --input_len $input_len --output_len $output_len --num_fuzzy_levels $num_fuzzy_levels
    python storage/store_signed_axes.py --data_path $data_path --divide $divide --num_nodes $num_nodes \
      --input_len $input_len --output_len $output_len --num_fuzzy_levels $num_fuzzy_levels \
      --model_name $model_name --hidden_layer 6 --legacy_tag gpt2 --out_tag $out_tag
  done
done

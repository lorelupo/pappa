#!/bin/bash

# model_name=gpt-3.5-turbo
model_name=text-davinci-003
max_len_model=4097

python main.py \
--data_file data/human_annotation/dim1.csv \
--instruction instructions/pappa_all_dirk.txt \
--output_prompt "\\n\\n" \
--model_name $model_name \
--max_len_model $max_len_model \
--task pappa_dim1_dirk \
--output_dir tmp

# pappa_dim1_reasoned_fewshot pappa_dim1_nodesc_fewshot pappa_dim1_long_fewshot pappa_dim1_long_zeroshot pappa_dim1_reasoned_zeroshot pappa_dim1_nodesc_zeroshot

for instruction in pappa_dim1_reasoned_fewshot pappa_dim1_long_fewshot
do
    python main.py \
    --data_file data/human_annotation/dim1.csv \
    --instruction instructions/$instruction.txt \
    --output_prompt "\\nLabel: " \
    --model_name $model_name \
    --max_len_model $max_len_model \
    --task pappa_dim1 \
    --output_dir tmp
done
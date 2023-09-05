#!/bin/bash

# model_name=text-davinci-003
model_name=gpt-4
max_len_model=4097
# max_len_model=4097

# python main.py \
# --data_file data/human_annotation/dim1.csv \
# --instruction instructions/pappa/all/dirk.txt \
# --output_prompt "\\n\\n" \
# --model_name $model_name \
# --max_len_model $max_len_model \
# --task pappa_dim1_dirk \
# --output_dir tmp

# pappa/alldim/long_fewshot pappa/dim1/long_fewshot_sh1 pappa/dim1/long_fewshot_sh2 pappa/dim1/long_fewshot pappa/dim1/long_zeroshot pappa/dim1/short_fewshot pappa/dim1/short_zeroshot pappa/dim1/nodesc_fewshot pappa/dim1/nodesc_zeroshot
# pappa/dim2/nodesc_fewshot pappa/dim2/nodesc_zeroshot
# pappa/dim3/short_fewshot pappa/dim3/short_zeroshot pappa/dim3/nodesc_fewshot pappa/dim3/nodesc_zeroshot

# for instruction in  pappa/alldim/long_fewshot
# do
#     python main.py \
#     --data_file data/human_annotation/dim1.csv \
#     --instruction instructions/$instruction.txt \
#     --output_prompt "\\nLabel: " \
#     --model_name $model_name \
#     --max_len_model $max_len_model \
#     --task pappa_all_together \
#     --output_dir tmp \
#     # --evaluation_only \
#     # --only_dim 2
# done

for instruction in  pappa/alldim/long_fewshot
do
    python main.py \
    --data_file data/human_annotation/dim3.csv \
    --instruction instructions/$instruction.txt \
    --output_prompt "\\nLabel: " \
    --model_name $model_name \
    --max_len_model $max_len_model \
    --task pappa_dim3 \
    --output_dir tmp \
    --evaluation_only \
    --only_dim 2
done

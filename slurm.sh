#!/bin/bash

# Example of usage: sbatch slurm.sh

#SBATCH --nodes=1                                                                       
#SBATCH --ntasks-per-node=2 # using more than 2 cpus per node will not change much
#SBATCH --gpus-per-node=2                                                      
#SBATCH --time=72:00:00                                    
#SBATCH --error=logs/slurm-%j.log
#SBATCH --output=logs/slurm-%j.log
# SBATCH --account=iscrc_mental
# SBATCH --partition=g100_usr_interactive

# check if logs directory exists, else create it
if [ ! -d "logs" ]; then
    mkdir logs
fi

CONDA_BASE=$(conda info --base)
MYENV=mentalenv

# activate conda environment
source $CONDA_BASE/bin/activate $MYENV

# Note: a bigger batch size will not benefit
CUDA_VISIBLE_DEVICES=3 python main.py \
    --data_file /data/mentalism/data/poverty_tweets_055.pkl \
    --instruction instructions/poverty/short_zeroshot.txt \
    --task_file tasks/poverty/ethnicity.json \
    --prompt_suffix "\\nLabel:" \
    --model_name google/flan-t5-small \
    --max_len_model 512 \
    --output_dir tmp \
    --evaluation_only False
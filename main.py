'''
This code exploits large language models (LLMs) as annotators.
Given an annotation task defined by a set of input texts, a set of possible labels,
and an instruction for the LLM, the code generates a prompt matching the input text and the instruction,
and uses an LLM to generate a label prediction for text. 
The text annotations produced by the LLM are then evaluated against the gold labels using accuracy and the Cohen's kappa metric.

Examples usage:

python llm_classifiers.py --data_file data/m3_eval_gender.tsv --instruction instructions/t5_gender.txt --output_prompt "Genere:" --checkpoint google/flan-t5-small --cache_dir ~/.cache/huggingface/hub/ --task user_gender --output_dir tmp --len_max_model 512
python llm_classifiers.py --data_file data/m3_eval_gender.tsv --instruction instructions/t5_gender.txt --output_prompt "Genere:" --checkpoint google/flan-t5-small --cache_dir $WORK/cache/huggingface/hub/ --task user_gender --output_dir tmp --len_max_model 512
python llm_classifiers.py --data_file data/m3_eval_gender.tsv --instruction instructions/t5_gender_en.txt --output_prompt "Gender:" --checkpoint google/flan-t5-xxl --cache_dir $WORK/cache/huggingface/hub/ --task user_gender --output_dir tmp --len_max_model 512
python llm_classifiers.py --data_file data/m3_eval_gender.tsv --instruction instructions/t5_gender_en_noname.txt --output_prompt "Gender:" --checkpoint google/flan-t5-xxl --cache_dir $WORK/cache/huggingface/hub/ --task user_gender_noname --output_dir tmp --len_max_model 512
'''

import argparse
import datetime
import os
from lm_classifier import LMClassifier
from task_manager import TaskManager

def main():
    
    parser = argparse.ArgumentParser()
    
    # Required parameters
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--data_file', type=str, required=True,
                            help='The input data file. Should contain the .tsv file for the Hate Speech dataset.')
    required_args.add_argument('--task', type=str, required=True,
                            help='The task selected.')
    required_args.add_argument('--instruction', type=str, required=True,
                            help='The instruction for the prompt.')
    required_args.add_argument('--output_prompt', type=str, required=True,
                            help='The instruction for the prompt.')
    required_args.add_argument('--checkpoint', type=str, required=True,
                            help='The model checkpoint.')
    required_args.add_argument('--cache_dir', type=str, required=True,
                            help='Directory with HF models.')
    required_args.add_argument('--len_max_model', type=int, required=True,
                            help='Maximum sequence length of the LLM.')
    required_args.add_argument('--output_dir', type=str, required=True,
                            help='File to write the results.')
    args = parser.parse_args()

    # Define task and load data
    tm = TaskManager(args.task)
    input_texts, gold_labels = tm.read_data(args.data_file)

    # Define LLM classifier
    classifier = LMClassifier(input_texts, tm.labels, gold_labels)

    # Generate predictions
    classifier.generate_predictions(
        instruction=args.instruction,
        output_prompt=args.output_prompt,
        checkpoint=args.checkpoint,
        cache_dir=args.cache_dir,
        len_max_model=args.len_max_model
        )

    # Evaluate predictions
    classifier.evaluate_predictions()  

    # Define output paths
    current_time = datetime.datetime.now().strftime("%d%m%y_%Hh%Mm%S")
    output_dir = f'{args.output_dir}/{args.task}_{args.checkpoint.split("/")[-1]}_{current_time}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save results
    classifier.df.to_csv(output_dir + '/pre.tsv', sep="\t", index=False)
    classifier.df_accuracy.to_csv(output_dir + '/acc.tsv', sep="\t", index=False)
    classifier.df_kappa.to_csv(output_dir + '/kap.tsv', sep="\t", index=False)

if __name__ == "__main__":
    main()
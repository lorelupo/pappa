'''
This code exploits large language models (LLMs) as annotators.
Given an annotation task defined by a set of input texts, a set of possible labels,
and an instruction for the LLM, the code generates a prompt matching the input text and the instruction,
and uses an LLM to generate a label prediction for text. 
The text annotations produced by the LLM are then evaluated against the gold labels using accuracy and the Cohen's kappa metric.

Examples usage:

python main.py --data_file data/human_annotation/dim1.csv --instruction instructions/t5_pappa_dim1.txt --output_prompt "Role of the father:" --checkpoint google/flan-t5-small --cache_dir ~/.cache/huggingface/hub/ --task pappa_dim1 --output_dir tmp --len_max_model 512
python main.py --data_file data/human_annotation/dim1.csv --instruction instructions/t5_pappa_dim1.txt --output_prompt "Role of the father:" --checkpoint google/flan-t5-small --cache_dir /g100_work/IscrC_mental/cache/huggingface/hub/ --task pappa_dim1 --output_dir tmp --len_max_model 512
python main.py --data_file data/human_annotation/dim1.csv --instruction instructions/t5_pappa_dim1_explained.txt --output_prompt "Role of the father:" --checkpoint google/flan-t5-xxl --cache_dir /g100_work/IscrC_mental/cache/huggingface/hub/ --task pappa_dim1 --output_dir tmp --len_max_model 512
task=dim1_binary
python main.py --task pappa_$task --data_file data/human_annotation/$task.csv --instruction instructions/t5_pappa_$task.txt --checkpoint google/flan-ul2 --output_prompt "Role of the father:" --cache_dir /g100_work/IscrC_mental/cache/huggingface/hub/ --output_dir tmp --len_max_model 512
'''

import argparse
from lm_classifier import LMClassifier
from task_manager import TaskManager
from utils import CopyStdoutToFile, incremental_path

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

    # Duplicate the output to stdout and a log file
    output_base_dir = f'{args.output_dir}/{args.task}_{args.checkpoint.split("/")[-1]}'
    output_dir = incremental_path(output_base_dir)
    with CopyStdoutToFile(output_dir + '/log.txt'):

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

        # Save results
        classifier.df.to_csv(output_dir + '/pre.tsv', sep="\t", index=True)
        classifier.df_accuracy.to_csv(output_dir + '/acc.tsv', sep="\t", index=True)
        classifier.df_kappa.to_csv(output_dir + '/kap.tsv', sep="\t", index=True)
        classifier.df_f1.to_csv(output_dir + '/f1.tsv', sep="\t", index=True)

        print('ciao')

if __name__ == "__main__":
    main()
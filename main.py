'''
This code exploits large language models (LLMs) as annotators.
Given an annotation task defined by a set of input texts, a set of possible labels,
and an instruction for the LLM, the code generates a prompt matching the input text and the instruction,
and uses an LLM to generate a label prediction for text. 
The text annotations produced by the LLM are then evaluated against the gold labels using accuracy and the Cohen's kappa metric.

Examples usage on local machine:

python main.py \
    --data_file data/human_annotation/dim1.csv \
    --instruction instructions/pappa_dim1_binary.txt \
    --output_prompt "Role of the father:" \
    --model_name google/flan-t5-small \
    --cache_dir ~/.cache/huggingface/hub/ \
    --task pappa_dim1 \
    --output_dir tmp \
    --len_max_model 512

python main.py \
--data_file data/human_annotation/dim1.csv \
--instruction instructions/pappa_dim1_short_fewshot.txt \
--output_prompt "\\nRole:" \
--model_name gpt \
--task pappa_dim1 \
--output_dir tmp

Examples usage on server:

python main.py \
    --task pappa_dim1 \
    --data_file data/human_annotation/dim1.csv \
    --instruction instructions/pappa_dim1_binary.txt \
    --model_name google/flan-ul2 \
    --output_prompt "Role of the father:" \
    --cache_dir /g100_work/IscrC_mental/cache/huggingface/hub/ \
    --output_dir tmp \
    --len_max_model 512
'''

from utils import CopyStdoutToFile, incremental_path
import argparse
from lm_classifiers import HFClassifier, GPTClassifier
from task_manager import TaskManager

def main():
    
    parser = argparse.ArgumentParser()
    
    # Required parameters
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--data_file', type=str, required=True,
                            help='The input data file. \
                                Should contain the .tsv file \
                                    for the Hate Speech dataset.')
    required_args.add_argument('--task', type=str, required=True,
                            help='The task selected.')
    required_args.add_argument('--instruction', type=str, required=True,
                            help='The instruction for the prompt.')
    required_args.add_argument('--output_prompt', type=str, required=True,
                            help='The instruction for the prompt.')
    required_args.add_argument('--model_name', type=str, required=True,
                            help='The name of the HuggingFace model, \
                            or "GPT" to use the API of OpenAI.')
    required_args.add_argument('--output_dir', type=str, required=True,
                            help='File to write the results.')
    args = parser.parse_args()
    # Optional parameters in case of HuggingFace models
    if args.model_name != 'gpt':
        required_args.add_argument('--cache_dir', type=str, required=True,
                                help='Directory with HF models.')
        required_args.add_argument('--len_max_model', type=int, required=True,
                                help='Maximum sequence length of the LLM.')
    args = parser.parse_args()

    # Duplicate the output to stdout and a log file
    output_base_dir = f'{args.output_dir}/{args.instruction.split("/")[-1].split(".")[0]}_{args.model_name.split("/")[-1]}'
    output_dir = incremental_path(output_base_dir)
    with CopyStdoutToFile(output_dir + '/log.txt'):

        # Define task and load data
        tm = TaskManager(args.task)
        input_texts, gold_labels = tm.read_data(args.data_file)

        # Define classifier
        if args.model_name == 'gpt':
            classifier = GPTClassifier(input_texts, tm.labels, gold_labels)

            # Generate predictions
            classifier.generate_predictions(
                instruction=args.instruction,
                output_prompt=args.output_prompt,
                )

        else:
            classifier = HFClassifier(input_texts, tm.labels, gold_labels)
        
            # Generate predictions
            classifier.generate_predictions(
                instruction=args.instruction,
                output_prompt=args.output_prompt,
                model_name=args.model_name,
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

if __name__ == "__main__":
    main()
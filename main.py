'''
This code exploits large language models (LLMs) as annotators.
Given an annotation task defined by a set of input texts, a set of possible labels,
and an instruction for the LLM, the code generates a prompt matching the input text and the instruction,
and uses an LLM to generate a label prediction for text. 
The text annotations produced by the LLM are then evaluated against the gold labels using accuracy and the Cohen's kappa metric.

Examples usage on local machine:

python main.py \
    --data_file data/human_annotation/dim1_test.csv \
    --instruction instructions/pappa_dim1_binary.txt \
    --output_prompt "Role of the father:" \
    --model_name "google/flan-t5-small" \
    --cache_dir ~/.cache/huggingface/hub/ \
    --task pappa_dim1 \
    --output_dir tmp \
    --max_len_model 512

python main.py \
--data_file data/human_annotation/dim1.csv \
--instruction instructions/pappa_dim1_reasoned_fewshot.txt \
--output_prompt "\\nRole:" \
--model_name gpt-3.5-turbo \
--task pappa_dim1 \
--output_dir tmp

python main.py \
--data_file data/human_annotation/dim1.csv \
--instruction instructions/pappa_dim1_long_fewshot_majority.txt \
--output_prompt "\\nLabel:" \
--model_name text-davinci-003 \
--max_len_model 512 \
--task pappa_dim1 \
--output_dir results \
--evaluation_only

Examples usage on server:

python main.py \
    --task pappa_dim1 \
    --data_file data/human_annotation/dim1.csv \
    --instruction instructions/pappa_dim1_binary.txt \
    --model_name google/flan-ul2 \
    --output_prompt "Role of the father:" \
    --cache_dir /g100_work/IscrC_mental/cache/huggingface/hub/ \
    --output_dir tmp \
    --max_len_model 512
'''

import sys
import os
import argparse
from utils import CopyStdoutToFile, incremental_path
from lm_classifiers import HFClassifier, GPTClassifier
from task_manager import TaskManager
import fire

OPENAI_MODELS = [
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-32k",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k-0613",
    "code-davinci-002",
    "text-davinci-003",
    "text-davinci-002",
    "text-davinci-001",
    "text-davinci",
    "text-curie-003",
    "text-curie-002",
    "text-curie-001",
    "text-curie",
    "davinci-codex",
    "curie-codex",
]

def main(data_file, task, instruction, output_prompt, model_name, max_len_model, output_dir, cache_dir=None, evaluation_only=False, only_dim=None):
    # Duplicate the output to stdout and a log file
    # strip points and slashes from the model name
    model_name_short = model_name.split("/")[-1].replace(".", "")
    instruction_name = instruction.split("/")[-1].split(".")[0]
    output_base_dir = f'{output_dir}/{instruction_name}_{model_name_short}'
    output_dir = incremental_path(output_base_dir) if not evaluation_only else output_base_dir
    print('---'*10)
    print(f'Working on {output_dir}')
    with CopyStdoutToFile(os.path.join(output_dir, 'log.txt')):

        # Print the command used to run the script
        # print(sys.argv[0], "\\")
        # for key, value in locals().items():
        #     print(f"--{key} {value} \\")
        # print("")

        # Define task and load data
        tm = TaskManager(task)
        input_texts, gold_labels = tm.read_data(data_file)

        if model_name in OPENAI_MODELS:
            # Define classifier
            classifier = GPTClassifier(input_texts, tm.labels, tm.label_dims)

            if not evaluation_only:
                # Generate raw predictions
                prompts, predictions = classifier.generate_predictions(
                    instruction=instruction,
                    output_prompt=output_prompt,
                    model_name=model_name,
                    max_len_model=max_len_model,
                    default_label=tm.default_label
                    )
                
                # Save raw predictions
                with open(os.path.join(output_dir, 'raw_predictions.txt'), 'w') as f:
                    for prediction in predictions:
                        f.write(prediction + "\n")
            
            if evaluation_only:
                # Load raw predictions
                with open(os.path.join(output_dir, 'raw_predictions.txt'), 'r') as f:
                    predictions = f.read().splitlines()
                prompts = None
                
            df_predicted_labels = classifier.retrieve_predicted_labels(
                predictions=predictions,
                default_label=tm.default_label,
                prompts=prompts,
                only_dim=only_dim
                )

        else:
            # Define classifier
            classifier = HFClassifier(input_texts, tm.labels)

            if not evaluation_only:
                # Generate predictions
                df_predicted_labels = classifier.generate_predictions(
                    instruction=instruction,
                    output_prompt=output_prompt,
                    model_name=model_name,
                    cache_dir=cache_dir,
                    max_len_model=max_len_model,
                    default_label=tm.default_label
                    )

        # Evaluate predictions
        df_kappa, df_accuracy, df_f1 = classifier.evaluate_predictions(df=df_predicted_labels, gold_labels=gold_labels)

        # Save results
        df_predicted_labels.to_csv(os.path.join(output_dir, f'pre_dim.tsv'), sep="\t", index=True)
        df_kappa.to_csv(os.path.join(output_dir, f'kap_dim.tsv'), sep="\t", index=True)
        df_accuracy.to_csv(os.path.join(output_dir, f'acc_dim.tsv'), sep="\t", index=True)
        df_f1.to_csv(os.path.join(output_dir, f'f1_dim.tsv'), sep="\t", index=True)

        print()

if __name__ == "__main__":
    fire.Fire(main)


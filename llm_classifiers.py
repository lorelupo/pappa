'''
This code exploits large language models (LLMs) as annotators.
Given an annotation task defined by a set of input texts, a set of possible labels,
and an instruction for the LLM, the code generates a prompt matching the input text and the instruction,
and uses an LLM to generate a label prediction for text. 
The text annotations produced by the LLM are then evaluated against the gold labels using accuracy and the Cohen's kappa metric.

Examples usage:

python llm_classifiers.py --data_file data/human_annotation/dim1.csv --instruction instructions/t5_pappa_dim1.txt --output_prompt "Role of the father:" --checkpoint google/flan-t5-small --cache_dir ~/.cache/huggingface/hub/ --task pappa_dim1 --output_file results --len_max_model 512
python llm_classifiers.py --data_file data/human_annotation/dim1.csv --instruction instructions/t5_pappa_dim1.txt --output_prompt "Role of the father:" --checkpoint google/flan-t5-small --cache_dir /g100_work/IscrC_mental/cache/huggingface/hub/ --task pappa_dim1 --output_file results --len_max_model 512
'''

import argparse
import os
import pandas as pd
import collections 
import torch
import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import cohen_kappa_score, accuracy_score


def read_data_pappa(path_data):
    # Read the xlsx data file to table
    df = pd.read_csv(path_data, sep=';')
    # Read text and labels
    input_texts = [text[:-1] + '.' if not text.endswith('.') else text for text in df['text_clean'].tolist()]
    gold_labels = df[['elin', 'lena', 'oscar', 'agg']]
    return input_texts, gold_labels

def generate_predictions_df(
        input_texts,
        instruction,
        output_prompt,
        dict_labels,
        gold_labels,
        checkpoint,
        cache_dir,
        len_max_model,
        ):
    
    # Set device
    # device = torch.device('cuda:1' if torch.cuda.device_count() >= 2 else 'cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'GPU' if torch.cuda.is_available() else 'CPU'
    print(f'Running on {device} device...')

    # Create an empty DataFrame with the desired column names
    df = pd.DataFrame(columns=['id', 'text', 'prompt', 'gold_label', 'prediction'])

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto", cache_dir=cache_dir)
    # Initialize empty lists for predictions and prompts
    predictions = []
    prompts = []

    # Define the instruction and output strings for prompt formulation
    # If instruction is a path to a file, read the file, else use the instruction as is
    instruction = open(instruction, 'r').read() if os.path.isfile(instruction) else instruction
    instruction = instruction.replace('\n', ' ')
    output = output_prompt

    # Encode the labels
    encoded_labels = tokenizer(list(dict_labels.keys()), padding=True, truncation=True, return_tensors="pt")['input_ids']
    print(f'Encoded labels: \n{encoded_labels}')
    # Retrieve the tokens associated to encoded labels and print them
    # decoded_labels = tokenizer.batch_decode(encoded_labels)
    # print(f'Decoded labels: \n{decoded_labels}')
    max_len = max(encoded_labels.shape[1:])
    print(f'Maximum length of the encoded labels: {max_len}')

    time = []
    # Generate predictions and prompts for each input text
    for i, input_text in enumerate(input_texts):
        # Record the start time
        start_time = datetime.datetime.now()

        # Formulate the prompt
        prompt = f'{instruction} {input_text} {output}'

        # Print progress every 100 sentences
        if (i+1) % 100 == 0:
            print(f"Processed {i+1} sentences")

        # Add the prompt to the list of prompts
        prompts.append(prompt)

        # Activate inference mode
        torch.inference_mode(True)
        
        # Encode the prompt using the tokenizer and generate a prediction using the model
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            # If inputs is longer then len_max_model, remove tokens from the encoded instruction
            len_inputs = inputs['input_ids'].shape[1]
            if len_inputs > len_max_model:
                print(f'Input text length: {len_inputs}. Input will be truncated to {len_max_model} tokens.')
                # get the number of tokens to remove from the encoded instruction
                len_remove = len_inputs - len_max_model
                # get the length of the output
                len_output = tokenizer(output, return_tensors="pt")['input_ids'].shape[1] + 1 # +1 for the full stop token
                # remove inputs tokens that come before the output in the encoded prompt
                inputs['input_ids'] = torch.cat((inputs['input_ids'][:,:-len_remove-len_output], inputs['input_ids'][:,-len_output:]),dim=1)
                inputs['attention_mask'] = torch.cat((inputs['attention_mask'][:,:-len_remove-len_output], inputs['attention_mask'][:,-len_output:]),dim=1)
                # Decode inputs and print them
                # decoded_inputs = tokenizer.decode(inputs['input_ids'][0].tolist())
                # print(f'Decoded inputs: \n{decoded_inputs}')
            
            outputs = model.generate(**inputs, max_new_tokens=max_len) # or max_length=inputs['input_ids'].shape[1]+max_len
            outputs_ = tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True)
            predictions.append(outputs_)
        time.append(start_time)

        # Clear the cache after each iteration
        torch.cuda.empty_cache()

    # Deactivate inference mode
    torch.inference_mode(False)

    # Count the number of predictions of each type and print the result
    print(collections.Counter(predictions))

    # Lowercase the predictions
    predictions =  list(map(str.lower,predictions))

    # Map the predictions to the labels (or empty string if label not found)
    predictions = [dict_labels.get(word) for word in predictions]
    
    # Count the number of predictions of each type and print the result
    print(collections.Counter(predictions))

    # Add the data to the DataFrame
    df_out = pd.DataFrame({'time': time, 'prompt': prompts, 'prediction': predictions})
    
    # Add the gold labels to df_out
    if isinstance(gold_labels, pd.DataFrame):
        for col in gold_labels.columns:
            df_out['gold_' + col] = gold_labels[col]    
    elif isinstance(gold_labels, list):
        df_out['gold'] = gold_labels
    else:
        raise ValueError('The gold labels must be either a list or a DataFrame.')

    return df_out

def evaluate_predictions(df_predictions):

    # retrieve columns starting with "gold" and their "names"
    gold_labels = df_predictions.filter(regex='^gold', axis=1)
    gold_names = [col.split('gold_')[-1] for col in gold_labels.columns]
    human_names = [name for name in gold_names if 'agg' not in name]

    # define tables where to store results
    df_kappa = pd.DataFrame(columns=gold_names+['model'], index=gold_names+['model']).fillna(1.0)
    df_accuracy = pd.DataFrame(columns=gold_names+['model'], index=gold_names+['model']).fillna(1.0)


    for i, col in enumerate(gold_labels.columns):
        # compare agreement with gold labels
        kappa = cohen_kappa_score(df_predictions['prediction'].astype(str), gold_labels[col].astype(str))
        accuracy = accuracy_score(df_predictions['prediction'].astype(str), gold_labels[col].astype(str))
        # store results
        df_kappa.loc['model', gold_names[i]] = df_kappa.loc[gold_names[i], 'model'] = kappa
        df_accuracy.loc['model', gold_names[i]] = df_accuracy.loc[gold_names[i], 'model'] = accuracy

        for j, col2 in enumerate(gold_labels.columns):
            if i < j:
                # compare agreement of gold labels with each other
                kappa = cohen_kappa_score(gold_labels[col].astype(str), gold_labels[col2].astype(str))
                accuracy = accuracy_score(gold_labels[col].astype(str), gold_labels[col2].astype(str))
                # store results
                df_kappa.loc[gold_names[i], gold_names[j]] = df_kappa.loc[gold_names[j], gold_names[i]] = kappa
                df_accuracy.loc[gold_names[i], gold_names[j]] = df_accuracy.loc[gold_names[j], gold_names[i]] = accuracy

    # compute average agreement between humans
    df_kappa['mean_human'] = df_kappa[human_names].mean(axis=1)
    df_accuracy['mean_human'] = df_accuracy[human_names].mean(axis=1)
    for name in human_names:
        # correct for humans fully agreeing with themselves
        df_kappa.mean_human[name] = (df_kappa[human_names].loc[name].sum() - 1.0) / (len(human_names) - 1.0)
        df_accuracy.mean_human[name] = (df_accuracy[human_names].loc[name].sum() - 1.0) / (len(human_names) - 1.0)

    print('ACCURACY:')
    print(df_accuracy.round(4)*100)
    print()
    print(f"Humans' mean accuracy: {100*df_accuracy.mean_human[:-1].mean():.2f}")
    print(f"Model's mean accuracy: {100*df_accuracy.model[:-1].mean():.2f}")
    print(f'Diff in mean accuracy: {100*(df_accuracy.mean_human[:-1].mean() - df_accuracy.model[:-1].mean()):.2f}')
    print()
    print('KAPPA:')
    print(df_kappa.round(4)*100)
    print()
    print(f"Humans' mean kappa: {100*df_kappa.mean_human[:-1].mean():.2f}")
    print(f"Model's mean kappa: {100*df_kappa.model[:-1].mean():.2f}")
    print(f'Diff in mean kappa: {100*(df_kappa.mean_human[:-1].mean() - df_kappa.model[:-1].mean()):.2f}')

    return df_accuracy, df_kappa

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
    required_args.add_argument('--output_file', type=str, required=True,
                            help='File to write the results.')
    args = parser.parse_args()

    # Build dictionary of labels {label: label_id}
    dict_pappa_dim1 = {'na': 'NA', 'passive': 'PASSIVE', 'active_negative': 'ACTIVE_NEG', 'active_positive_challenging': 'ACTIVE_POS_CHALLENGING', 'active_positive_caring': 'ACTIVE_POS_CARING', 'active_positive_other': 'ACTIVE_POS_OTHER'}
    dict_pappa_dim1_reduced = {'na': 'NA', 'passive': 'PASSIVE', 'active_negative': 'ACTIVE_NEG', 'active_positive': 'ACTIVE_POS'}
    dict_pappa_dim1_binary = {'na': 'NA', 'passive': 'PAS', 'active': 'POS'}
    dict_pappa_dim2 = {'explicit': 'EXPLICIT', 'implicit': 'IMPLICIT'}
    dict_pappa_dim3 = {'descriptive': 'DESCRIPTIVE', 'ideal': 'IDEAL'}

    # Use a dictionary to select the function to execute and the labels to adopt
    dict_task_func = {'pappa_dim1': read_data_pappa, 'pappa_dim2': read_data_pappa, 'pappa_dim3': read_data_pappa}
    dic_task_labels = {'pappa_dim1': dict_pappa_dim1, 'pappa_dim1_reduced': dict_pappa_dim1_reduced, 'pappa_dim1_binary': dict_pappa_dim1_binary, 'pappa_dim2': dict_pappa_dim2, 'pappa_dim3': dict_pappa_dim3}

    # Read input_text and gold_labels from data_file selecting the right loading function
    input_texts, gold_labels = dict_task_func[args.task](args.data_file)

    # Label the input_texts with the model
    df = generate_predictions_df(
        input_texts=input_texts,
        instruction=args.instruction,
        output_prompt=args.output_prompt,
        dict_labels=dic_task_labels[args.task],
        gold_labels=gold_labels,
        checkpoint=args.checkpoint,
        cache_dir=args.cache_dir,
        len_max_model=args.len_max_model,
        )

    # Evaluate model's labels
    df_accuracy, df_kappa = evaluate_predictions(df)

    # Save to output files
    current_time = datetime.datetime.now().strftime("%d%m_%H%M%S")
    output_file_name = f'{args.output_file}/{args.task}_{args.checkpoint.split("/")[-1]}_{current_time}'
    df.to_csv(output_file_name + '.pre.csv', sep=";", index=False)
    df_accuracy.to_csv(output_file_name + '.acc.csv', sep=";", index=False)
    df_kappa.to_csv(output_file_name + '.kap.csv', sep=";", index=False)

if __name__ == "__main__":
    main()
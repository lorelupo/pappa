import argparse
import os
import pandas as pd
import collections 
import torch
import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

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
        output_file,
        ):

    # Check if cuda is available
    print(f'Running on {torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")} device...')

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
            inputs = tokenizer(prompt, return_tensors="pt")
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

    # Save to output file
    current_time = datetime.datetime.now().strftime("%d%m_%H%M%S")
    output_file_with_time = f'{output_file}/{checkpoint.split("/")[-1]}_{current_time}.tsv'
    df_out.to_csv(output_file_with_time, sep="\t", index=False)
    

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


    generate_predictions_df(
        input_texts=input_texts,
        instruction=args.instruction,
        output_prompt=args.output_prompt,
        dict_labels=dic_task_labels[args.task],
        gold_labels=gold_labels,
        checkpoint=args.checkpoint,
        cache_dir=args.cache_dir,
        len_max_model=args.len_max_model,
        output_file=args.output_file,
        )

if __name__ == "__main__":
    main()

# python example_llms.py \
#     --data_file 'data/human_annotation/ELINsample_for_check_human.xlsx' \
#         --checkpoint 'google/flan-t5-small' \
#             --task 'pappa' \
#                 --instruction  'instructions/t5_pappa_dim1.txt' \
#                     --output_file 'results'
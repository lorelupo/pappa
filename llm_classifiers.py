import argparse
import os
import pandas as pd
import collections 
import torch
import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def read_data_pappa_dim1(path_data):
    # Read the xlsx data file to table
    df = pd.read_excel(path_data, sheet_name='main')
    # Read text and labels
    input_texts = [text[:-1] + '.' if not text.endswith('.') else text for text in df['text_clean'].tolist()]
    gold_labels = [df['dim1'].tolist(), df['dim2'].tolist(), df['dim3'].tolist()]
    return input_texts, gold_labels[0]

def read_data_pappa_dim2(path_data):
    # Read the xlsx data file to table
    df = pd.read_excel(path_data, sheet_name='main')
    # Read text and labels
    input_texts = [text[:-1] + '.' if not text.endswith('.') else text for text in df['text_clean'].tolist()]
    gold_labels = [df['dim1'].tolist(), df['dim2'].tolist(), df['dim3'].tolist()]
    return input_texts, gold_labels[1]

def read_data_pappa_dim3(path_data):
    # Read the xlsx data file to table
    df = pd.read_excel(path_data, sheet_name='main')
    # Read text and labels
    input_texts = [text[:-1] + '.' if not text.endswith('.') else text for text in df['text_clean'].tolist()]
    gold_labels = [df['dim1'].tolist(), df['dim2'].tolist(), df['dim3'].tolist()]
    return input_texts, gold_labels[2]

def generate_predictions_df(input_texts, gold_labels, checkpoint, dict_task, instruction, output_file, len_max_model):

    # Check if cuda is available
    print(f'Running on {torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")} device...')

    # Create an empty DataFrame with the desired column names
    df = pd.DataFrame(columns=['id', 'text', 'prompt', 'gold_label', 'prediction'])

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir="~/.cache/huggingface/transformers")
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto", cache_dir="~/.cache/huggingface/transformers")

    # Initialize empty lists for predictions and prompts
    predictions = []
    prompts = []

    # Define the instruction and output strings for prompt formulation
    # If instruction is a path to a file, read the file, else use the instruction as is
    instruction = open(instruction, 'r').read() if os.path.isfile(instruction) else instruction
    output = "Output:"

    # Encode the labels
    encoded_labels = tokenizer(list(dict_task.keys()), padding=True, truncation=True, return_tensors="pt")['input_ids']
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
                print(f'Input text length: {len_inputs}. Input will be truncated tpo {len_max_model} tokens.')
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
    prediction_counts = collections.Counter(predictions)
    print(prediction_counts)

    # Lowercase the predictions
    predictions =  list(map(str.lower,predictions))

    # Count the number of predictions of each type and print the result
    prediction_counts = collections.Counter(predictions)
    print(prediction_counts)

    # Map the predictions to the labels (or empty string if label not found)
    predictions = [dict_task.get(word) for word in predictions]

    prediction_counts = collections.Counter(predictions)
    print(prediction_counts)

    # Add the data to the DataFrame
    df = pd.DataFrame({'text': input_texts, 'gold_label': gold_labels, 'prediction': predictions, 'prompt': prompts, 'time': time})

    # Save to output file
    current_time = datetime.datetime.now().strftime("%d_%B_%Y_%H_%M_%S")
    output_file_with_time = f'{output_file}/{checkpoint.split("/")[-1]}_{current_time}.tsv'
    df.to_csv(output_file_with_time, sep="\t", index=False)
    

def main():
    
    parser = argparse.ArgumentParser()
    
    # Required parameters
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('--data_file', type=str, required=True,
                            help='The input data file. Should contain the .tsv file for the Hate Speech dataset.')
    required_args.add_argument('--checkpoint', type=str, required=True,
                            help='The model checkpoint.')
    required_args.add_argument('--task', type=str, required=True,
                            help='The task selected.')
    required_args.add_argument('--instruction', type=str, required=True,
                            help='The instruction for the prompt.')
    required_args.add_argument('--output_file', type=str, required=True,
                            help='File to write the results.')
    required_args.add_argument('--len_max_model', type=int, required=True, default=512,
                            help='Maximum sequence length of the LLM.')
    args = parser.parse_args()


    dict_pappa_dim1 = {'na': 'na', 'passive': 'passive', 'active_neg': 'active_neg:', 'active_pos_challenging': 'active_pos_challenging', 'active_pos_caring': 'active_pos_caring', 'active_pos_other': 'active_pos_other'}
    dict_pappa_dim1_reduced = {'na': 'na', 'passive': 'passive', 'active_neg': 'active_neg', 'active_pos': 'active_pos'}
    dict_pappa_dim1_binary = {'na': 'na', 'passive': 'passive', 'active': 'active'}
    dict_pappa_dim2 = {'explicit': 'explicit', 'implicit': 'implicit'}
    dict_pappa_dim3 = {'descriptive': 'descriptive', 'ideal': 'ideal'}
    #dict_hs = {'hate': 1, 'non-hate': 0}

    # Use a dictionary to select the function to execute and the labels to adopt
    dict_task_func = {'pappa_dim1': read_data_pappa_dim1, 'pappa_dim2': read_data_pappa_dim2, 'pappa_dim3': read_data_pappa_dim3}
    dic_task_labels = {'pappa_dim1': dict_pappa_dim1, 'pappa_dim1_reduced': dict_pappa_dim1_reduced, 'pappa_dim1_binary': dict_pappa_dim1_binary}

    # Read input_text and gold_labels from data_file selecting the read_data_pappa_dim1 or read_data_pappa_dim2 or ead_data_pappa_dim3
    input_texts, gold_labels = dict_task_func[args.task](args.data_file)

    generate_predictions_df(
        input_texts, gold_labels,
        args.checkpoint,
        dic_task_labels[args.task],
        args.instruction,
        args.output_file,
        len_max_model=args.len_max_model,
        )

if __name__ == "__main__":
    main()

# python example_llms.py \
#     --data_file 'data/human_annotation/ELINsample_for_check_human.xlsx' \
#         --checkpoint 'google/flan-t5-small' \
#             --task 'pappa' \
#                 --instruction  'instructions/t5_pappa_dim1.txt' \
#                     --output_file 'results'
import argparse
import pandas as pd
import collections 
import torch
import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def read_data_hateval(path_data):
    # Read the data file
    hateval = pd.read_csv(path_data, sep=",")
    input_texts = [text[:-1] + '.' if not text.endswith('.') else text for text in hateval['text'].tolist()]
    return hateval['id'].tolist(), input_texts, hateval['HS'].tolist()

def read_data_topic(path_data):
    # Read the data file
    dbpedia = pd.read_csv(path_data, sep="\t")
    input_texts = [text[:-1] + '.' if not text.endswith('.') else text for text in dbpedia['text'].tolist()]
    return input_texts, dbpedia['label'].tolist()

def read_data_tweet_eval(path_data):
    # Read the data file
    sst2 = pd.read_csv(path_data, sep="\t")
    input_texts = [text[:-1] + '.' if not text.endswith('.') else text for text in sst2['text'].tolist()]
    return input_texts, sst2['label'].tolist()

def read_data_emotion(path_data):
    # Read the data file
    tec = pd.read_csv(path_data, sep="\t")
    input_texts = [text[:-1] + '.' if not text.endswith('.') else text for text in tec['text'].tolist()]
    return input_texts, tec['label'].tolist()

def generate_predictions_df(input_texts, gold_labels, checkpoint, dict_task, instruction, output_file):

    # Check if cuda is available
    print(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    # Create an empty DataFrame with the desired column names
    df = pd.DataFrame(columns=['id', 'text', 'prompt', 'gold_label', 'prediction'])

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir="/sharedData/hf_models/")
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto", cache_dir="/sharedData/hf_models/")

    # Initialize empty lists for predictions and prompts
    predictions = []
    prompts = []

    # Define the instruction and output strings for prompt formulation
    #instruction = "Classify this text as \"hate\" or \"non-hate\"."
    output = "Output:"

    encoded_labels = tokenizer(list(dict_task.keys()), padding=True, truncation=True, return_tensors="pt")['input_ids']
    print(encoded_labels)
    max_len = max(encoded_labels.shape[1:])
    print(max_len)

    time = []
    # Generate predictions and prompts for each input text
    for i, input_text in enumerate(input_texts):
        # Record the start time
        start_time = datetime.datetime.now()
        prompt = f"{instruction} {input_text} {output}"

        # Print progress every 100 messages
        if (i+1) % 1000 == 0:
            print(f"Processed {i+1} messages")

        # Add the prompt to the list of prompts
        prompts.append(prompt)

        # Activate inference mode
        torch.inference_mode(True)
        
        # Encode the prompt using the tokenizer and generate a prediction using the model
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_length=inputs['input_ids'].shape[1]+max_len)
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

    predictions = [dict_task.get(word) for word in predictions]

    prediction_counts = collections.Counter(predictions)
    print(prediction_counts)

    # Add the data to the DataFrame
    df = pd.DataFrame({'text': input_texts, 'gold_label': gold_labels, 'prediction': predictions, 'prompt': prompts, 'time': time})

    df.to_csv(f'{output_file}{"/"}{checkpoint.split("/")[-1]}{".tsv"}', sep="\t", index=False)

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

    args = parser.parse_args()

    input_texts, gold_labels = read_data_tweet_eval(args.data_file)

    dict_hs = {'hate': 1, 'non-hate': 0}
    #dict_tc = {'company': 1, 'educational institution': 2, 'artist': 3, 'athlete': 4, 'office holder': 5, 'mean of transportation': 6, 'building': 7, 'natural place': 8, 'village': 9, 'animal': 10, 'plant': 11, 'album': 12, 'film': 13, 'written work': 14}
    dict_tc = {'technology': 0, 'business': 1, 'sport': 2, 'entertainment': 3, 'politics': 4}
    dict_tweet_eval = {'positive': 1, 'negative': 0, 'neutral': 2}
    dict_emo = {'anger': 'anger', 'fear': 'fear', 'sadness': 'sadness', 'joy': 'joy', 'disgust': 'disgust', 'surprise': 'surprise'}
    dic_task = {'hs': dict_hs, 'tc': dict_tc, 'sent': dict_tweet_eval, 'emo': dict_emo}

    generate_predictions_df(input_texts, gold_labels, args.checkpoint, dic_task[args.task], args.instruction, args.output_file)

if __name__ == "__main__":
    main()
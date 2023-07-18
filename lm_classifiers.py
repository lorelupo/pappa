import openai
import backoff
from dotenv import load_dotenv
import os
import pandas as pd
import collections 
import torch
import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score

import logging
logging.getLogger('backoff').addHandler(logging.StreamHandler())

AVG_TOKENS_PER_WORD_EN = 4/3 # according to: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
AVG_TOKENS_PER_WORD_NONEN = 5 # adjust according to the language of the input text
AVG_TOKENS_PER_WORD_AVG = (AVG_TOKENS_PER_WORD_EN + AVG_TOKENS_PER_WORD_NONEN) / 2

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError), max_tries=5)
def completions_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs) 

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError), max_tries=5)
def chat_completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)  

class LMClassifier:
    def __init__(self, input_texts, labels_dict, label_dims):

        self.input_texts = input_texts
        self.labels_dict = labels_dict

        # check the dimensionality of the labels:
        # dimensionality greater than 1 means dealing with
        # multiple classification tasks at a time
        self.label_dims = label_dims
        assert self.label_dims > 0, "Labels dimensions must be greater than 0."

    def generate_predictions(self):
        raise NotImplementedError
    
    def retrieve_predicted_labels(self, predictions, default_label, prompts, only_dim=None):

        # convert the predictions to lowercase
        predictions =  list(map(str.lower,predictions))

        # retrieve the labels that are contained in the predictions
        predicted_labels = []
        if self.label_dims == 1:
            # retrieve a single label for each prediction since a single classification task is performed at a time
            print("Retrieving predictions...")
            for prediction in predictions:
                labels_in_prediction = [self.labels_dict.get(label) for label in self.labels_dict.keys() if label in prediction]
                predicted_labels.append(labels_in_prediction[0]) if len(labels_in_prediction) > 0 else predicted_labels.append(self.labels_dict.get(default_label))
            # Count the number of predictions of each type and print the result
            print(collections.Counter(predicted_labels))
        else:
            # retrieve multiple labels for each prediction since multiple classification tasks are performed at a time
            print(f"Retrieving predictions for {self.label_dims} dimensions...")
            for prediction in predictions:
                labels_in_prediction = []
                for dim in self.labels_dict.keys():
                    dim_label = []
                    for label in self.labels_dict[dim].keys():
                        if label in prediction:
                            dim_label.append(self.labels_dict[dim].get(label))   
                    dim_label = dim_label[0] if len(dim_label) > 0 else self.labels_dict[dim].get(default_label)
                    labels_in_prediction.append(dim_label)                                            
                predicted_labels.append(labels_in_prediction)
            # Count the number of predictions of each type and print the result
            print(collections.Counter([",".join(labels) for labels in predicted_labels]))
        
        # Add the data to the DataFrame
        if self.label_dims == 1:
            df = pd.DataFrame({'prompt': prompts, 'prediction': predicted_labels})
        elif self.label_dims > 1:
            if only_dim is not None:
                # retrieve only the predictions for a specific dimension
                print(f"Retrieved predictions for dimension {only_dim}")
                df = pd.DataFrame({'prompt': prompts, 'prediction': pd.DataFrame(predicted_labels).to_numpy()[:,only_dim]})
            else:
                print("Retrieved predictions for all dimensions")
                df = pd.DataFrame(predicted_labels).fillna(default_label)
                # rename columns to prediction_n
                df.columns = [f"prediction_dim{i}" for i in range(1, len(df.columns)+1)]
                # add prompts to df
                df['prompt'] = prompts

        return df

    def evaluate_predictions(self, df, gold_labels):

        # Add the gold labels to df
        if isinstance(gold_labels, pd.DataFrame):
            for col in gold_labels.columns:
                df['gold_' + col] = gold_labels[col]    
        elif isinstance(gold_labels, list):
            df['gold'] = gold_labels
        else:
            raise ValueError('The gold labels must be either a list or a DataFrame.')
        
        print("Evaluating predicitons...")
        print(df.head())
        
        # define gold_labels method variable
        gold_labels = df.filter(regex='^gold', axis=1)

        # retrieve the name of each gold annotation
        gold_names = [col.split('gold_')[-1] for col in gold_labels.columns]

        # define tables where to store results
        df_kappa = pd.DataFrame(columns=gold_names+['model'], index=gold_names+['model']).fillna(1.0)
        df_accuracy = pd.DataFrame(columns=gold_names+['model'], index=gold_names+['model']).fillna(1.0)
        df_f1 = pd.DataFrame(columns=gold_names+['model'], index=gold_names+['model']).fillna(1.0)

        for i, col in enumerate(gold_labels.columns):
            # compare agreement with gold labels
            kappa = cohen_kappa_score(df['prediction'].astype(str), gold_labels[col].astype(str))
            accuracy = accuracy_score(df['prediction'].astype(str), gold_labels[col].astype(str))
            f1 = f1_score(df['prediction'].astype(str), gold_labels[col].astype(str), average='macro')
            # store results
            df_kappa.loc['model', gold_names[i]] = df_kappa.loc[gold_names[i], 'model'] = kappa
            df_accuracy.loc['model', gold_names[i]] = df_accuracy.loc[gold_names[i], 'model'] = accuracy
            df_f1.loc['model', gold_names[i]] = df_f1.loc[gold_names[i], 'model'] = f1

            if len(gold_labels.columns) > 1:
                for j, col2 in enumerate(gold_labels.columns):
                    if i < j:
                        # compare agreement of gold labels with each other
                        kappa = cohen_kappa_score(gold_labels[col].astype(str), gold_labels[col2].astype(str))
                        accuracy = accuracy_score(gold_labels[col].astype(str), gold_labels[col2].astype(str))
                        f1 = f1_score(gold_labels[col].astype(str), gold_labels[col2].astype(str), average='macro')
                        # store results
                        df_kappa.loc[gold_names[i], gold_names[j]] = df_kappa.loc[gold_names[j], gold_names[i]] = kappa
                        df_accuracy.loc[gold_names[i], gold_names[j]] = df_accuracy.loc[gold_names[j], gold_names[i]] = accuracy
                        df_f1.loc[gold_names[i], gold_names[j]] = df_f1.loc[gold_names[j], gold_names[i]] = f1

        # in case of multiple gold annotations, there could be a column "gold_agg",
        # referring to the aggregated annotation (computed with tools like MACE)
        non_agg_names = [name for name in gold_names if 'agg' not in name]

        # compute average agreement between gold annotations (except the aggregated one)
        if len(gold_labels.columns) > 1:
            df_kappa['mean_non_agg'] = df_kappa[non_agg_names].mean(axis=1)
            df_accuracy['mean_non_agg'] = df_accuracy[non_agg_names].mean(axis=1) 
            df_f1['mean_non_agg'] = df_f1[non_agg_names].mean(axis=1)
            for name in non_agg_names:
                # correct for humans fully agreeing with themselves
                df_kappa.mean_non_agg[name] = (df_kappa[non_agg_names].loc[name].sum() - 1.0) / (len(non_agg_names) - 1.0)
                df_accuracy.mean_non_agg[name] = (df_accuracy[non_agg_names].loc[name].sum() - 1.0) / (len(non_agg_names) - 1.0)
                df_f1.mean_non_agg[name] = (df_f1[non_agg_names].loc[name].sum() - 1.0) / (len(non_agg_names) - 1.0)
        
        # print info
        print('KAPPA:')
        print(df_kappa.round(4)*100)
        print()
        if len(gold_labels.columns) > 1:
            print(f"Annotators' mean kappa: {100*df_kappa.mean_non_agg[:-1].mean():.2f}")
            print(f"Model's mean kappa: {100*df_kappa.model[:-1].mean():.2f}")
            print(f'Diff in mean kappa: {100*(df_kappa.mean_non_agg[:-1].mean() - df_kappa.model[:-1].mean()):.2f}')
        print()
        print('ACCURACY:')
        print(df_accuracy.round(4)*100)
        print()
        if len(gold_labels.columns) > 1:
            print(f"Annotators' mean accuracy: {100*df_accuracy.mean_non_agg[:-1].mean():.2f}") 
            print(f"Model's mean accuracy: {100*df_accuracy.model[:-1].mean():.2f}")
            print(f'Diff in mean accuracy: {100*(df_accuracy.mean_non_agg[:-1].mean() - df_accuracy.model[:-1].mean()):.2f}')
        print()
        print('F1:')
        print(df_f1.round(4)*100)
        print()
        if len(gold_labels.columns) > 1:
            print(f"Annotators' mean F1: {100*df_f1.mean_non_agg[:-1].mean():.2f}")
            print(f"Model's mean F1: {100*df_f1.model[:-1].mean():.2f}")
            print(f'Diff in mean F1: {100*(df_f1.mean_non_agg[:-1].mean() - df_f1.model[:-1].mean()):.2f}')

        return df_kappa, df_accuracy, df_f1

class GPTClassifier(LMClassifier):
    def __init__(self, input_texts, labels_dict, gold_labels):
        """
        Args:
            input_texts (List[str]): List of input texts to generate predictions for.
            labels_dict (Dict[str, int]): Dictionary mapping label names to label IDs.
            gold_labels (Union[pd.DataFrame, List[str]]): Gold labels corresponding to the input texts.
        """
        super().__init__(input_texts, labels_dict, gold_labels)

        # load environment variables
        load_dotenv('.env')
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def generate_predictions(self, instruction, output_prompt, model_name, max_len_model, default_label):
        """
        Generate predictions for the input texts using the GPT language model.

        Args:
            instruction (str): The instruction text for the LM, or path to a file containing the instruction.
            output_prompt (str): The output prompt to use for generating predictions.

        Returns:
            pd.DataFrame: DataFrame containing the generated predictions and input prompts.
        """

        # Define the instruction and output strings for prompt formulation
        # If instruction is a path to a file, read the file, else use the instruction as is
        instruction = open(instruction, 'r').read() if os.path.isfile(instruction) else instruction
        output = output_prompt.replace('\\n', '\n')
        
        # if prompt is longer then max_len_model, we will remove words from the imput text
        # differently from HF models, where we have access to the tokenizer, here we work on full words
        len_instruction = len(instruction.split())
        len_output = len(output.split())
        max_len_input_text = int((max_len_model - len_instruction*AVG_TOKENS_PER_WORD_EN - len_output*AVG_TOKENS_PER_WORD_EN) / AVG_TOKENS_PER_WORD_AVG)

        prompts = []
        predictions = []
        # Generate predictions and prompts for each input text
        for i, input_text in enumerate(self.input_texts):

            # Formulate the prompt
            prompt = f'{instruction} {input_text} {output}'
            print(prompt) if i == 0 else None
            # if prompt is longer then max_len_model, remove words from the imput text
            len_prompt = int(len(prompt.split())*AVG_TOKENS_PER_WORD_AVG)
            if len_prompt > max_len_model:
                input_text = input_text.split()
                input_text = input_text[:max_len_input_text]
                input_text = ' '.join(input_text)
                prompt = f'{instruction} {input_text} {output}'
                # print detailed info about the above operation
                print(f'Prompt n.{i} was too long, so we removed words from it. Approx original length: {len_prompt}, approx new length: {int(len(prompt.split())*AVG_TOKENS_PER_WORD_AVG)}')

            # Print progress every 100 sentences
            if (i+1) % 20 == 0:
                print(f"Processed {i+1} sentences")

            # Add the prompt to the list of prompts
            prompts.append(prompt)

            # if model name starts with gpt-3, use the following code
            try:
                if model_name.startswith('gpt'):
                    gpt_out = chat_completions_with_backoff(
                        model=model_name,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0,
                        max_tokens=15,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
                    # Extract the predicted label from the output
                    predicted_label = gpt_out['choices'][0]['message']['content'].strip()
                # Generate predictions using the OpenAI API
                else:
                    gpt_out = completions_with_backoff(
                        model=model_name,
                        prompt=prompt,
                        temperature=0,
                        max_tokens=15,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
                    # Extract the predicted label from the output
                    predicted_label = gpt_out['choices'][0]['text'].strip()
            except Exception as e:
                print(f'Error in generating prediction for prompt n.{i}: {e}')
                # select a random label from the list of labels
                predicted_label = default_label
                print(f'Selected default label: {predicted_label}')

            # Add the predicted label to the list of predictionss
            predictions.append(predicted_label)

        # Count the number of predictions of each type and print the result
        print(collections.Counter(predictions))

        return prompts, predictions

class HFClassifier(LMClassifier):

    def __init__(self, input_texts, labels_dict, gold_labels):
        """
        Args:
            input_texts (List[str]): List of input texts to generate predictions for.
            labels_dict (Dict[str, int]): Dictionary mapping label names to label IDs.
            gold_labels (Union[pd.DataFrame, List[str]]): Gold labels corresponding to the input texts.
        """
        super().__init__(input_texts, labels_dict, gold_labels)

    def generate_predictions(self, instruction, output_prompt, model_name, cache_dir, max_len_model, default_label):
        """
        Generate predictions for the input texts using a language model.

        Args:
            instruction (str): The instruction text for the LM, or path to a file containing the instruction.
            output_prompt (str): The output prompt to use for generating predictions.
            model_name (str): Path or identifier of the pre-trained language model model_name.
            cache_dir (str): Directory to cache the language model files.
            max_len_model (int): Maximum length of the language model.

        Returns:
            pd.DataFrame: DataFrame containing the generated predictions and input prompts.
        """
    
        # Set device
        # device = torch.device('cuda:1' if torch.cuda.device_count() >= 2 else 'cuda:0' if torch.cuda.is_available() else 'cpu')
        device = 'GPU' if torch.cuda.is_available() else 'CPU'
        print(f'Running on {device} device...')

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", cache_dir=cache_dir)
        # Initialize empty lists for predictions and prompts
        predictions = []
        prompts = []

        # Define the instruction and output strings for prompt formulation
        # If instruction is a path to a file, read the file, else use the instruction as is
        instruction = open(instruction, 'r').read() if os.path.isfile(instruction) else instruction
        instruction = instruction.replace('\n', ' ')
        output = output_prompt

        # Encode the labels
        encoded_labels = tokenizer(list(self.labels_dict.keys()), padding=True, truncation=True, return_tensors="pt")['input_ids']
        print(f'Encoded labels: \n{encoded_labels}')
        # Retrieve the tokens associated to encoded labels and print them
        # decoded_labels = tokenizer.batch_decode(encoded_labels)
        # print(f'Decoded labels: \n{decoded_labels}')
        max_len = max(encoded_labels.shape[1:])
        print(f'Maximum length of the encoded labels: {max_len}')

        time = []
        # Generate predictions and prompts for each input text
        for i, input_text in enumerate(self.input_texts):
            # Record the start time
            start_time = datetime.datetime.now()

            # Formulate the prompt
            prompt = f'{instruction} {input_text} {output}'
            print(prompt) if i == 0 else None

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
                # If inputs is longer then max_len_model, remove tokens from the encoded instruction
                len_inputs = inputs['input_ids'].shape[1]
                if len_inputs > max_len_model:
                    print(f'Input text length: {len_inputs}. Input will be truncated to {max_len_model} tokens.')
                    # get the number of tokens to remove from the encoded instruction
                    len_remove = len_inputs - max_len_model
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

        # TODO: Map the predictions to the labels (or empty string if label not found)
        # for now, just use the predictions as is. If only a substring equals a label, it does not get mapped to the label.
        predictions = [self.labels_dict.get(word) for word in predictions]
        
        # Count the number of predictions of each type and print the result
        print(collections.Counter(predictions))

        # Add the data to the DataFrame
        df = pd.DataFrame({'time': time, 'prompt': prompts, 'prediction': predictions})

        return df
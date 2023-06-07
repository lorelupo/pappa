import os
import pandas as pd
import collections 
import torch
import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score

class LMClassifier:
    """
    Classifier for generating predictions using a language model and evaluating the predictions.

    Attributes:
        df (pd.DataFrame): DataFrame containing the generated predictions and gold labels.
        df_accuracy (pd.DataFrame): DataFrame containing the accuracy of the predictions.
        df_kappa (pd.DataFrame): DataFrame containing the Cohen's kappa of the predictions.
    """

    def __init__(self, input_texts, dict_labels, gold_labels):
        """
        Args:
            input_texts (List[str]): List of input texts to generate predictions for.
            dict_labels (Dict[str, int]): Dictionary mapping label names to label IDs.
            gold_labels (Union[pd.DataFrame, List[str]]): Gold labels corresponding to the input texts.
        """
                
        self.input_texts = input_texts
        self.dict_labels = dict_labels
        self.gold_labels = gold_labels

    def generate_predictions(self, instruction, output_prompt, checkpoint, cache_dir, len_max_model):
        """
        Generate predictions for the input texts using a language model.

        Args:
            instruction (str): The instruction text for the LM, or path to a file containing the instruction.
            output_prompt (str): The output prompt to use for generating predictions.
            checkpoint (str): Path or identifier of the pre-trained language model checkpoint.
            cache_dir (str): Directory to cache the language model files.
            len_max_model (int): Maximum length of the language model.

        Returns:
            pd.DataFrame: DataFrame containing the generated predictions and input prompts.
        """
    
        # Set device
        # device = torch.device('cuda:1' if torch.cuda.device_count() >= 2 else 'cuda:0' if torch.cuda.is_available() else 'cpu')
        device = 'GPU' if torch.cuda.is_available() else 'CPU'
        print(f'Running on {device} device...')

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
        encoded_labels = tokenizer(list(self.dict_labels.keys()), padding=True, truncation=True, return_tensors="pt")['input_ids']
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
        predictions = [self.dict_labels.get(word) for word in predictions]
        
        # Count the number of predictions of each type and print the result
        print(collections.Counter(predictions))

        # Add the data to the DataFrame
        self.df = pd.DataFrame({'time': time, 'prompt': prompts, 'prediction': predictions})

        return self.df

    def evaluate_predictions(self):
        """
        Evaluate the generated predictions against the gold labels and compute evaluation metrics.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: DataFrames containing the accuracy and kappa values.
        """

        # Add the gold labels to df
        if isinstance(self.gold_labels, pd.DataFrame):
            for col in self.gold_labels.columns:
                self.df['gold_' + col] = self.gold_labels[col]    
        elif isinstance(self.gold_labels, list):
            self.df['gold'] = self.gold_labels
        else:
            raise ValueError('The gold labels must be either a list or a DataFrame.')
        
        # define gold_labels method variable
        gold_labels = self.df.filter(regex='^gold', axis=1)

        # retrieve the name of each gold annotation
        gold_names = [col.split('gold_')[-1] for col in gold_labels.columns]

        # define tables where to store results
        self.df_kappa = pd.DataFrame(columns=gold_names+['model'], index=gold_names+['model']).fillna(1.0)
        self.df_accuracy = pd.DataFrame(columns=gold_names+['model'], index=gold_names+['model']).fillna(1.0)
        self.df_f1 = pd.DataFrame(columns=gold_names+['model'], index=gold_names+['model']).fillna(1.0)


        for i, col in enumerate(gold_labels.columns):
            # compare agreement with gold labels
            kappa = cohen_kappa_score(self.df['prediction'].astype(str), gold_labels[col].astype(str))
            accuracy = accuracy_score(self.df['prediction'].astype(str), gold_labels[col].astype(str))
            f1 = f1_score(self.df['prediction'].astype(str), gold_labels[col].astype(str), average='macro')
            # store results
            self.df_kappa.loc['model', gold_names[i]] = self.df_kappa.loc[gold_names[i], 'model'] = kappa
            self.df_accuracy.loc['model', gold_names[i]] = self.df_accuracy.loc[gold_names[i], 'model'] = accuracy
            self.df_f1.loc['model', gold_names[i]] = self.df_f1.loc[gold_names[i], 'model'] = f1

            if len(gold_labels.columns) > 1:
                for j, col2 in enumerate(gold_labels.columns):
                    if i < j:
                        # compare agreement of gold labels with each other
                        kappa = cohen_kappa_score(gold_labels[col].astype(str), gold_labels[col2].astype(str))
                        accuracy = accuracy_score(gold_labels[col].astype(str), gold_labels[col2].astype(str))
                        f1 = f1_score(gold_labels[col].astype(str), gold_labels[col2].astype(str), average='macro')
                        # store results
                        self.df_kappa.loc[gold_names[i], gold_names[j]] = self.df_kappa.loc[gold_names[j], gold_names[i]] = kappa
                        self.df_accuracy.loc[gold_names[i], gold_names[j]] = self.df_accuracy.loc[gold_names[j], gold_names[i]] = accuracy
                        self.df_f1.loc[gold_names[i], gold_names[j]] = self.df_f1.loc[gold_names[j], gold_names[i]] = f1

        # in case of multiple gold annotations, there could be a column "gold_agg",
        # referring to the aggregated annotation (computed with tools like MACE)
        non_agg_names = [name for name in gold_names if 'agg' not in name]

        # compute average agreement between gold annotations (except the aggregated one)
        if len(gold_labels.columns) > 1:
            self.df_kappa['mean_non_agg'] = self.df_kappa[non_agg_names].mean(axis=1)
            self.df_accuracy['mean_non_agg'] = self.df_accuracy[non_agg_names].mean(axis=1) 
            self.df_f1['mean_non_agg'] = self.df_f1[non_agg_names].mean(axis=1)
            for name in non_agg_names:
                # correct for humans fully agreeing with themselves
                self.df_kappa.mean_non_agg[name] = (self.df_kappa[non_agg_names].loc[name].sum() - 1.0) / (len(non_agg_names) - 1.0)
                self.df_accuracy.mean_non_agg[name] = (self.df_accuracy[non_agg_names].loc[name].sum() - 1.0) / (len(non_agg_names) - 1.0)
                self.df_f1.mean_non_agg[name] = (self.df_f1[non_agg_names].loc[name].sum() - 1.0) / (len(non_agg_names) - 1.0)
        
        # print info
        print('KAPPA:')
        print(self.df_kappa.round(4)*100)
        print()
        if len(gold_labels.columns) > 1:
            print(f"Annotators' mean kappa: {100*self.df_kappa.mean_non_agg[:-1].mean():.2f}")
            print(f"Model's mean kappa: {100*self.df_kappa.model[:-1].mean():.2f}")
            print(f'Diff in mean kappa: {100*(self.df_kappa.mean_non_agg[:-1].mean() - self.df_kappa.model[:-1].mean()):.2f}')
        print()
        print('ACCURACY:')
        print(self.df_accuracy.round(4)*100)
        print()
        if len(gold_labels.columns) > 1:
            print(f"Annotators' mean accuracy: {100*self.df_accuracy.mean_non_agg[:-1].mean():.2f}") 
            print(f"Model's mean accuracy: {100*self.df_accuracy.model[:-1].mean():.2f}")
            print(f'Diff in mean accuracy: {100*(self.df_accuracy.mean_non_agg[:-1].mean() - self.df_accuracy.model[:-1].mean()):.2f}')
        print()
        print('F1:')
        print(self.df_f1.round(4)*100)
        print()
        if len(gold_labels.columns) > 1:
            print(f"Annotators' mean F1: {100*self.df_f1.mean_non_agg[:-1].mean():.2f}")
            print(f"Model's mean F1: {100*self.df_f1.model[:-1].mean():.2f}")
            print(f'Diff in mean F1: {100*(self.df_f1.mean_non_agg[:-1].mean() - self.df_f1.model[:-1].mean()):.2f}')

        return self.df_accuracy, self.df_kappa
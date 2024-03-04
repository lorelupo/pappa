from openai import OpenAI
import openai
import backoff
import os
import time
import re
import pandas as pd
import collections 
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from simple_generation import SimpleGenerator
from utils import setup_logging
from logging import getLogger, StreamHandler
logger = getLogger(__name__)
logger_backoff = getLogger('backoff').addHandler(StreamHandler())

# load environment variables
load_dotenv('.env')
client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig

AVG_TOKENS_PER_EN_WORD = 4/3 # according to: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
AVG_TOKENS_PER_NONEN_WORD = 5 # adjust according to the language of the input text

class LMClassifier:
    def __init__(
            self,
            labels_dict,
            label_dims,
            default_label,
            instruction,
            prompt_suffix,
            model_name,
            max_len_model,
            output_dir=None,
            log_to_file=True
            ):

        setup_logging(os.path.basename(__file__).split('.')[0], logger, output_dir if log_to_file else None)

        self.labels_dict = labels_dict
        # check the dimensionality of the labels:
        # dimensionality greater than 1 means dealing with
        # multiple classification tasks at a time
        self.label_dims = label_dims
        assert self.label_dims > 0, "Labels dimensions must be greater than 0."
        self.default_label = default_label
        
        # Define the instruction and ending ending string for prompt formulation
        # If instruction is a path to a file, read the file, else use the instruction as is
        self.instruction = open(instruction, 'r', encoding='utf-8').read() if os.path.isfile(instruction) else instruction
        self.prompt_suffix = prompt_suffix.replace('\\n', '\n')

        self.max_len_model = max_len_model
        self.model_name = model_name

    def generate_predictions(self):
        raise NotImplementedError

    # adapt this function to the specific model instead of using a generic AVG_TOKENS_PER_EN_WORD
    def set_max_len_input_text(self):
        # set the average number of tokens per word in order to compute the max length of the input text
        self.avg_tokens_per_en_word = AVG_TOKENS_PER_EN_WORD
        self.avg_tokens_per_nonen_word = AVG_TOKENS_PER_NONEN_WORD # TODO infer these numbers automatically for HF models tokenizing a sample of data and comparing #tokens/#words
        self.avg_tokens_per_word_avg = (self.avg_tokens_per_en_word + self.avg_tokens_per_nonen_word) / 2

        # if prompt is longer then max_len_model, we will remove words from the input text
        # differently from HF models, where we have access to the tokenizer, here we work on full words
        len_instruction = len(self.instruction.split())
        len_output = len(self.prompt_suffix.split())
        self.max_len_input_text = int(
            (self.max_len_model - len_instruction*self.avg_tokens_per_en_word - len_output*self.avg_tokens_per_en_word) / self.avg_tokens_per_word_avg
            )
    
    def adapt_prompt_to_max_len_model(self, prompt, input_text, text_id):
        # if prompt is longer then max_len_model, remove words from the imput text
        len_prompt = int(len(prompt.split())*self.avg_tokens_per_word_avg)
        if len_prompt > self.max_len_model:
            # remove words from the input text
            input_text = prompt.split()
            input_text = input_text[:self.max_len_input_text]
            input_text = ' '.join(input_text)
            prompt = f'{self.instruction} {input_text} {self.prompt_suffix}'
            # print detailed info about the above operation
            logger.info(
                f'Prompt {text_id} is too long for the model. '
                f'Approx original length: {len_prompt}; '
                f'Approx new length: {int(len(prompt.split())*self.avg_tokens_per_word_avg)}'
                )
        return prompt
    
    def range_robust_get_label(self, prediction, bounds):
        # more robust get label function that manages numbers in the returned text and assigns them to the correct range in case of number ranges
        # extract all two digit numbers or 0 from the prediction
        numbers = [int(n) for n in re.findall('\d{2}|[0]',prediction)]
        if len(numbers)==0:
            return self.labels_dict.get(self.default_label)
        if len(numbers)>0:
            if (numbers[-1]>bounds[-1][-1]) or (numbers[0]<bounds[0][0]):
                return self.labels_dict.get(self.default_label)
            elif len(numbers)==1:
                # check which list in bounds the number belongs to
                for i, bound in enumerate(bounds):
                    if numbers[0] in bound:
                        return self.labels_dict.get(list(self.labels_dict.keys())[i])
            elif len(numbers)>1:
                # just use the first 2 numbers
                # check the overlap of the range between numbers with bounds
                overlaps = [len(set(range(numbers[0],numbers[1])).intersection(set(bound))) for bound in bounds]
                return self.labels_dict.get(list(self.labels_dict.keys())[overlaps.index(max(overlaps))])


    def retrieve_predicted_labels(self, predictions, prompts=None, only_dim=None):

        # convert the predictions to lowercase
        predictions =  list(map(str.lower,predictions))

        # retrieve the labels that are contained in the predictions
        predicted_labels = []
        if self.label_dims == 1:
            # retrieve a single label for each prediction since a single classification task is performed at a time
            logger.info("Retrieving predictions...")
            for prediction in predictions:
                labels_in_prediction = [self.labels_dict.get(label) for label in self.labels_dict.keys() if label in prediction.split()]
                labels_in_prediction = [label for label in labels_in_prediction if label is not None]  # Filter out None values
                if len(labels_in_prediction) > 0:
                    predicted_labels.append(labels_in_prediction[0])
                else:
                    # first check if there is a range in all the labels
                    bounds = [[int(n) for n in key.split('-') if n.isnumeric()] for key in self.labels_dict.keys()]
                    if all(bounds): # if all labels have a number range
                        bounds = [list(range(b[0],b[1]+1)) for b in bounds]
                        predicted_labels.append(self.range_robust_get_label(prediction,bounds))
                    else:
                        predicted_labels.append(self.labels_dict.get(self.default_label))
            # Count the number of predictions of each type and print the result
            logger.info(collections.Counter(predicted_labels))
        else:
            # retrieve multiple labels for each prediction since multiple classification tasks are performed at a time
            logger.info(f"Retrieving predictions for {self.label_dims} dimensions...")
            for prediction in predictions:
                labels_in_prediction = []
                for dim in self.labels_dict.keys():
                    dim_label = []
                    for label in self.labels_dict[dim].keys():
                        if label in prediction:
                            dim_label.append(self.labels_dict[dim].get(label))   
                    dim_label = dim_label[0] if len(dim_label) > 0 else self.labels_dict[dim].get(self.default_label)
                    labels_in_prediction.append(dim_label)                                            
                predicted_labels.append(labels_in_prediction)
            # Count the number of predictions of each type and print the result
            logger.info(collections.Counter([",".join(str(label) if label is not None else '' for label in labels) for labels in predicted_labels]))

        
        # Add the data to a DataFrame
        if self.label_dims == 1:
            df = pd.DataFrame({'prompt': prompts, 'prediction': predicted_labels}) if prompts else pd.DataFrame({'prediction': predicted_labels})
        elif self.label_dims > 1:
            if only_dim is not None:
                # retrieve only the predictions for a specific dimension
                logger.info(f"Retrieved predictions for dimension {only_dim}")
                df = pd.DataFrame({'prompt': prompts, 'prediction': pd.DataFrame(predicted_labels).to_numpy()[:,only_dim]}) if prompts else pd.DataFrame({'prediction': pd.DataFrame(predicted_labels).to_numpy()[:,only_dim]})
            else:
                logger.info("Retrieved predictions for all dimensions")
                df = pd.DataFrame(predicted_labels).fillna(self.default_label)
                # rename columns to prediction_n
                df.columns = [f"prediction_dim{i}" for i in range(1, len(df.columns)+1)]
                # add prompts to df
                if prompts:
                    df['prompt'] = prompts
        return df


class GPTClassifier(LMClassifier):
    def __init__(
            self,
            labels_dict,
            label_dims,
            default_label,
            instruction,
            prompt_suffix,
            model_name,
            max_len_model,
            gpt_system_role="You are a helpful assistant.",
            **kwargs,
            ):
        super().__init__(labels_dict, label_dims, default_label, instruction, prompt_suffix, model_name, max_len_model, **kwargs)

        self.set_max_len_input_text()

        # define the role of the system in the conversation
        self.system_role = gpt_system_role

    @staticmethod
    @backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError), max_tries=5)
    def completions_with_backoff(**kwargs):
        return client.completions.create(**kwargs) 

    @staticmethod
    @backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError), max_tries=5)
    def chat_completions_with_backoff(**kwargs):
        return client.chat.completions.create(**kwargs)

    def generate_predictions(
            self,
            input_texts,
            sleep_after_step=0,
            ):

        prompts = []
        predictions = []

        # Generate a prompt and a prediction for each input text
        for i, input_text in enumerate(input_texts):
            # Create the prompt
            prompt = f'{self.instruction} {input_text} {self.prompt_suffix}'

            # adapt prompt to max_len_model
            prompt = self.adapt_prompt_to_max_len_model(prompt, input_text, text_id=i)

            # log first prompt
            logger.info(prompt) if i == 0 else None

            # Print progress every 100 sentences
            if (i+1) % 20 == 0:
                logger.info(f"Processed {i+1} sentences")

            # Add the prompt to the list of prompts
            prompts.append(prompt)

            # call OpenAI's API to generate predictions
            try:
                # use chat completion for GPT3.5/4 models
                if self.model_name.startswith('gpt'):
                    gpt_out = self.chat_completions_with_backoff(
                        model=self.model_name,
                        messages=[
                            {"role": "system","content": self.system_role},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0,
                        max_tokens=15,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
                    # Extract the predicted label from the output
                    predicted_label = gpt_out.choices[0].message.content
                    predicted_label = predicted_label.strip().replace('\n', ' ')

                    # Save predicted label to file, together with the index of the prompt
                    with open('raw_predictions_cache.txt', 'a') as f:
                        f.write(f'{i}\t{predicted_label}\n')

                    # Sleep in order to respect OpenAPI's rate limit
                    time.sleep(sleep_after_step)

                # use simple completion for GPT3 models (text-davinci, etc.)
                else:
                    gpt_out = self.completions_with_backoff(
                        model=self.model_name,
                        prompt=prompt,
                        temperature=0,
                        max_tokens=15,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
                    # Extract the predicted label from the output
                    predicted_label = gpt_out.choices[0].message.text.strip()
                    predicted_label = predicted_label.strip().replace('\n', ' ')

            # manage API errors
            except Exception as e:
                logger.error(f'Error in generating prediction for prompt n.{i}: {e}')
                # since the prediction was not generated, use the default label
                predicted_label = self.default_label
                logger.warning(f'Selected default label "{predicted_label}" for prompt n.{i}.')

            # Add the predicted label to the list of predictionss
            predictions.append(predicted_label)

        return prompts, predictions

class HFLMClassifier(LMClassifier):
    def __init__(
            self,
            labels_dict,
            label_dims,
            default_label,
            instruction,
            prompt_suffix,
            model_name,
            max_len_model,
            output_dir=None,
            cache_dir=None,
            **kwargs,
            ):
                
        super().__init__(labels_dict, label_dims, default_label, instruction, prompt_suffix, model_name, max_len_model, output_dir, **kwargs)

        # Set device
        self.device = 'GPU' if torch.cuda.is_available() else 'CPU'
        logger.info(f'Running on {self.device} device...')

        # Load config and inspect whether the model is a seq2seq or causal LM
        config = None
        try:
            config = AutoConfig.from_pretrained(model_name)

            if config.architectures == "LLaMAForCausalLM":
                logger.warning(
                    "We found a deprecated LLaMAForCausalLM architecture in the model's config and updated it to LlamaForCausalLM."
                )
                config.architectures == "LlamaForCausalLM"

            is_encoder_decoder = getattr(config, "is_encoder_decoder", None)
            if is_encoder_decoder == None:
                logger.warning(
                    "Could not find 'is_encoder_decoder' in the model config. Assuming it's an autoregressive model."
                )
                is_encoder_decoder = False

        except:
            logger.warning(
                f"Could not find config in {model_name}. Assuming it's an autoregressive model."
            )
            is_encoder_decoder = False

        self.is_encoder_decoder = is_encoder_decoder

        if is_encoder_decoder:
            model_cls = AutoModelForSeq2SeqLM
        else:
            model_cls = AutoModelForCausalLM

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, config=config, padding_side="left"
        )

        # padding_size="left" is required for autoregressive models, and should not make a difference for every other model as we use attention_masks. See: https://github.com/huggingface/transformers/issues/3021#issuecomment-1454266627 for a discussion on why left padding is needed on batched inference
        self.tokenizer.padding_side = "left"

        logger.debug("Setting off the deprecation warning for padding")
        # see https://github.com/huggingface/transformers/issues/22638
        # and monitor https://github.com/huggingface/transformers/pull/23742
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        
        # Load model
        try:
            self.model = model_cls.from_pretrained(model_name, torch_dtype="auto", device_map="auto", cache_dir=cache_dir)
        except:    
            logger.debug("Removig device_map and trying loading model again")
            self.model = model_cls.from_pretrained(model_name, torch_dtype="auto", cache_dir=cache_dir)

        if not getattr(self.tokenizer, "pad_token", None):
            logger.warning(
                "Couldn't find a PAD token in the tokenizer, using the EOS token instead."
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # try:
        #     self.generation_config = GenerationConfig.from_pretrained(
        #         model_name
        #     )
        # except Exception as e:
        #     logger.warning("Could not load generation config. Using default one.")
        #     self.generation_config = DefaultGenerationConfig()


    def generate_predictions(self, input_texts, remove_prompt_from_output=False):

        # Encode the labels
        encoded_labels = self.tokenizer(list(self.labels_dict.keys()), padding=True, truncation=True, return_tensors="pt")['input_ids']
        logger.info(f'Encoded labels: \n{encoded_labels}')

        # Retrieve the tokens associated to encoded labels and print them
        # decoded_labels = tokenizer.batch_decode(encoded_labels)
        # print(f'Decoded labels: \n{decoded_labels}')
        max_len = max(encoded_labels.shape[1:])
        logger.info(f'Maximum length of the encoded labels: {max_len}')

        predictions = []
        prompts = []

        # Generate a prompt and a prediction for each input text
        for i, input_text in enumerate(input_texts):
            # Create the prompt
            prompt = f'{self.instruction} {input_text} {self.prompt_suffix}'

            # log first prompt
            logger.info(prompt) if i == 0 else None

            # Print progress every 100 sentences
            if (i+1) % 100 == 0:
                logger.info(f"Processed {i+1} sentences")

            # Add the prompt to the list of prompts
            prompts.append(prompt)

            # Activate inference mode
            torch.inference_mode(True)
            
            # Encode the prompt using the tokenizer and generate a prediction using the model
            with torch.no_grad():

                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

                # If inputs is longer then max_len_model, remove tokens from the encoded instruction
                len_inputs = inputs['input_ids'].shape[1]
                if len_inputs > self.max_len_model:
                    # get the number of tokens to remove from the encoded instruction
                    len_remove = len_inputs - self.max_len_model

                    # get the length of the output
                    len_output = self.tokenizer(self.prompt_suffix, return_tensors="pt")['input_ids'].shape[1] + 1 # +1 for the full stop token

                    # remove inputs tokens that come before the output in the encoded prompt
                    inputs['input_ids'] = torch.cat((inputs['input_ids'][:,:-len_remove-len_output], inputs['input_ids'][:,-len_output:]),dim=1)
                    inputs['attention_mask'] = torch.cat((inputs['attention_mask'][:,:-len_remove-len_output], inputs['attention_mask'][:,-len_output:]),dim=1)
                    
                    # print info about the truncation
                    logger.info(f'Original input text length: {len_inputs}. Input has been truncated to {self.max_len_model} tokens.')
                
                # Generate a prediction
                outputs = self.model.generate(**inputs, max_new_tokens=max_len)[0].tolist() # or max_length=inputs['input_ids'].shape[1]+max_len
                if remove_prompt_from_output:                    
                    outputs = outputs[len(inputs["input_ids"][0]) :]
                predicted_label = self.tokenizer.decode(outputs, skip_special_tokens=True)
                predicted_label = predicted_label.strip().replace('\n', ' ')
                # Store it in the list of predictions
                predictions.append(predicted_label)

            # Clear the cache after each iteration
            torch.cuda.empty_cache()

        return prompts, predictions
    

class HFLMClassifier2(LMClassifier):
    def __init__(
            self,
            labels_dict,
            label_dims,
            default_label,
            instruction,
            prompt_suffix,
            model_name,
            max_len_model,
            output_dir=None,
            device_map="auto",
            tokenizer_name=None,
            lora_weights=None,
            compile_model=False,
            use_bettertransformer=False,
            **kwargs
            ):
                
        super().__init__(labels_dict, label_dims, default_label, instruction, prompt_suffix, model_name, max_len_model, output_dir, **kwargs)

        # Login to Hugging Face to be able to load gated models from the hub
        load_dotenv('.env')
        if os.getenv("HF_API_KEY"):
            login(token=os.getenv("HF_API_KEY"))
        

        self.set_max_len_input_text()

        self.generator = SimpleGenerator(
            model_name_or_path=model_name,
            tokenizer_name_or_path=tokenizer_name,
            lora_weights=lora_weights,
            compile_model=compile_model,
            use_bettertransformer=use_bettertransformer,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )


    def generate_predictions(
            self,
            input_texts,
            batch_size="auto",
            starting_batch_size=256,
            num_workers=4,
            skip_prompt=False,
            log_batch_sample=-1,
            show_progress_bar=True,
            apply_chat_template=False,
            add_generation_prompt=False,
            max_new_tokens=1000,
            ):

        """
        # Encode the labels
        encoded_labels = self.generator.tokenizer(list(self.labels_dict.keys()), padding=True, truncation=True, return_tensors="pt")['input_ids']
        logger.info(f'Encoded labels: \n{encoded_labels}')

        # Retrieve the tokens associated to encoded labels and print them
        # decoded_labels = tokenizer.batch_decode(encoded_labels)
        # print(f'Decoded labels: \n{decoded_labels}')
        max_len_output = max(encoded_labels.shape[1:])
        logger.info(f'Maximum length of the encoded labels: {max_len_output}')
        """

        # Create the prompts
        prompts = [
            self.adapt_prompt_to_max_len_model(
                prompt=f'{self.instruction} {input_text} {self.prompt_suffix}',
                input_text=input_text,
                text_id=i
                ) for i, input_text in enumerate(input_texts)
                ]
        
        # Log first prompt
        logger.info(prompts[0])

        # Generate predictions
        predictions = self.generator(
            prompts,
            batch_size=batch_size,
            starting_batch_size=starting_batch_size,
            num_workers=num_workers,
            skip_prompt=skip_prompt,
            log_batch_sample=log_batch_sample,
            show_progress_bar=show_progress_bar,
            apply_chat_template=apply_chat_template,
            add_generation_prompt=add_generation_prompt,
            max_new_tokens=max_new_tokens,
            temperature=0,
            )
        
        predictions = [pred.strip().replace('\n', ' ') for pred in predictions]

        return prompts, predictions
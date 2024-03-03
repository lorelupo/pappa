import os
import fire
import pandas as pd 
from utils import incremental_path, setup_logging
from task_manager import TaskManager
from classifiers import HFLMClassifier2, GPTClassifier, LMClassifier
from evaluate import evaluate_predictions
from logging import getLogger
logger = getLogger(__name__)

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

def annotate_and_evaluate(
        data_file,
        task_file,
        instruction,
        prompt_suffix,
        model_name,
        max_len_model,
        output_dir,
        device_map="auto",
        tokenizer_name=None,
        lora_weights=None,
        compile_model=False,
        use_bettertransformer=False,
        evaluation_only=False,
        only_dim=None,
        gpt_system_role="You are a helpful assistant.",
        sleep_after_step=0,
        aggregated_gold_name="agg",
        log_to_file=True,
        raw_predictions_good=False,
        batch_size="auto",
        starting_batch_size=256,
        num_workers=4,
        skip_prompt=True,
        log_batch_sample=-1,
        show_progress_bar=True,
        apply_chat_template=True,
        add_generation_prompt=False,
        max_new_tokens=1000
        ):

    """
    Params:
        data_file: path to the data file
        task_file: path to the task file
        instruction: path to the instruction file
        prompt_suffix: suffix to add to the prompt
        model_name: name of the model to use (for HuggingFace models, use the full name, e.g. "username/model_name")
        max_len_model: maximum input length of the model
        output_dir: path to the output directory
        evaluation_only: if True, only evaluate the predictions that are already present in the output_dir
        only_dim: if not None, only evaluate the predictions for the given dimension
        gpt_system_role: if model_name is an OpenAI model, this is the role of the system in the conversation
        sleep_after_step: if model_name is an OpenAI model, this is the number of seconds to sleep after each step (might be useful in case of API limits)
        aggregated_gold_name: name of the aggregated gold label, if any
        log_to_file: if True, log to a file in the output_dir
        raw_predictions_good: if True, the raw predictions are already formatted as the final labels and thus don't need to be further processed
    Output:
        raw_predictions.txt: txt file with the raw predictions
        predictions.csv: csv file with the predictions and the probabilities for each class
        *.log: log files with the logs from the predictions process and the evaluation of the predictions
    """

    # Duplicate the output to stdout and a log file
    # strip points and slashes from the model name
    model_name_short = model_name.split("/")[-1].replace(".", "") # remove "username/" in case of HF models
    # if instruction is a path, remove the path and the extension
    if "/" in instruction:
        instruction_name = "/".join(instruction.split("/")[1:]).split(".")[0] # remove "instruction/"" and ".txt" from the instruction path
    else:
        instruction_name = instruction.split(" ")[0]
    output_base_dir = f"{output_dir}/{instruction_name}_{model_name_short}"
    output_dir = incremental_path(output_base_dir, select_last=evaluation_only)

    setup_logging(os.path.basename(__file__).split('.')[0], logger, output_dir if log_to_file else None)

    logger.info(f'Working on {output_dir}')

    # Define task and load data
    tm = TaskManager(task_file)
    logger.info(f'Loading data: {data_file}')
    input_texts, gold_labels = tm.read_data(data_file)

    # Define classifier
    if evaluation_only:
        classifier = LMClassifier(
            labels_dict=tm.labels,
            label_dims=tm.label_dims,
            default_label=tm.default_label,
            instruction=instruction,
            prompt_suffix=prompt_suffix,
            model_name=model_name,
            max_len_model=max_len_model,
            output_dir=output_dir,
            log_to_file=log_to_file,
            )
    elif model_name in OPENAI_MODELS:
            classifier = GPTClassifier(
                labels_dict=tm.labels,
                label_dims=tm.label_dims,
                default_label=tm.default_label,
                instruction=instruction,
                prompt_suffix=prompt_suffix,
                model_name=model_name,
                max_len_model=max_len_model,
                output_dir=output_dir,
                gpt_system_role=gpt_system_role,
                log_to_file=log_to_file,
                )
    else:
        classifier = HFLMClassifier2(
            labels_dict=tm.labels,
            label_dims=tm.label_dims,
            default_label=tm.default_label,
            instruction=instruction,
            prompt_suffix=prompt_suffix,
            model_name=model_name,
            max_len_model=max_len_model, # TODO infer these numbers automatically for HF models though the attribute self.model.config.max_position_embeddings
            output_dir=output_dir,
            device_map=device_map,
            tokenizer_name=tokenizer_name,
            lora_weights=lora_weights,
            compile_model=compile_model,
            use_bettertransformer=use_bettertransformer,
            )

    if evaluation_only:
        logger.info(f'Evaluation only:')
        # Load raw predictions
        logger.info(f'Loading raw predictions from: {os.path.join(output_dir, "raw_predictions.txt")}')
        with open(os.path.join(output_dir, 'raw_predictions.txt'), 'r') as f:
            predictions = f.read().splitlines()
        prompts = None

    else:
        logger.info(f'Generating annotations:')
        # Generate raw predictions
        if model_name in OPENAI_MODELS:
            prompts, predictions = classifier.generate_predictions(input_texts, sleep_after_step=sleep_after_step)
        else:
            prompts, predictions = classifier.generate_predictions(
                            input_texts,
                            batch_size=batch_size,
                            starting_batch_size=starting_batch_size,
                            num_workers=num_workers,
                            skip_prompt=skip_prompt,
                            log_batch_sample=log_batch_sample,
                            show_progress_bar=show_progress_bar,
                            apply_chat_template=apply_chat_template,
                            add_generation_prompt=add_generation_prompt,
                            max_new_tokens=max_new_tokens,
            )

        # Save raw predictions
        with open(os.path.join(output_dir, 'raw_predictions.txt'), 'w') as f:
            for prediction in predictions:
                f.write(prediction + "\n")

    if gold_labels is not None:
        logger.info(f'Gold labels found. Evaluating predictions.')

        # If raw predictions are not yet final labels convert them
        if not raw_predictions_good:
            df_predicted_labels = classifier.retrieve_predicted_labels(
                predictions=predictions,
                prompts=prompts,
                only_dim=only_dim
                )
        else:
            # just create a df from the raw predictions
            df_predicted_labels = pd.DataFrame(predictions, columns=['prediction'])

        # Evaluate predictions
        evaluate_predictions(
            df=df_predicted_labels,
            gold_labels=gold_labels,
            aggregated_gold_name=aggregated_gold_name
            )

        # Save results
        # Save results
        df_predicted_labels.to_csv(os.path.join(output_dir, f'predictions.csv'))
    
    logger.info(f'Done!')

if __name__ == "__main__":
    fire.Fire(annotate_and_evaluate)
# Large Language Models for Text Coding

Annotate your texts using large generative language models (LMs) from Hugging Face or OpenAI's API (GPT), following the methodology described in [How to Use Large Language Models for Text Coding: The Case of Fatherhood Roles in Public Policy Documents](https://arxiv.org/abs/2311.11844).

## Install

```
git clone git@github.com:lorelupo/lm_for_annotation
cd lm_for_annotation
pip install -r ./requirements.txt
```

## Annotation

The file `main.py` runs the annotation of a `--data_file` of texts given a set of possible labels, defined in the `--task_file`, and an `--instruction` for the LM. Supported language models are GPT models by OpenAI, or whatever generative LM hosted on HuggingFace. For an effective annotation, it is also suggested to add a suffix to be appended to the prompt for the LM, consisting of the instruction, the text to be annotated and the suffix.

Example usage with GPT3.5:

```bash
python main.py \
    --data_file data/pappa/human_annotation/dim1.csv \
    --instruction instructions/pappa/dim1/short_zeroshot.txt \
    --task_file tasks/pappa/dim1.json \
    --prompt_suffix "\\nLabel:" \
    --model_name google/flan-t5-small \
    --max_len_model 512 \
    --output_dir tmp
```

Example usage with an open-source LM hosted on Hugging Face:

```bash
python main.py \
    --data_file data/pappa/human_annotation/dim1.csv \
    --instruction instructions/pappa/dim1/long_fewshot.txt \
    --task_file tasks/pappa/dim1.json \
    --prompt_suffix "\\nLabel:" \
    --model_name gpt-3.5-turbo \
    --max_len_model 2048 \
    --output_dir tmp
```

### OpenAI API key

In order to run LMs through the OpenAI's API, an API key is needed. You can create in this folder a file called [.env](./.env) containing your API key:

```
OPENAI_API_KEY = "write-your-key-here"
```

## Evaluation

The annotations produced by the LM can be evaluated against a set of "gold" labels (reference labels) using **Cohen's kappa**, **accuracy**, and **F1** scores. Reference labels should be included in the `--data_file` provided to the `main.py` function, as a column containing the word "gold" in its name. The dataset can also contain multiple reference columns, e.g.:

|ID|text                         |gold_1|gold_2                                    |gold_3        |
|------|-----------------------------|---------|----------------|----------------|
|291392|man kan samtidigt disku- tera om det är lämpligt att surrogatmodern...|PASSIVE  |PASSIVE         |PASSIVE         |
|305276|tabell 0 exempel när avgiften baseras på barnets folkbokföring mammans...|PASSIVE  |NA                                      |NA         |
|328458|att så många av de familjehemsplacerade barnens mödrar är döda...|ACTIVE_POS_OTHER  |PASSIVE         |ACTIVE_POS_OTHER         |

The option `--evaluation_only True` allows to evaluate the annotation by the LM without running it again, e.g.: 

```bash
python main.py \
    --data_file data/pappa/human_annotation/dim1.csv \
    --instruction instructions/pappa/dim1/short_zeroshot.txt \
    --task_file tasks/pappa/dim1.json \
    --prompt_suffix "\\nLabel:" \
    --model_name google/flan-t5-small \
    --max_len_model 512 \
    --output_dir tmp \
    --evaluation_only True
```


Or with generative LMs hosted on HuggingFace, in a zero/few-shot setting:

```
python classification_generative.py \
    --data_file data/user_classification/data_for_models_test.pkl \
    --task_file tasks/gender_classification/bio_tweets.json \
    --instruction instructions/gender_classification/bio_tweets_hf.txt \
    --prompt_suffix \\n\"\"\"\\nGender: \
    --model_name google/flan-t5-xxl \
    --max_len_model 512 \
    --output_dir tmp \
    --cache_dir /data/mentalism/cache/
```

Or with generative LMs by OpenAI, in a zero/few-shot setting:

```
python classification_generative.py \
    --data_file data/user_classification/data_for_models_test.pkl \
    --task_file tasks/gender_classification/bio_tweets.json \
    --instruction instructions/gender_classification/gpt_fewshot_bio_tweets_it.txt \
    --prompt_suffix \\nGender: \
    --model_name gpt-3.5-turbo \
    --max_len_model 2048 \
    --output_dir tmp
```

The  available tasks are:
    
- `gender_classification/`:
    - `bio`: only the users' bio
    - `bio_tweeets`: both the users' bio and tweets
    - `bio_tweeets_int`: : both the users' bio and tweets, when the labels output by the classifier are the integer number of the class (e.g., 0/1 instead of "male"/"female")
- `age_classification`, classifying users' age in 4 groups given the following information as features: 
    - `bio`: only the users' bio
    - `bio_tweeets`: both the users' bio and tweets
    - `bio_tweeets_int`: both the users' bio and tweets, when the labels output by the classifier are the integer number of the class (e.g., 0/1/2/3 instead of "0-19"/"20-29"/"30-39"/"40-100")

Check the folder [instructions](instructions) to see available instructions for generative LMs and add new ones.

## TODO Defining instructions and tasks

It is possible to define new classification tasks by creating a new `.json` file describing the dictionary of labels and the data-reading function. See the age classification task defined in [tasks/age_classification/bio_tweets.json](tasks/age_classification/bio_tweets.json) as an example:

```json
{
    "labels": {
        "0-19": "0",
        "20-29": "1",
        "30-39": "2",
        "40-100": "3"
        },
    "read_function": "twitter_features_age_interval_bio_tweets"
}
```

In the labels dictionary, the keys are the labels in the format output by the classifier, while the values are the labels as represented in your data. In this case, a classifier outputs an integer referring to the age group of the Twitter user.

The data-reading function needs to be defined in the [task_manager.py](task_manager.py) as a static method. For instance, see the definition of [twitter_features_age_interval_bio_tweets](twitter_features_age_interval_bio_tweets.py?plain=1#L105), a utility function that reads [data/user_classification/data_for_models_test.pkl](data/user_classification/data_for_models_test.pkl) and creates a string containing both users' bio and tweets as a feature for the classifier.

## Citation

```
@misc{lupo2023use,
      title={How to Use Large Language Models for Text Coding: The Case of Fatherhood Roles in Public Policy Documents}, 
      author={Lorenzo Lupo and Oscar Magnusson and Dirk Hovy and Elin Naurin and Lena Wängnerud},
      year={2023},
      eprint={2311.11844},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
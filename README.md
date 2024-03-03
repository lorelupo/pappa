# Large Language Models for Text Coding

Annotate your texts using large generative language models (LMs) from [Hugging Face](https://huggingface.co/models) or [OpenAI's API](https://openai.com/blog/openai-api) (GPT).

The methodology is described in [How to Use Large Language Models for Text Coding: The Case of Fatherhood Roles in Public Policy Documents](https://arxiv.org/abs/2311.11844). The data used in the paper are available in the folder [data/pappa](data/pappa).



## Install

Open a terminal and launch the following commands:
```
git clone git@github.com:lorelupo/pappa
cd pappa
pip install -r ./requirements.txt
```

## Annotation

Example usage with GPT3.5:

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
```command terminal
python main.py --data_file data/pappa/human_annotation/dim1.csv --instruction instructions/pappa/dim1/long_fewshot.txt --task_file tasks/pappa/dim1.json --prompt_suffix "\\nLabel:" --model_name gpt-4-0125-preview --max_len_model 2048 --output_dir tmp --evaluation_only False
```

Example usage with an open-source LM hosted on Hugging Face:

```bash
python main.py \
    --data_file data/pappa/human_annotation/dim1.csv \
    --instruction instructions/pappa/dim1/short_zeroshot.txt \
    --task_file tasks/pappa/single/dim1.json \
    --prompt_suffix "\\nLabel:" \
    --model_name google/flan-t5-small \
    --max_len_model 512 \
    --output_dir tmp
```

### Data file

The supported data file formats are `.xlsx` (Excel), semicolon-separated `.csv`, and `.pkl`. The texts to be annotated should be listed under a column named `text`. 

The datafile fills two purposes: It supplies the chosen model with texts to be annotated, and it provides it with gold_labels for evaluation purposes. 

### Prompt

When adequately prompted, a LM can annotate a text according to a given labelling scheme.
In our framework, the prompt consists of three elements: an $\textcolor{blue}{\textsf{instruction}}$ (which can also contain an explanation of the available labels and some examples), a $\textcolor{red}{\textsf{text}}$ to be annotated, and a $\textcolor{green}{\textsf{suffix}}$, to be appended after the text. For example:

---
$\textcolor{blue}{\textsf{
Label the Swedish text according to how it describes the role of the father in the family.
Possible labels are:
}}$

- $\textcolor{blue}{\textsf{passive: fathers who are not actively involved in hands-on care and upbringing of the child;}}$
- $\textcolor{blue}{\textsf{active\\_negative: fathers exhibiting harmful behaviours like aggression, violence, or neglect;}}$
- $\textcolor{blue}{\textsf{active\\_positive\\_caring: fathers providing care, warmth, empathy, and support;}}$
- $\textcolor{blue}{\textsf{active\\_positive\\_challenging: fathers encouraging risk-taking, growth, and educational activities;}}$
- $\textcolor{blue}{\textsf{active\\_positive\\_other: fathers displaying competence, responsibility, trustworthiness, etc.,}}$
      $\textcolor{blue}{\textsf{without specifying a specific role;}}$
- $\textcolor{blue}{\textsf{not\\_applicable: not applicable.}}$



$\textcolor{blue}{\textsf{Text:}}$ $\textcolor{red}{\textsf{i båda fallen är modern genetisk mor till barnet .}}$

$\textcolor{green}{\textsf{
Label:
}}$

---

Instructions can be given to the LM by passing a `.txt` file to the `--instruction` argument of `main.py`.
The `--suffix` should be changed according to the instruction.

### Task

A classification task is defined by a `.json` file describing the dictionary of labels and the data-reading function. For instance, see the task "dimension 1" defined in [tasks/pappa/dim1.json](tasks/pappa/dim1.json):

```json
{
    "labels": {
        "not_applicable": "NA",
        "passive": "PASSIVE",
        "active_negative": "ACTIVE_NEG",
        "active_positive_challenging": "ACTIVE_POS_CHALLENGING",
        "active_positive_caring": "ACTIVE_POS_CARING",
        "active_positive_other": "ACTIVE_POS_OTHER"
        },
    "default_label": "not_applicable",
    "read_function": "read_data"
}
```

In the labels dictionary, the keys are the labels in the format required to the LM, while the values are the labels as represented in your data. We make this distinction because different LMs might work more effectively with  different labels format. Defining the task in this way allows to flexibly tests different labels format by simply changing the dictionary keys and without modifying the dataset.

The data-reading function needs to be defined in the [task_manager.py](task_manager.py). The current, basic data-reading function takes in input a tabular file where the texts to be annotated are under the column `text`, and the reference labels under one or more columns (if more than a reference label per text is available) containing the word `gold` in their name.

### Models

Supported language models are all generative models that are downloadable from [Hugging Face](https://huggingface.co/models) (e.g., [google/flan-t5-large](https://huggingface.co/google/flan-t5-large)) and the following OpenAI models: 

```python

OPENAI_MODELS = [
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-turbo-preview",
    "gpt-4-1106-preview",
    "gpt-4-0125-preview",
    "gpt-4-32k",
    "gpt-4-32k-0613",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
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
]*
````
\* this list can be easily expanded to new OpenAI models by simply adding them to it in the `main.py` file.

### OpenAI API key

In order to run LMs through the OpenAI's API, an API key is needed. You can create in this folder a file called [.env](./.env) containing your API key:

```
OPENAI_API_KEY = "write-your-key-here"
```

## Evaluation

The annotations produced by the LM can be evaluated against a set of "gold" labels (reference labels) using **Cohen's kappa**, **accuracy**, and **F1** scores. Reference labels should be included in the `--data_file` provided to the `main.py` function, under one or more columns (if more than a reference label per text is available) containing the word `gold` in their name, e.g.:

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
    --task_file tasks/pappa/single/dim1.json \
    --prompt_suffix "\\nLabel:" \
    --model_name google/flan-t5-small \
    --max_len_model 512 \
    --output_dir tmp \
    --evaluation_only True
```

Note: When choosing --evaluation_only True, the evaluation will be run against the latest saved tmp-file matching the command. For example, if you have [text](tmp/pappa/alldim/long_fewshot_gpt-35-turbo-0613_4) but want to evaluate an earlier instance [text](tmp/pappa/alldim/long_fewshot_gpt-35-turbo-0613_3), you ought to temporarily rename one of the files to allow for the tmp file to be evaluated to have the highest number. 

## Evaluation multiple classification

The current setup does not allow for evaluating several classification dimensions at one time. Instead, the dimensions are evaluated one at a time. The option 'eval_dim' enables the selection of the dimension to be evaluated. When running multiple classification annotation/validation, make sure that the task file is placed in the correct "multi" subfolder. For example:  

```bash
python main.py \
    --data_file data/pappa/human_annotation/dim1.csv \
    --instruction instructions/pappa/alldim/long_fewshot.txt \
    --task_file tasks/pappa/multi/all.json \
    --prompt_suffix "\\nLabel:" \
    --model_name gpt-4-0125-preview \
    --max_len_model 512 \
    --output_dir tmp \
    --evaluation_only False \
    --eval_dim dim1
```

The above example will generate annotations for all dimensions but only evaluate the chosen dimension. If you want to evaluate the another dimension, make sure to change "evaluation only" to True, "data_file" (for correct gold_labels) and "eval_dim" (for corresponding annotations) but not the "task_file". For example: 

```bash
python main.py \
    --data_file data/pappa/human_annotation/dim2.csv \
    --instruction instructions/pappa/alldim/long_fewshot.txt \
    --task_file tasks/pappa/multi/all.json \
    --prompt_suffix "\\nLabel:" \
    --model_name gpt-4-0125-preview \
    --max_len_model 512 \
    --output_dir tmp \
    --evaluation_only True \
    --eval_dim dim2
```
Each time the script is run for a new dimension, a new confusion matrix and evaluation.log for the specified dimension will be saved. 

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

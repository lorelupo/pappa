# Large Language Models for Text Coding

Annotate your texts using large generative language models (LMs) from Hugging Face or OpenAI's API (GPT), following the methodology described in [How to Use Large Language Models for Text Coding: The Case of Fatherhood Roles in Public Policy Documents](https://arxiv.org/abs/2311.11844).

## Install

```
git clone git@github.com:lorelupo/lm_for_annotation
cd lm_for_annotation
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

Example usage with an open-source LM hosted on Hugging Face:

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

### Data file

The supported data file formats are `.xlsx` (Excel), semicolon-separated `.csv`, and `.pkl`. The texts to be annotated should be listed under a column named `text`. 

### Prompt

When adequately prompted, a LM can annotate a text according to a given labelling scheme.
In our framework, the prompt consists of three elements: an instruction</span> (which can also contain an explanation of the available labels and some examples), a <span style="color:red">text</span> to be annotated, and a <span style="color:green">suffix</span>, to be appended after the text. For example:

---

Label the Swedish text according to how it describes the role of the father in the family.
Possible labels are:

$\textcolor{blue}{\textsf{
- passive: fathers who are not actively involved in hands-on care and upbringing of the child;
- active_negative: fathers exhibiting harmful behaviours like aggression, violence, or neglect;
- active_positive_caring: fathers providing care, warmth, empathy, and support;
- active_positive_challenging: fathers encouraging risk-taking, growth, and educational activities;
- active_positive_other: fathers displaying competence, responsibility, trustworthiness, etc., without specifying a specific role;
- not_applicable: not applicable.
}}$


<span style="color:red">
Text: i båda fallen är modern genetisk mor till barnet .
</span>

<span style="color:green">
Label:
</span>

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
    --task_file tasks/pappa/dim1.json \
    --prompt_suffix "\\nLabel:" \
    --model_name google/flan-t5-small \
    --max_len_model 512 \
    --output_dir tmp \
    --evaluation_only True
```

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
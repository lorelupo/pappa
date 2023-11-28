import json
import pandas as pd

def read_data_base(path_data):
    """
    Parameters
    ----------
    path_data : str
        Path to the data file (.csv, .tsv, .xlsx, .pkl, .json)
    
    Returns
    -------
    input_texts : list
        List of input texts
    gold_labels : pd.DataFrame
        DataFrame of gold labels
    """

    # Read the csv/xlsx data file to table
    if path_data.endswith('.csv'):
        df = pd.read_csv(path_data, sep=';').fillna('NA')
    elif path_data.endswith('.xlsx'):
        df = pd.read_excel(path_data).fillna('NA')
    elif path_data.endswith('.pkl'):
        df = pd.read_pickle(path_data)
    elif path_data.endswith('.json'):
        df = pd.read_json(path_data)
    else:
        raise ValueError(f'File format not supported: {path_data}')
    
    # Read texts and add a period at the end if missing
    input_texts = df['text'].tolist()

    # Read gold labels if any (all columns starting with 'gold_')
    gold_labels = df.filter(regex='^gold', axis=1)
    
    return input_texts, gold_labels


# define dictionary of data-reading functions
DATA_READING_FUNCTIONS = {
    "read_data": read_data_base,
}

class TaskManager:
    """
    Utility class for reading data tables (.csv, .tsv, ...) and returning the input texts and gold labels
    for an LLM classifier.
    """

    def __init__(self, task_file):

        # read task specs from json task_file
        self.task = json.load(open(task_file, 'r'))
        # setup labels
        self.labels = self.task['labels']
        self.label_dims = self.task['label_dims'] if 'label_dims' in self.task else 1
        self.default_label = self.task['default_label'] if 'default_label' in self.task else list(self.labels.keys())[0] if self.label_dims == 1 else list(self.labels['dim1'].keys())[0]

        # select data reading function
        if self.task['read_function'] in DATA_READING_FUNCTIONS:
            self.read_data = DATA_READING_FUNCTIONS[self.task['read_function']]
        else:
            raise ValueError(
                f"Data-reading function '{self.task['read_function']}' not supported."
                f"Supported functions: {self.data_reading_functions.keys()}"
                f"You can implement your own data-reading function in task_manager.py."
                )
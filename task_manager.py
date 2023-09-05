import json
import pandas as pd

class TaskManager:
    """
    Utility class for reading data tables (.csv, .tsv, ...) and returning the input texts and gold labels
    for an LLM classifier.
    """

    def __init__(self, task_file):

        # define dictionary of data-reading functions
        self.data_reading_functions = {
            "pappa": self.read_data_pappa,
        }

        # read task specs from json task_file
        self.task = json.load(open(task_file, 'r'))
        # setup labels
        self.labels = self.task['labels']
        self.label_dims = self.task['label_dims'] if 'label_dims' in self.task else 1
        self.default_label = list(self.labels.keys())[0] if self.label_dims == 1 else list(self.labels['dim1'].keys())[0]
        # setup data reading function
        if self.task['read_function'] in self.data_reading_functions:
            self.read_data = self.data_reading_functions[self.task['read_function']]
        else:
            raise ValueError(
                f"Data-reading function '{self.task['read_function']}' not supported."
                f"Supported functions: {self.data_reading_functions.keys()}"
                )

    @staticmethod
    def read_data_pappa(path_data):
        """
        Read data from a file and extract input texts and gold labels for the 'user_gender_noname' task.

        Args:
            path_data (str): Path to the data file.

        Returns:
            Tuple: Tuple containing input texts (List[str]) and gold labels (Pandas DataFrame).
        """
        # Read the xlsx data file to table
        df = pd.read_csv(path_data, sep=';').fillna('NA')
        # Read text and labels
        input_texts = [text[:-1] + '.' if not text.endswith('.') else text for text in df['text_clean'].tolist()]
        gold_labels = df[['elin', 'lena', 'oscar', 'agg']]
        return input_texts, gold_labels
import pandas as pd


class TaskManager:
    """
    Utility class for reading data tables (.csv, .tsv, ...) and returning the input texts and gold labels
    for an LLM classifier.
    """

    def __init__(self, task):
 
        self.task_dict = {
            'pappa_dim1': {
                'labels': {
                    'not_applicable': 'NA',
                    'passive': 'PASSIVE',
                    'active_negative': 'ACTIVE_NEG',
                    'active_positive_challenging': 'ACTIVE_POS_CHALLENGING',
                    'active_positive_caring': 'ACTIVE_POS_CARING',
                    'active_positive_other': 'ACTIVE_POS_OTHER'
                    },
                'read_function': self.read_data_pappa,
            },
            'pappa_dim1_reduced': {
                'labels': {'not_applicable': 'NA',
                           'passive': 'PASSIVE',
                           'active_negative': 'ACTIVE_NEG',
                           'active_positive': 'ACTIVE_POS'},
                'read_function': self.read_data_pappa,
            },
            'pappa_dim1_binary': {
                'labels': {
                    'not_applicable': 'NA',
                    'passive': 'PAS',
                    'active': 'POS'
                    },
                'read_function': self.read_data_pappa,
            },
            'pappa_dim2': {
                'labels': {
                    'explicit': 'EXPLICIT',
                    'implicit': 'IMPLICIT'
                    },
                'read_function': self.read_data_pappa,
            },
            'pappa_dim3': {
                'labels': {
                    'descriptive': 'DESCRIPTIVE',
                    'ideal': 'IDEAL'
                    },
                'read_function': self.read_data_pappa,
            },             
        }
        self.task = self.task_dict[task]
        self.labels = self.task['labels']
        self.read_data = self.task['read_function']

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
        df = pd.read_csv(path_data, sep=';')
        # Read text and labels
        input_texts = [text[:-1] + '.' if not text.endswith('.') else text for text in df['text_clean'].tolist()]
        gold_labels = df[['elin', 'lena', 'oscar', 'agg']]
        # Rename columns adding a prefix
        gold_labels = gold_labels.add_prefix('gold_')
        return input_texts, gold_labels
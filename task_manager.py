import pandas as pd


class TaskManager:
    """
    Utility class for reading data tables (.csv, .tsv, ...) and returning the input texts and gold labels
    for an LLM classifier.
    """

    def __init__(self, task):
 
        self.task_dict = {
            'pappa_all': {
                'label_dims': 3,
                'labels': {
                    'dim1': {
                        'not_applicable': 'NA',
                        'passive': 'PASSIVE',
                        'active_negative': 'ACTIVE_NEG',
                        'active_positive_challenging': 'ACTIVE_POS_CHALLENGING',
                        'active_positive_caring': 'ACTIVE_POS_CARING',
                        'active_positive_other': 'ACTIVE_POS_OTHER',
                    },
                    'dim2': {
                        'not_applicable': 'NA',
                        'explicit': 'EXPLICIT',
                        'implicit': 'IMPLICIT',
                    },
                    'dim3': {
                        'not_applicable': 'NA',
                        'descriptive': 'DESCRIPTIVE',
                        'ideal': 'IDEAL'
                    },
                },
                'read_function': self.read_data_pappa,
            },
            'pappa_all_together': {
                'label_dims': 1,
                'labels': {
                    'not_applicable': 'NA',
                    'passive': 'PASSIVE',
                    'active_negative': 'ACTIVE_NEG',
                    'active_positive_challenging': 'ACTIVE_POS_CHALLENGING',
                    'active_positive_caring': 'ACTIVE_POS_CARING',
                    'active_positive_other': 'ACTIVE_POS_OTHER',
                    'explicit': 'EXPLICIT',
                    'implicit': 'IMPLICIT',
                    'descriptive': 'DESCRIPTIVE',
                    'ideal': 'IDEAL'
                    },
                'read_function': self.read_data_pappa,
            },
            'pappa_dim1': {
                'label_dims': 1,
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
                'label_dims': 1,
                'labels': {'not_applicable': 'NA',
                           'passive': 'PASSIVE',
                           'active_negative': 'ACTIVE_NEG',
                           'active_positive': 'ACTIVE_POS'},
                'read_function': self.read_data_pappa,
            },
            'pappa_dim1_binary': {
                'label_dims': 1,
                'labels': {
                    'not_applicable': 'NA',
                    'passive': 'PAS',
                    'active': 'POS'
                    },
                'read_function': self.read_data_pappa,
            },
            'pappa_dim1_dirk': {
                'label_dims': 1,
                'labels': {
                    'NA': 'NA',
                    'PASSIVE': 'PASSIVE',
                    'ACTIVE_NEG': 'ACTIVE_NEG',
                    'ACTIVE_POS_CHALLENGING': 'ACTIVE_POS_CHALLENGING',
                    'ACTIVE_POS_CARING': 'ACTIVE_POS_CARING',
                    'ACTIVE_POS_OTHER': 'ACTIVE_POS_OTHER'
                    },
                'read_function': self.read_data_pappa,
            },
            'pappa_dim2': {
                'label_dims': 1,
                'labels': {
                    'explicit': 'EXPLICIT',
                    'implicit': 'IMPLICIT'
                    },
                'read_function': self.read_data_pappa,
            },
            'pappa_dim3': {
                'label_dims': 1,
                'labels': {
                    'descriptive': 'DESCRIPTIVE',
                    'ideal': 'IDEAL'
                    },
                'read_function': self.read_data_pappa,
            },             
        }
        self.task = self.task_dict[task]
        self.label_dims = self.task['label_dims']
        self.labels = self.task['labels']
        # select the first label in the dictionary as default
        self.default_label = list(self.labels.keys())[0] if self.label_dims == 1 else list(self.labels['dim1'].keys())[0]
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
        df = pd.read_csv(path_data, sep=';').fillna('NA')
        # Read text and labels
        input_texts = [text[:-1] + '.' if not text.endswith('.') else text for text in df['text_clean'].tolist()]
        gold_labels = df[['elin', 'lena', 'oscar', 'agg']]
        return input_texts, gold_labels
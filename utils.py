import traceback
import sys
import os
import logging

def incremental_path(path):
    """
    Create a directory or file with an incremental number if a directory or file with the same name already exists.
    """

    # Define the base path and extension
    base_path, extension = os.path.splitext(path)

    # Check if extension is empty
    is_file = (extension != '')

    # Add an incremental number if a directory or file with the same name already exists
    increment = 1
    while os.path.exists(path):
        path = f'{base_path}_{increment}{extension}'
        increment += 1

    # Create the directory or file
    if is_file:
        with open(path, 'w'):
            pass
    else:
        os.makedirs(path)

    return path

def setup_logging(module_name:str, logdir:str=None, verbose:str=True):
    """
    Setup logging to console and file.
    """
    # activate logging
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s [%(module)s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler()
        ]
    )
    if logdir is not None:
            logging.basicConfig(
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(os.path.join(logdir, f'{module_name}.log'), mode='w')
                ]
        )
    else:
        logging.basicConfig(
            handlers=[
                logging.StreamHandler()
            ]
        )

# Context manager that copies stdout and any exceptions to a log file
class CopyStdoutToFile(object):
    """
    Context manager to copy stdout to a file.
    """
    def __init__(self, filename):
        self.file = open(filename, 'a')
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self

    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self.stdout
        if exc_type is not None:
            self.file.write(traceback.format_exc())
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()
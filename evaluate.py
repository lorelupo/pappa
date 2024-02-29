
import os
import fire
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, classification_report, confusion_matrix
from utils import setup_logging
from logging import getLogger
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

logger = getLogger(__name__)

def copy_and_rename_log(output_dir, eval_dim):
    if output_dir is not None and eval_dim is not None:
        original_log_file = os.path.join(output_dir, 'evaluate.log')
        new_log_file = os.path.join(output_dir, f'evaluate_{eval_dim}.log')
        if os.path.exists(original_log_file):
            shutil.copy2(original_log_file, new_log_file)
            logger.info(f"Copied and renamed log file to {new_log_file}")
        else:
            logger.warning("Original log file does not exist. No file was copied.")

def evaluate_predictions(df, gold_labels, aggregated_gold_name='agg', output_dir=None, log_to_file=True, eval_dim=None):

    setup_logging(os.path.basename(__file__).split('.')[0], logger, output_dir if log_to_file else None)

    # Add gold_ to the gold labels' column names if 'gold' is missing
    for col in gold_labels.columns:
        if not 'gold' in col:
            gold_labels.rename(columns={col: f'gold_{col}'}, inplace=True)
    # Add the gold labels to df
    if isinstance(gold_labels, pd.DataFrame):
        df = pd.concat([df, gold_labels], axis=1)   
    elif isinstance(gold_labels, list):
        df['gold'] = gold_labels
    else:
        raise ValueError('The gold labels must be either a list or a DataFrame.')
    
    logger.info("Evaluating the predictions contained in the following dataset:")
    logger.info(f"\n{df.head()}\n")
    
    # define gold_labels method variable
    gold_labels = df.filter(regex='^gold', axis=1)

    # retrieve the name of each gold annotation
    gold_names = gold_labels.columns.tolist()

    # Lowercase all labels in predictions and gold labels
    df['prediction'] = df['prediction'].str.lower()
    gold_labels = gold_labels.apply(lambda x: x.str.lower())

    # define tables where to store results
    df_kappa = pd.DataFrame(columns=gold_names+['model'], index=gold_names+['model']).fillna(1.0)
    df_accuracy = pd.DataFrame(columns=gold_names+['model'], index=gold_names+['model']).fillna(1.0)
    df_f1 = pd.DataFrame(columns=gold_names+['model'], index=gold_names+['model']).fillna(1.0)

    for i, col in enumerate(gold_names):
        # compare agreement with gold labels
        class_report = classification_report(gold_labels[col].astype(str), df['prediction'].astype(str), output_dict=False)
        kappa = cohen_kappa_score(df['prediction'].astype(str), gold_labels[col].astype(str))
        accuracy = accuracy_score(df['prediction'].astype(str), gold_labels[col].astype(str))
        # Compute F1 with average=binary if only 2 labels, otherwise average=macro
        if len(gold_labels[col].unique()) == 2:
            f1 = f1_score((df['prediction']==gold_labels[col][0]).astype(int), (gold_labels[col]==gold_labels[col][0]).astype(int), average='binary')
        else:
            f1 = f1_score(df['prediction'].astype(str), gold_labels[col].astype(str), average='macro')
        
        if len(gold_names) > 1:
            # store results if multiple gold annotations
            df_kappa.loc['model', gold_names[i]] = df_kappa.loc[gold_names[i], 'model'] = kappa
            df_accuracy.loc['model', gold_names[i]] = df_accuracy.loc[gold_names[i], 'model'] = accuracy
            df_f1.loc['model', gold_names[i]] = df_f1.loc[gold_names[i], 'model'] = f1

            for j, col2 in enumerate(gold_names):
                if i < j:
                    # compare agreement of gold labels with each other
                    kappa = cohen_kappa_score(gold_labels[col].astype(str), gold_labels[col2].astype(str))
                    accuracy = accuracy_score(gold_labels[col].astype(str), gold_labels[col2].astype(str))
                    f1 = f1_score(gold_labels[col].astype(str), gold_labels[col2].astype(str), average='macro')
                    # store results
                    df_kappa.loc[gold_names[i], gold_names[j]] = df_kappa.loc[gold_names[j], gold_names[i]] = kappa
                    df_accuracy.loc[gold_names[i], gold_names[j]] = df_accuracy.loc[gold_names[j], gold_names[i]] = accuracy
                    df_f1.loc[gold_names[i], gold_names[j]] = df_f1.loc[gold_names[j], gold_names[i]] = f1

    # In case of multiple gold annotations, there could be a column
    # containing their aggregation computed with majority voting or tools like MACE - https://github.com/dirkhovy/MACE)
    non_agg_gold_names = [name for name in gold_names if aggregated_gold_name not in name]

    # Extract all unique labels from predictions and gold labels for plotting
    all_labels_predicted = set(df['prediction'].unique())
    all_labels_gold = set(gold_labels[gold_names].values.flatten())
    all_unique_labels = sorted(all_labels_predicted.union(all_labels_gold))

    # Calculate confusion matrix and classification report for each set of gold labels
    conf_matrices = []
    for col in gold_names:
        conf_matrix = confusion_matrix(gold_labels[col], df['prediction'], labels=None)
        conf_matrices.append(conf_matrix)

        # Log the classification report
        class_report = classification_report(gold_labels[col], df['prediction'], output_dict=False)
        logger.info(f"Classification Report for model/{col}:\n{class_report}\n")

    # Calculate the average confusion matrix
    average_conf_matrix = np.mean(conf_matrices, axis=0)

    logger.info(f"output_dir: {output_dir}, eval_dim: {eval_dim}")

    # Before the "plt.show()" command, specify the filename and directory for saving the plot
    if eval_dim is not None: 
        plot_filename = f'average_confusion_matrix_{eval_dim}.png'
    else: 
        plot_filename = 'average_confusion_matrix.png'

    plot_path = os.path.join(output_dir, plot_filename)

    # Plot the average confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(average_conf_matrix, annot=True, fmt=".2f", cmap='Blues', xticklabels=all_unique_labels, yticklabels=all_unique_labels)
    plt.title('Average Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')  # Improve label readability
    plt.yticks(rotation=45)
    plt.savefig(plot_path)
    plt.close()

    # Log the location of the saved plot
    logger.info(f"Plot saved to {plot_path}")
    
    # compute average agreement between gold annotations (except the aggregated annotation)
    if len(gold_names) > 1:
        df_kappa['mean_non_agg'] = df_kappa[non_agg_gold_names].mean(axis=1)
        df_accuracy['mean_non_agg'] = df_accuracy[non_agg_gold_names].mean(axis=1) 
        df_f1['mean_non_agg'] = df_f1[non_agg_gold_names].mean(axis=1)
        for name in non_agg_gold_names:
            # correct for gold labels fully agreeing with themselves
            df_kappa.mean_non_agg[name] = (df_kappa[non_agg_gold_names].loc[name].sum() - 1.0) / (len(non_agg_gold_names) - 1.0)
            df_accuracy.mean_non_agg[name] = (df_accuracy[non_agg_gold_names].loc[name].sum() - 1.0) / (len(non_agg_gold_names) - 1.0)
            df_f1.mean_non_agg[name] = (df_f1[non_agg_gold_names].loc[name].sum() - 1.0) / (len(non_agg_gold_names) - 1.0)
    
    # print info
    if len(gold_names) > 1:
        logger.info(f"KAPPA:\n{df_kappa.round(4)*100}\n")
        logger.info(f"Golds' mean kappa: {100*df_kappa.mean_non_agg[:-1].mean():.2f}")
        logger.info(f"Model's mean kappa: {100*df_kappa.model[:-1].mean():.2f}")

        logger.info(f"ACCURACY:\n{df_accuracy.round(4)*100}\n")
        logger.info(f"Golds' mean accuracy: {100*df_accuracy.mean_non_agg[:-1].mean():.2f}") 
        logger.info(f"Model's mean accuracy: {100*df_accuracy.model[:-1].mean():.2f}")

        logger.info(f"F1:\n{df_f1.round(4)*100}\n")
        logger.info(f"Golds' mean F1: {100*df_f1.mean_non_agg[:-1].mean():.2f}")
        logger.info(f"Model's mean F1: {100*df_f1.model[:-1].mean():.2f}")
    
        copy_and_rename_log(output_dir, eval_dim)

        return df_kappa, df_accuracy, df_f1
    else:
        logger.info(f"KAPPA: {kappa*100:.2f}")
        logger.info(f"ACCURACY: {accuracy*100:.2f}")
        logger.info(f"F1: {f1*100:.2f}")

        copy_and_rename_log(output_dir, eval_dim)
        
        return kappa, accuracy, f1
    
def evaluate_predictions_cli(df_path, gold_labels_path, aggregated_gold_name='agg', logdir=None):
    """
    Wrapper function for evaluate_predictions that takes file paths for DataFrame and gold labels.
    """
    df = pd.read_csv(df_path)  # Assuming a CSV file for simplicity, adjust accordingly
    gold_labels = pd.read_csv(gold_labels_path)  # Assuming a CSV file for simplicity, adjust accordingly

    return evaluate_predictions(df, gold_labels, aggregated_gold_name, logdir)

if __name__ == "__main__":
    fire.Fire(evaluate_predictions_cli)
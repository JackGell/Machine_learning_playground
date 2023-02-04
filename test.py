import pandas as pd
import os
import json


cols = ['btc','accuracy_x', 'buy_acc', 'sell_acc', 'hold_acc', 'feature', 'period',
        'n_layers', 'output', 'output_nodes',
       'batch_size', 'optimizer', 'normaliser',
        'TP', 'TN', 'FP', 'FN', 'precision', 'NPV', 'sensitivity',
       'specificity', 'accuracy_y', 'file_name']

def create_data_frame_from_folders(folder_path):
    data_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r') as f:
                data = json.load(f)
                try:
                    data['file_name'] = filename
                    data_list.append(data)
                except:
                    pass
    return  pd.DataFrame(data_list)

trial_df = create_data_frame_from_folders('generated_data/trials')
confusion_df = create_data_frame_from_folders('generated_data/confusion_matrix')
metric_df = create_data_frame_from_folders('generated_data/metrics')
trial_df = trial_df.merge(confusion_df, on='file_name', how='outer').merge(metric_df, on='file_name', how='outer')

trial_df = trial_df[cols]

# Define the condition to create the mask
condition = trial_df['btc'] != 1
# Create the mask
mask = condition
# Use the mask to index into the dataframe
converged = trial_df[mask]

# Define the condition to create the mask
condition = trial_df['btc'] == 1
# Create the mask
mask = condition
# Use the mask to index into the dataframe
not_converged = trial_df[mask]

a=1
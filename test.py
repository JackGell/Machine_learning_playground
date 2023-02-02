import json
import os

import numpy as np
import pandas as pd

from evaluate_model import confusion_matrix_logic, plot_data

folder = 'generated_data/3d_data'
n_files = len(os.listdir(folder)) / 2
counter = 0
predictions = []
confusion_matrix = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

while counter < n_files:
    sample = pd.read_csv(f'generated_data/3d_data/sample{counter}.csv')
    label = pd.read_csv(f'generated_data/3d_data/label{counter}.csv').values[0][1]

    output = np.array([[0, 1]]) if counter%50 else np.array([[1, 0]])
    if output.shape[1] > 1:
        index_of_max = np.argmax(output)
        prediction = np.zeros(output.shape)
        prediction[0, index_of_max] = 1
        predictions.append(prediction)
    else:
        predictions.append(output)
        prediction = output

    confusion_matrix = confusion_matrix_logic(label, prediction, confusion_matrix)
    counter += 1

trial = 10000
btc = plot_data(predictions, trial)
print(btc)
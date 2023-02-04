import json
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt


def get_dataset_balance(arr):
    unique, counts = np.unique(arr, axis=0, return_counts=True)
    print("Unique rows: ", unique)
    print("Frequencies: ", counts)


def get_dataset_balance_lin(arr):
    unique, counts = np.unique(arr, axis=0, return_counts=True)
    print("Unique rows: ", unique)
    print("Frequencies: ", counts)


def test_model_on_target(x, y, model, target_val):
    featidx = np.where((y == target_val).all(axis=1))
    sub_x = [np.array(x[idx]) for idx in featidx][0]
    sub_y = [np.array(y[idx]) for idx in featidx][0]

    arr = model.predict(sub_x, verbose=0)
    # Find the index of the highest element in each row
    index = np.argmax(arr, axis=1)
    # Create an array of zeros with the same shape as the input array
    result = np.zeros_like(arr)
    # Set the element at the index found above to 1
    result[np.arange(result.shape[0]), index] = 1

    # Compare the rows of two arrays
    count = 0
    for idx in range(sub_y.shape[0]):
        if np.all(sub_y[idx] == result[idx]):
            count += 1
    accuracy = count / sub_y.shape[0]

    return accuracy


def test_model_on_target_lin(x, y, model):
    predictions = model.predict(x, verbose=0)
    accuracy = {'buy': [], 'sell': [], 'hold': []}

    for result in zip(predictions, y):
        if result[1] > 1 + (0.1 / 100):
            if result[0] > 1 + (0.1 / 100):
                accuracy['buy'].append(1)
            else:
                accuracy['buy'].append(0)
        elif result[1] < 1 - (0.1 / 100):
            if result[0] < 1 - (0.1 / 100):
                accuracy['sell'].append(1)
            else:
                accuracy['sell'].append(0)
        else:
            if result[0] > 1 - (0.1 / 100) and result[1] < 1 + (0.1 / 100):
                accuracy['hold'].append(1)
            else:
                accuracy['hold'].append(0)

    accuracy['buy'] = sum(accuracy['buy']) / len(accuracy['buy']) if len(accuracy['buy']) else 0
    accuracy['sell'] = sum(accuracy['sell']) / len(accuracy['sell']) if len(accuracy['sell']) else 0
    accuracy['hold'] = sum(accuracy['hold']) / len(accuracy['hold']) if len(accuracy['hold']) else 0
    return accuracy


def saved_trial_parameters_so_far():
    folder = 'generated_data/trials'
    files = os.listdir(folder)
    return len(files)


def test_model_on_last_months_data(model, trial):
    folder = 'generated_data/3d_data'
    n_files = len(os.listdir(folder)) / 2
    counter = 0
    predictions = []
    confusion_matrix = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

    while counter < n_files:
        sample = pd.read_csv(f'generated_data/3d_data/sample{counter}.csv')
        label = pd.read_csv(f'generated_data/3d_data/label{counter}.csv').values[0][1]

        output = model.predict(sample.values.reshape(1, sample.shape[0], sample.shape[1]), verbose=0)
        if output.shape[1] > 1:
            index_of_max = np.argmax(output)
            prediction = np.zeros(output.shape)
            prediction[0, index_of_max] = 1
            if np.all(output == [1, 0, 0]).all() or np.all(output == [1, 0]).all():
                prediction = 'buy'
            else:
                prediction = 'sell'
        else:
            if output > 1:
                prediction = 'buy'
            else:
                prediction = 'sell'
        predictions.append(prediction)

        confusion_matrix = confusion_matrix_logic(label, prediction, confusion_matrix)
        counter += 1

    with open(f'generated_data/confusion_matrix/trial{trial}.json', 'w') as outfile:
        # Use the json.dump() method to write the dictionary to the file
        json.dump(confusion_matrix, outfile)

    precision, NPV, sensitivity, specificity, accuracy = get_confusion_matrix_metrics(confusion_matrix)
    confusion_matrix = {'precision': precision, 'NPV': NPV, 'sensitivity': sensitivity, 'specificity': specificity,
                        'accuracy': accuracy}
    with open(f'generated_data/metrics/trial{trial}.json', 'w') as outfile:
        # Use the json.dump() method to write the dictionary to the file
        json.dump(confusion_matrix, outfile)

    btc = plot_data(predictions, trial)
    return btc


def plot_data(predictions, trial):
    reference_data = pd.read_csv('generated_data/last_month_raw_data.csv')

    cross_over_index = change_points(predictions)
    ref = predictions[0]
    x, y = reference_data['timestamp'].to_list(), reference_data['close'].to_list()
    x2, y2 = reference_data['timestamp'].to_list(), predictions
    x2, y2 = get_data_at_cross_over_points(x2, y2, cross_over_index)

    btc = calculate_btc_after_test_period(cross_over_index, ref, y2)
    buy = True if ref == 'buy' else False

    plt.clf()
    for line in x2:
        if buy:
            plt.axvline(x=line, color='green', label='axvline - full height')
            buy = False
        else:
            plt.axvline(x=line, color='r', label='axvline - full height')
            buy = True
    plt.plot(x, y, color='blue')
    plt.title(f'final result: {btc} inc trading fees')
    plt.savefig(f'generated_data/plots/plot{trial}.png')
    return btc


def calculate_btc_after_test_period(idx, ref, y2):

    btc = 1 if ref == 'buy' else 0
    dollar_val_df = pd.read_csv('generated_data/last_month_raw_data.csv')['close']
    dollar_val = dollar_val_df[0]
    dollars = 0 if btc else dollar_val_df[0]

    ref = False if btc else True

    for i, y in enumerate(y2):

        dollar_val = dollar_val_df[idx[i]]
        if not ref:
            btc, dollars = 0, btc * dollar_val - (btc * dollar_val * (0.1 / 100))
        else:
            btc, dollars = dollar_val / dollars - (dollar_val / dollars) * (0.1 / 100), 0

        ref = False if ref else True

    return btc if btc else dollars / dollar_val


def get_data_at_cross_over_points(x, y, indexs):
    x1, y1 = [], []
    for idx in indexs:
        x1.append(x[idx])
        y1.append(y[idx])
    return x1, y1


def change_points(lst):
    change_points = []
    for i in range(1, len(lst)):
        if lst[i] != lst[i-1]:
            change_points.append(i)
    return change_points


def confusion_matrix_logic(label, prediction, confusion_matrix):
    if label > 1:
        if prediction == 'buy':
            confusion_matrix['TP'] += 1
        else:
            confusion_matrix['FN'] += 1
    else:
        if prediction == 'buy':
            confusion_matrix['FP'] += 1
        else:
            confusion_matrix['TN'] += 1

    return confusion_matrix


def get_confusion_matrix_metrics(confusion_matrix):
    try:
        precision = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FP'])
    except:
        precision = None
    try:
        NPV = confusion_matrix['TN'] / (confusion_matrix['TN'] + confusion_matrix['FN'])
    except:
        NPV = None
    try:
        sensitivity = confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FN'])
    except:
        sensitivity = None
    try:
        specificity = confusion_matrix['TN'] / (confusion_matrix['TN'] + confusion_matrix['FP'])
    except:
        specificity = None
    try:
        accuracy = (confusion_matrix['TN'] + confusion_matrix['TP']) / (
                confusion_matrix['TP'] + confusion_matrix['FP'] + confusion_matrix['TN'] + confusion_matrix['FN'])
    except:
        accuracy = None
    return precision, NPV, sensitivity, specificity, accuracy

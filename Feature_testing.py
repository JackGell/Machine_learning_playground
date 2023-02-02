import random
import shutil

import tensorflow as tf
from keras.regularizers import l1
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
import os

def clear_3d_data_folder():
    folder = 'generated_data/3d_data'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def resample(x, y, output, percentage=0.5):

    if output == 'softmax':
        unique_target_arrays, counts = np.unique(y, axis=0, return_counts=True)
        Max, Min = np.amax(counts), np.amin(counts)
    if output == 'linear':
        count_sell = np.sum(np.less(y, 1-(0.1/100)))
        count_buy = np.sum(np.greater(y, 1+(0.1/100)))
        Max, Min = max(count_sell, count_buy), min(count_sell, count_buy)

    diff = Max - Min
    Height = Min + diff * percentage


    def upsample_subarrays(x, y, target_val, Height):
        if x.shape[0] == Height:
            return x, y

        featidx = np.where((y == target_val).all(axis=1))
        sub_x = [np.array(x[idx]) for idx in featidx][0]
        sub_y = [np.array(y[idx]) for idx in featidx][0]

        while (sub_x.shape[0] < Height):
            # Select a random index from the original array
            random_index = np.random.randint(0, sub_x.shape[0])

            # Select the same row from both arrays using the random index
            random_row1 = sub_x[random_index, :]
            random_row2 = sub_y[random_index, :]

            # Append the random rows to both arrays
            sub_x = np.append(sub_x, [random_row1], axis=0)
            sub_y = np.append(sub_y, [random_row2], axis=0)
        return sub_x, sub_y

    def upsample_subarrays_lin(x, y, target_val, Height):
        if target_val > 1 + (0.1 / 100):
            featidx = np.where((y > 1 + (0.1 / 100)))
        elif target_val < 1 - (0.1 / 100):
            featidx = np.where((y < 1 - (0.1 / 100)))
        else:
            featidx = np.where((y >= 1 - (0.1 / 100)) & (y <= 1 + (0.1 / 100)))
        sub_x = [np.array(x[idx]) for idx in featidx][0]
        sub_y = [np.array(y[idx]) for idx in featidx][0]

        while (sub_x.shape[0] < Height):
            # Select a random index from the original array
            random_index = np.random.randint(0, sub_x.shape[0])

            # Select the same row from both arrays using the random index
            random_row1 = sub_x[random_index, :]
            random_row2 = sub_y[random_index]

            # Append the random rows to both arrays
            sub_x = np.append(sub_x, [random_row1], axis=0)
            sub_y = np.append(sub_y, [random_row2], axis=0)
        return sub_x, sub_y

    def downsample_subarrays(subx, suby, Height):
        if subx.shape[0] > Height:
            while (subx.shape[0] > Height):
                # Select a random index from the original array
                random_index = np.random.randint(0, subx.shape[0])

                # Delete the same row from both arrays using the random index
                subx = np.delete(subx, random_index, axis=0)
                suby = np.delete(suby, random_index, axis=0)
        return subx, suby

    subarraysy, subarraysx = [], []
    targeta = unique_target_arrays if output == 'softmax' else [0, 1, 2]
    for target in targeta:
        if output == 'softmax':
            sub_x, sub_y = upsample_subarrays(x, y, target, Height)
        else:
            sub_x, sub_y = upsample_subarrays_lin(x, y, target, Height)
        sub_x, sub_y = downsample_subarrays(sub_x, sub_y, Height)
        subarraysx.append(sub_x)
        subarraysy.append(sub_y)

    # Concatenate the arrays along the first axis (rows)
    x = np.concatenate([subarraysx[idx] for idx in range(len(subarraysx))], axis=0)
    y = np.concatenate([subarraysy[idx] for idx in range(len(subarraysy))], axis=0)

    return x, y


def random_int(a, b):
    return random.randint(a, b)


def random_float(a, b):
    return random.uniform(a, b)


def rand_activation():
    activations = ['sigmoid',
                   'relu',
                   'tanh',
                   'gru',
                   'lstm',
                   'bi-lstm']
    return activations[random_int(0, len(activations) - 1)]

def rand_optimizer(selection='Rand'):
    opt = [tf.keras.optimizers.Adam(),
           tf.keras.optimizers.SGD(),
           tf.keras.optimizers.RMSprop(),
           tf.keras.optimizers.Adadelta()]
    string_opt = ['Adam', 'SGD', 'RMSprop', 'Adadelta']

    if selection == 'Rand':
        pointer = random_int(0, len(opt) - 1)
        object_opt = opt[pointer]
        str_opt = string_opt[pointer]
    else:
        pointer = string_opt.index(selection)
        str_opt = selection
        object_opt = opt[pointer]
    return object_opt, str_opt

def shape_data(data, Y, output, rows_per_period = 20):

    # Initialize a list to store the periods
    periods_list = []
    Y_periods_list = []
    # Iterate through the periods
    for i in range(data.shape[0], rows_per_period, -1):
        # Extract the data for this period
        end_index = i
        start_index = end_index - rows_per_period
        period_data = data[start_index:end_index, :]

        # Append the data for this period to the list
        Y_period = Y[end_index-1]

        # Append the data and Y element for this period to their respective lists
        periods_list.append(period_data)
        Y_periods_list.append(Y_period)

    # Convert the lists of periods and Y elements to numpy arrays
    periods_array = np.array(periods_list)
    Y_periods_array = np.array(Y_periods_list)

    # Convert the Y values
    if output == 'softmax':
        Y_converted = np.zeros((Y_periods_array.shape[0], Y_periods_array.shape[1]))
        for i in range(Y_periods_array.shape[0]):
            if Y_periods_array.shape[1] == 3:
                if all(Y_periods_array[i] == [1, 0, 0]):
                    Y_converted[i, :] = [1, 0, 0]
                elif all(Y_periods_array[i] == [0, 1, 0]):
                    Y_converted[i, :] = [0, 1, 0]
                else:
                    Y_converted[i, :] = [0, 0, 1]
            if Y_periods_array.shape[1] == 2:
                if all(Y_periods_array[i] == [1, 0]):
                    Y_converted[i, :] = [1, 0]
                elif all(Y_periods_array[i] == [0, 1]):
                    Y_converted[i, :] = [0, 1]
    else:
        Y_converted = Y_periods_array

    # Replace Y_periods_array with the converted values
    Y_periods_array = Y_converted

    clear_3d_data_folder()
    test_set_raw = pd.read_csv('generated_data/last_month_raw_data.csv')

    for idx in range(test_set_raw.shape[0]-1, -1, -1):
        df = pd.DataFrame(periods_array[idx])
        df.to_csv(f"generated_data/3d_data/sample{idx}.csv")
        if isinstance(Y_periods_array[0], float):
            df = pd.DataFrame([Y_periods_array[idx]])
        else:
            if np.all(Y_periods_array[idx] == [1, 0]) or np.all(Y_periods_array[idx] == [1, 0, 0]):
                df = pd.DataFrame([1.1])
            else:
                df = pd.DataFrame([0.9])
        df.to_csv(f"generated_data/3d_data/label{idx}.csv")

    return periods_array[:-test_set_raw.shape[0]], Y_periods_array[:-test_set_raw.shape[0]]

def set_label(close_value):
    if close_value > 1+(0.1/100):
        return [1, 0, 0]    #buy
    elif close_value < 1-(0.1/100):
        return [0, 1, 0]    #sell
    else:
        return [0, 0, 1]    #hold

def set_label2(close_value):
    if close_value > 1:
        return [1, 0]  # buy
    else:
        return [0, 1]  # sell

# Min-Max Scaling
def min_max_scaler(data):
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Z-Score Normalization
def z_score_normalizer(data):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Log Transformation
def log_transformer(data):
    transformer = PowerTransformer(method='yeo-johnson')
    return pd.DataFrame(transformer.fit_transform(data), columns=data.columns)

def scale_data(df, x=0):

    # Select the columns you want to scale
    df = df.rename(columns={'close': 'label'})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    target = df['label']
    df_scaled = df.drop(columns=['label'])

    if x == 0:
        scaled_df = min_max_scaler(df_scaled)
        normaliser = 'min_max_scaler'
    elif x == 1:
        scaled_df = z_score_normalizer(df_scaled)
        normaliser = 'z_score_normalizer'
    elif x == 2:
        scaled_df = log_transformer(df_scaled)
        normaliser = 'log_transformer'

    # Replace the original data with the scaled data
    return scaled_df, target, normaliser


def random_float(a, b):
    return random.uniform(a, b)

def rand_activation():
    activations = ['sigmoid',
                   'relu',
                   'tanh',
                   'lstm',
                   'bi-lstm',
                   'gru']
    return activations[random_int(0, len(activations)-1)]

def rand_optimizer(selection = 'Rand'):
    opt = [tf.keras.optimizers.Adam(),
            tf.keras.optimizers.SGD(),
            tf.keras.optimizers.RMSprop(),
            tf.keras.optimizers.Adadelta()]
    string_opt = ['Adam', 'SGD', 'RMSprop', 'Adadelta']

    if selection == 'Rand':
        pointer = random_int(0, len(opt)-1)
        object_opt = opt[pointer]
        str_opt = string_opt[pointer]
    else:
        pointer = string_opt.index(selection)
        str_opt = selection
        object_opt = opt[pointer]
    return object_opt, str_opt


def recommend(metric, epsilon_parameters):

    # Find the highest accuracy score
    max_accuracy = max(epsilon_parameters['accuracy'])
    # Find the index of the trial with the highest accuracy score
    max_accuracy_index = epsilon_parameters['accuracy'].index(max_accuracy)
    return epsilon_parameters[metric][max_accuracy_index]

def hidden_layer(activation, l1f, neurons):
    if 'lstm' == activation:
        return tf.keras.layers.LSTM(neurons, kernel_regularizer=l1(l1f), return_sequences=True)
    elif 'bi-lstm' == activation:
        return tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(neurons, kernel_regularizer=l1(l1f), return_sequences=True))
    elif 'gru' == activation:
        return tf.keras.layers.GRU(neurons, kernel_regularizer=l1(l1f), return_sequences=True)
    else:
        return tf.keras.layers.Dense(neurons, kernel_regularizer=l1(l1f), activation=activation)

def layer_one(activaion1, l1f, rows, cols, neurons):
    if activaion1 == 'bi-lstm':
        return tf.keras.layers.LSTM(neurons, kernel_regularizer=l1(l1f), input_shape=(rows, cols), return_sequences=True)
    elif activaion1 == 'lstm':
        return tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(neurons, kernel_regularizer=l1(l1f), input_shape=(rows, cols), return_sequences=True))
    else:
        return tf.keras.layers.GRU(neurons, kernel_regularizer=l1(l1f), input_shape=(rows, cols), return_sequences=True)
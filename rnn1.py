
from sklearn.model_selection import train_test_split
from Feature_testing import shape_data, recommend, random_int, random_float, resample, rand_activation, rand_optimizer, \
    hidden_layer, layer_one
from assemble_data import generate_features
from keras.layers import Dropout
import json
from evaluate_model import test_model_on_target, test_model_on_target_lin, saved_trial_parameters_so_far, \
    test_model_on_last_months_data
import tensorflow as tf
import warnings
import threading
from telegram import telegram_thread
warnings.filterwarnings("ignore", category=FutureWarning)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

trial_parameters ={}
epochs = 5000000000000

outputs = ['linear', 'softmax']
output = 'linear'
data, target, kline_interval, normaliser = generate_features(output, output_nodes=2)
features = data.columns
best_sofar = 0
init = False
trial = saved_trial_parameters_so_far()
h_layer_range = 6

epsilon_parameters = {'btc': [],
                      'recurrent2': [],
                      'feature': [],
                      'output_nodes': [],
                      'period': [],
                      'percentage': [],
                      'n_layers': [],
                      'output': [],
                      'accuracy': [0],
                      'learning_rate': [],
                      'batch_size': [],
                      'kline_interval': [],
                      'optimizer': [],
                      'normaliser': [],
                      'config': [],
                      'buy_acc': [],
                      'sell_acc': [],
                      'hold_acc': []}

for i in range(h_layer_range+1):
    epsilon_parameters[f'activation_{i}'] = []
    epsilon_parameters[f'neurons_{i}'] = []
    epsilon_parameters[f'l1_{i}'] = []
    epsilon_parameters[f'dropout_{i}'] = []

while 1:
    '''
    output = outputs[random_int(0, 1)] if random_float(0,1)>0.5 else recommend('output', epsilon_parameters) if init else 'linear'
    output_nodes = 1 if output == 'linear' else random_int(2, 3) if init else 2
    learning_rate = random_float(0.01, 0.001) if random_float(0,1)>0.1 else recommend('learning_rate', epsilon_parameters) if init else 0.01
    period = random_int(10, 100) if random_float(0, 1)>0.2 else recommend('period', epsilon_parameters) if init else 100
    feature = features[random_int(0, len(features)-1)] if random_int(0, len(features) - 1) == 0 else 'None'
    percentage = random_float(0.1, 0.9) if random_float(0, 1)>0.2 else recommend('percentage', epsilon_parameters) if init else 0.5
    l1f = random_float(0, 0.2) if random_float(0, 1)>0.2 else recommend(f'l1_{0}', epsilon_parameters) if init else 0.2
    neurons1 = random_int(100, 600) if random_float(0, 1)>0.2 else recommend(f'neurons_{0}', epsilon_parameters) if init else 30
    n_layers = random_int(2,h_layer_range) if random_float(0, 1)>0.2 else recommend('n_layers', epsilon_parameters) if init else 1
    batch_size = (random_int(8, 128) if random_float(0, 1)>0.5 else recommend('batch_size', epsilon_parameters)) if init else 64
    activation1 = ['lstm', 'bi-lstm', 'gru'][random_int(0, 2)]
    activation = rand_activation()
    '''
    output = outputs[random_int(0, 1)] if init else 'linear'
    output_nodes = 1 if output == 'linear' else random_int(2, 3) if init else 2
    #learning_rate = random_float(0.01, 0.001) if random_float(0,1)>0.1 else recommend('learning_rate', epsilon_parameters) if init else 0.01
    period = random_int(10, 100) if init else 100
    feature = features[random_int(0, len(features)-1)] if random_int(0, len(features) - 1) == 0 else 'None'
    percentage = random_float(0.1, 0.9) if random_float(0, 1)>0.2 else recommend('percentage', epsilon_parameters) if init else 0.5
    l1f = random_float(0, 0.2) if random_float(0, 1)>0.2 else recommend(f'l1_{0}', epsilon_parameters) if init else 0.2
    neurons1 = random_int(100, 600) if random_float(0, 1)>0.2 else recommend(f'neurons_{0}', epsilon_parameters) if init else 30
    n_layers = random_int(2,h_layer_range) if init else 1
    batch_size = random_int(8, 128) if init else 64
    activation1 = ['lstm', 'bi-lstm', 'gru'][random_int(0, 2)]
    activation = rand_activation()

    '''Overwriters'''
    l1f = 0
    feature = 'None'
    learning_rate = 'None'
    dropout = 0
    neurons = 500
    neurons1 = 400
    activation = ['relu', 'sigmoid', 'tanh'][random_int(0, 2)]

    data, target, kline_interval, normaliser = generate_features(output, output_nodes)
    data = data.drop(labels=feature, axis=1) if feature != 'None' else data
    data, target = shape_data(data.values, target.values, output, rows_per_period=period)
    #data, target = resample(data, target, output, percentage=percentage)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, shuffle=False)
    model = tf.keras.Sequential()
    model.add(layer_one(activation1, l1f, X_test.shape[1], X_test.shape[2], neurons1))
    if random_float(0, 1) > 0.5:
        epsilon_parameters['recurrent2'].append(True)
        model.add(hidden_layer(activation1, l1f, neurons))
    else:
        epsilon_parameters['recurrent2'].append(False)

    for i in range(n_layers):
        #neurons = random_int(30,700)
        #dropout = random_float(0, 0.4)

        model.add(hidden_layer(activation, l1f, neurons))
        model.add(Dropout(dropout))

        epsilon_parameters[f'activation_{i+1}'].append(activation)
        epsilon_parameters[f'neurons_{i+1}'].append(neurons)
        epsilon_parameters[f'l1_{i+1}'].append(l1f)
        epsilon_parameters[f'dropout_{i+1}'].append(dropout)

    model.add(tf.keras.layers.Flatten())
    if output == 'softmax':
        model.add(tf.keras.layers.Dense(output_nodes, activation='softmax'))
        loss = 'categorical_crossentropy' if output_nodes>2 else 'binary_crossentropy'
    else:
        model.add(tf.keras.layers.Dense(1, activation='linear'))
        loss=['mean_squared_error', 'mean_absolute_error'][random_int(0, 1)]
    metric = ['val_loss']

    optimizer, str_opt = rand_optimizer(selection=('Rand'))
    early_stop = tf.keras.callbacks.EarlyStopping(monitor=metric[0], patience=5)
    #lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: learning / (1 + epoch * 0.02))
    #lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 0.9)

    print('Training')
    model.compile(loss=loss, optimizer=optimizer)
    model.fit(X_train, y_train, verbose=0, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stop])
    print('Trained')

    if output == 'softmax':
        if output_nodes == 3:
            buy_dict = test_model_on_target(X_test, y_test, model, [1, 0, 0])
            sell_dict = test_model_on_target(X_test, y_test, model, [0, 1, 0])
            hold_dict = test_model_on_target(X_test, y_test, model, [0, 0, 1])
            acc_dict = (hold_dict+sell_dict+buy_dict)/3
        else:
            buy_dict = test_model_on_target(X_test, y_test, model, [1, 0])
            sell_dict = test_model_on_target(X_test, y_test, model, [0, 1])
            hold_dict = 'none'
            acc_dict = (sell_dict+buy_dict)/2
    else:
        accuracy = test_model_on_target_lin(X_test, y_test, model)
        buy_dict = accuracy['buy']
        sell_dict = accuracy['sell']
        hold_dict = accuracy['hold']
        acc_dict = (accuracy['buy']+accuracy['sell']+accuracy['hold'])/3

    if init:
        epsilon_parameters['accuracy'].append(acc_dict)
    else:
        epsilon_parameters['accuracy'][0] = acc_dict

    trial += 1
    best_sofar = max(epsilon_parameters['accuracy'][-1], best_sofar)

    btc = test_model_on_last_months_data(model, trial)

    print(f'trial :{trial}')
    #print(f'accuracies buy: {buy_dict} sell: {sell_dict} hold: {hold_dict} all: {acc_dict}')
    print(f'validation set this trial: {epsilon_parameters["accuracy"][-1]}')
    print(f'best so far is: {best_sofar}')
    print('x'*100)

    epsilon_parameters['btc'].append(btc)
    epsilon_parameters['learning_rate'].append(learning_rate)
    epsilon_parameters['output_nodes'].append(output_nodes)
    epsilon_parameters['normaliser'].append(normaliser)
    epsilon_parameters['kline_interval'].append(kline_interval)
    epsilon_parameters['feature'].append(feature)
    epsilon_parameters['output'].append(output)
    epsilon_parameters['period'].append(period)
    epsilon_parameters['percentage'].append(percentage)
    epsilon_parameters[f'neurons_{0}'].append(neurons1)
    epsilon_parameters[f'l1_{0}'].append(l1f)
    epsilon_parameters[f'activation_{0}'].append(activation1)
    epsilon_parameters[f'dropout_{0}'].append(0)
    epsilon_parameters['n_layers'].append(n_layers)
    #epsilon_parameters['config'].append(model.get_config())
    epsilon_parameters['batch_size'].append(batch_size)
    epsilon_parameters['optimizer'].append(str_opt)
    epsilon_parameters['buy_acc'].append(buy_dict)
    epsilon_parameters['sell_acc'].append(sell_dict)
    epsilon_parameters['hold_acc'].append(hold_dict)

    trial_parameters[f'trial{trial}'] = {'btc': epsilon_parameters['btc'][-1],
                                         'recurrent2': epsilon_parameters['recurrent2'][-1],
                                         'accuracy': epsilon_parameters['accuracy'][-1],
                                         'buy_acc': epsilon_parameters['buy_acc'][-1],
                                         'sell_acc': epsilon_parameters['sell_acc'][-1],
                                         'hold_acc': epsilon_parameters['hold_acc'][-1],
                                         'feature': epsilon_parameters['feature'][-1],
                                         'period': epsilon_parameters['period'][-1],
                                         'percentage': epsilon_parameters['percentage'][-1],
                                         'n_layers': epsilon_parameters['n_layers'][-1],
                                         'output': epsilon_parameters['output'][-1],
                                         'output_nodes': epsilon_parameters['output_nodes'][-1],
                                         'learning_rate': epsilon_parameters['learning_rate'][-1],
                                         'batch_size': epsilon_parameters['batch_size'][-1],
                                         'kline_interval': epsilon_parameters['kline_interval'][-1],
                                         'optimizer': epsilon_parameters['optimizer'][-1],
                                         'normaliser': epsilon_parameters['normaliser'][-1]}

    for i in range(n_layers+1):
        trial_parameters[f'trial{trial}'][f'activation_{i}'] = epsilon_parameters[f'activation_{i}'][-1]
        trial_parameters[f'trial{trial}'][f'neurons_{i}'] = epsilon_parameters[f'neurons_{i}'][-1]
        trial_parameters[f'trial{trial}'][f'l1_{i}'] = epsilon_parameters[f'l1_{i}'][-1]
        trial_parameters[f'trial{trial}'][f'dropout_{i}'] = epsilon_parameters[f'dropout_{i}'][-1]


    # Open a file to write the JSON data to
    with open(f'generated_data/trials/trial{trial}.json', 'w') as outfile:
      # Use the json.dump() method to write the dictionary to the file
      json.dump(trial_parameters[f'trial{trial}'], outfile)
    with open(f'generated_data/feature_tester.json', 'w') as outfile:
      # Use the json.dump() method to write the dictionary to the file
      json.dump(trial_parameters, outfile)

    #if not init:
    #    telegram_thread = threading.Thread(target=telegram_thread)
    #    telegram_thread.start()
    init = True


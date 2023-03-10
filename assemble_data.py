import numpy as np
import pandas as pd
import talib.abstract as ta
from binance.client import Client
import csv
import random

from Feature_testing import set_label, scale_data, set_label2


def get_bitcoin_data():

    intervals = [Client.KLINE_INTERVAL_4HOUR,
                 Client.KLINE_INTERVAL_6HOUR,
                 Client.KLINE_INTERVAL_8HOUR,
                 Client.KLINE_INTERVAL_12HOUR,
                 Client.KLINE_INTERVAL_1DAY]
    interval = intervals[random.randint(0, len(intervals) - 1)]

    # Create a client instance
    try:
        client = Client("api_key", "api_secret")

        print(f'intervals {interval}')
        # Retrieve the data
        klines = client.get_historical_klines("BTCUSDT", interval, "1 Jan, 2017")

        # Open a CSV file for writing
        with open(f"generated_data/bitcoin_data{interval}.csv", "w", newline="") as file:
            # Create a CSV writer
            writer = csv.writer(file)

            # Write the header row
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "num_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])

            # Loop through the candle sticks
            for kline in klines:
                # Write the data to the CSV file
                writer.writerow(kline)
        return interval
    except:
        return interval

def populate_indicators(interval):

    dataframe = pd.read_csv(f'generated_data/bitcoin_data{interval}.csv')
    # Convert the data to a DataFrame
    keep_columns = ["timestamp", "open", "high", "low", "close", "volume", "num_trades"]
    dataframe = dataframe.drop([col for col in dataframe.columns if col not in keep_columns], axis=1)
    # Momentum Indicators
    # ------------------------------------

    # ADX
    dataframe['adx'] = ta.ADX(dataframe)

    # # Plus Directional Indicator / Movement
    dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
    dataframe['plus_di'] = ta.PLUS_DI(dataframe)

    # # Minus Directional Indicator / Movement
    dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
    dataframe['minus_di'] = ta.MINUS_DI(dataframe)

    # # Aroon, Aroon Oscillator
    aroon = ta.AROON(dataframe)
    dataframe['aroonup'] = aroon['aroonup']
    dataframe['aroondown'] = aroon['aroondown']
    dataframe['aroonosc'] = ta.AROONOSC(dataframe)

    # # Ultimate Oscillator
    dataframe['uo'] = ta.ULTOSC(dataframe)

    # # Commodity Channel Index: values [Oversold:-100, Overbought:100]
    dataframe['cci'] = ta.CCI(dataframe)

    # RSI
    dataframe['rsi'] = ta.RSI(dataframe)

    # # Inverse Fisher transform on RSI: values [-1.0, 1.0] (https://goo.gl/2JGGoy)
    rsi = 0.1 * (dataframe['rsi'] - 50)
    dataframe['fisher_rsi'] = (np.exp(2 * rsi) - 1) / (np.exp(2 * rsi) + 1)

    # # Inverse Fisher transform on RSI normalized: values [0.0, 100.0] (https://goo.gl/2JGGoy)
    dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)

    # # Stochastic Slow
    stoch = ta.STOCH(dataframe)
    dataframe['slowd'] = stoch['slowd']
    dataframe['slowk'] = stoch['slowk']

    # Stochastic Fast
    stoch_fast = ta.STOCHF(dataframe)
    dataframe['fastd'] = stoch_fast['fastd']
    dataframe['fastk'] = stoch_fast['fastk']

    # # Stochastic RSI
    # Please read https://github.com/freqtrade/freqtrade/issues/2961 before using this.
    # STOCHRSI is NOT aligned with tradingview, which may result in non-expected results.
    stoch_rsi = ta.STOCHRSI(dataframe)
    dataframe['fastd_rsi'] = stoch_rsi['fastd']
    dataframe['fastk_rsi'] = stoch_rsi['fastk']

    # MACD
    macd = ta.MACD(dataframe)
    dataframe['macd'] = macd['macd']
    dataframe['macdsignal'] = macd['macdsignal']
    dataframe['macdhist'] = macd['macdhist']

    # MFI
    dataframe['mfi'] = ta.MFI(dataframe)

    # # ROC
    dataframe['roc'] = ta.ROC(dataframe)



    # # EMA - Exponential Moving Average
    dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)
    dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
    dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
    dataframe['ema21'] = ta.EMA(dataframe, timeperiod=21)
    dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
    dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

    # # SMA - Simple Moving Average
    dataframe['sma3'] = ta.SMA(dataframe, timeperiod=3)
    dataframe['sma5'] = ta.SMA(dataframe, timeperiod=5)
    dataframe['sma10'] = ta.SMA(dataframe, timeperiod=10)
    dataframe['sma21'] = ta.SMA(dataframe, timeperiod=21)
    dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
    dataframe['sma100'] = ta.SMA(dataframe, timeperiod=100)

    # Parabolic SAR
    dataframe['sar'] = ta.SAR(dataframe)

    # TEMA - Triple Exponential Moving Average
    dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)

    # Cycle Indicator
    # ------------------------------------
    # Hilbert Transform Indicator - SineWave
    hilbert = ta.HT_SINE(dataframe)
    dataframe['htsine'] = hilbert['sine']
    dataframe['htleadsine'] = hilbert['leadsine']


    return dataframe

def generate_features(output, output_nodes):
    kline_interval = get_bitcoin_data()
    df = populate_indicators(kline_interval)

    n_samples_for_testing = calculate_samples_per_month(df['timestamp'][0], df['timestamp'][1])
    df[-n_samples_for_testing:].to_csv("generated_data/last_month_raw_data.csv", index=False)
    df.drop(columns=['timestamp'], inplace=True)

    # Divide the shifted DataFrame by the original DataFrame
    cols_to_shift = ['open', 'high', 'low', 'close', 'volume', 'num_trades',
                     'ema3', 'ema5', 'ema10', 'ema21', 'ema50', 'ema100',
                     'sma3', 'sma5', 'sma10', 'sma21', 'sma50', 'sma100']
    for col in cols_to_shift:
        df[col] = df[col] / df[col].shift(1)

    df.dropna(inplace=True)

    if output == 'softmax':
        if output_nodes == 3:
            label_df = df['close'].apply(set_label)
        else:
            label_df = df['close'].apply(set_label2)
    else:
        label_df = df['close']

    df.drop(columns=['close'], inplace=True)
    df = pd.concat([df, label_df], axis=1)
    df.dropna(inplace=True)
    df, target, normaliser = scale_data(df, x=random.randint(0, 2))

    return df, target, kline_interval, normaliser


def calculate_samples_per_month(sample1, sample2):
    # Calculate the difference in days between start and end
    difference = abs(sample1 - sample2)
    # Calculate the number of samples per day
    samples_per_day = 24 * 60 * 60 * 1000 / difference
    # Calculate the number of samples per month (assuming 30 days per month)
    samples_per_month = int(samples_per_day * 30)
    return samples_per_month
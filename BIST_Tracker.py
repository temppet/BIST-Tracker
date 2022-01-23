import datetime
import json
import math
import os
import requests
import sqlite3
import time
import urllib
import urllib.request
import numpy as np
import pandas as pd
from telegram.ext import Updater

# TELEGRAM Constants
TG_TOKEN = '1079945138:AAFpAWUd8AYgTsq8dkjJKg_HOEWAIXHItTA'
TG_URL = "https://api.telegram.org/bot{}/".format(TG_TOKEN)
ChatID = '-434324601'  # '990652176': me, #'-553704593': group
# Telegram BOT starter functions
updater = Updater(TG_TOKEN)
dispatcher = updater.dispatcher
MyBot = updater.bot  # Bot instance to use that is bound to the token.

FIRE_EMOJI, CLOCK_EMOJI, CHART_EMOJI, GREEN_CHECK_EMOJI = '\U0001F525', '\U0001F55B', '\U0001F4C8', '\U00002705'
GREEN_CIRCLE_EMOJI, RED_CIRCLE_EMOJI, GREEN_SQUARE_EMOJI = '\U0001F7E2', '\U0001F534', '\U0001F7E9'
RED_SQUARE_EMOJI = '\U0001F7E5'


class stock:
    def __init__(self, name):
        self.name = name
        self.data = {'TIME': 0, 'OPEN': 0, 'HIGH': 0, 'LOW': 0, 'CLOSE': 0, 'VOLUME': 0}
        self.df = pd.DataFrame(data=self.data, index=[0])
        self.State = 1
        self.C1, self.C2, self.C3 = 0, 0, 0
        print(f"{self.name} is created...")
        # self.filename_tf1 = "C:/Users/forga/OneDrive/Masaüstü/Codecamp/Python/TeleReport/DBs/H1_" + self.name

    @staticmethod
    def update_indicators(df):
        df['RSI'] = calc_rsi(df['CLOSE'])
        df['IFTRSI'] = calc_iftrsi(df['RSI'])
        df['BOLL_H'], df['BOLL_L'], df['BOLL_M'] = calc_bollinger(df['CLOSE'])
        df['MACD'], df['MACD_S'] = calc_macd(df['CLOSE'])
        df['STD'] = calc_std(df['CLOSE'])
        df['MA_200'] = sma(df['CLOSE'], 200)
        df['MA_50'] = sma(df['CLOSE'], 50)
        df['MA_10'] = sma(df['CLOSE'], 10)
        df['VOLUME(10)'] = sma(df['VOLUME'], 10)
        df['FUN'], df['FUN(8)'] = calc_fun(df)

    def initialize_dfs(self, start_time, end_time, resolution):
        t = time.time()
        url = 'https://web-cloud-new.foreks.com/tradingview-services/trading-view/history?symbol=' + self.name + '.E.BIST&resolution=' + resolution + '&from=' + str(
            int(start_time)) + '&to=' + str(int(end_time)) + '&currency=TRL'
        data = urllib.request.urlopen(url).read()
        data = json.loads(data.decode('utf-8'))
        print(f'{self.name} data were collected in {time.time() - t:.2f} seconds...')
        self.df.drop(index=self.df.index[0], axis=0, inplace=True)
        for i in range(len(data['t'])):
            self.df = self.df.append({
                'TIME': data['t'][i],
                'DATE': str(datetime.datetime.fromtimestamp(data['t'][i] / 1000.0)),
                'OPEN': data['o'][i],
                'HIGH': data['h'][i],
                'LOW': data['l'][i],
                'CLOSE': data['c'][i],
                'VOLUME': data['v'][i],
                'RSI': 0, 'IFTRSI': 0, 'BOLL_H': 0, 'BOLL_M': 0, 'BOLL_L': 0, 'STD': 0, 'MACD': 0, 'MACD_S': 0,
                'MA_200': 0, 'MA_50': 0, 'MA_10': 0, 'VOLUME(10)': 0, 'BUY': 0
            }, ignore_index=True)
        self.update_indicators(self.df)
        print(f"{self.name} dataframe is initialized...")


def create_Stocks():
    StockList = ['TCELL', ]
    MyList = np.array([])
    for i in range(len(StockList)):
        MyList = np.append(MyList, stock(StockList[i]))
    return MyList


def calc_rsi(series):
    rsi_length = 14
    delta = series.diff().dropna()
    delta = delta[1:]
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up1 = up.ewm(alpha=1 / rsi_length).mean()
    roll_down1 = down.abs().ewm(alpha=1 / rsi_length).mean()
    rsi1 = 100.0 - (100.0 / (1.0 + roll_up1 / roll_down1))
    return rsi1


def calc_iftrsi(series):
    period = 9
    v1 = 0.1 * (series - 50)
    v2 = v1.ewm(alpha=1 / period).mean()
    iftrsi = (np.exp(2 * v2) - 1) / (np.exp(2 * v2) + 1)
    return iftrsi


def calc_bollinger(df):
    length = 20
    std = 2
    return (df.rolling(window=length).mean() + std * df.rolling(window=length).apply(
        np.std)), (df.rolling(window=length).mean() - std * df.rolling(window=length).apply(np.std)), (
               df.rolling(window=length).mean())


def sma(df, period):
    return df.rolling(window=period).mean()


def calc_std(df):
    length = 20
    return df.rolling(window=length).apply(np.std)


def calc_macd(series):
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def calc_fun(df):
    rsi_data = 0.01 * (100 - df['RSI'])
    iftrsi_data = 1 - (df['IFTRSI'] / 2 + 0.5)
    macd_data = np.array([])
    for i in range(len(df['MACD'])):
        if df['MACD'].iloc[i] > 0 or df['MACD_S'].iloc[i] > 0:
            macd_data = np.append(macd_data, 0)
        elif df['MACD'].iloc[i] < 0 and 0 > df['MACD_S'].iloc[i] > df['MACD'].iloc[i]:
            macd_data = np.append(macd_data, 0.75)
        elif 0 > df['MACD'].iloc[i] > df['MACD_S'].iloc[i] and df['MACD_S'].iloc[i] < 0:
            macd_data = np.append(macd_data, 0.99)
        else:
            macd_data = np.append(macd_data, 0)
    macd_data = pd.Series(macd_data)

    w_rsi = 20
    w_iftrsi = 60
    w_macd = 20
    indicator = (w_rsi * rsi_data + w_iftrsi * iftrsi_data + w_macd * macd_data)
    indicator_sma = sma(indicator, 8)
    return indicator, indicator_sma


stock_list = create_Stocks()
stock_1 = stock_list[0]

START_TIME = 1637355600000
END_TIME = 1642712400000
RESOLUTION = '15'
stock_1.initialize_dfs(START_TIME, END_TIME, RESOLUTION)
print(stock_1.df.tail())

msg = 'BIST tracker baslatildi.'
try:
    MyBot.send_message(chat_id=ChatID, text=msg)
except Exception as e:
    print(str(e))
    try:
        time.sleep(20)
        MyBot.send_message(chat_id=ChatID, text=msg)
    except:
        print("Telegram send error...")

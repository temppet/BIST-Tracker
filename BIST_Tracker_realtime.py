import datetime
import json
import time
import urllib
import urllib.request

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from telegram.ext import Updater

# TELEGRAM Constants
TG_TOKEN = '1079945138:AAFpAWUd8AYgTsq8dkjJKg_HOEWAIXHItTA'
TG_URL = "https://api.telegram.org/bot{}/".format(TG_TOKEN)
ChatID = '-434324601'  # '990652176': me, #'-553704593': group
# Telegram BOT starter functions
updater = Updater(TG_TOKEN, use_context=True)
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
                'MA_200': 0, 'MA_50': 0, 'MA_10': 0, 'VOLUME(10)': 0, 'BUY': np.nan, 'FUN': 0, 'FUN(8)': 0
            }, ignore_index=True)
        self.update_indicators(self.df)
        print(f"{self.name} dataframe is initialized...")


def create_Stocks():
    StockList = ['GUBRF', ]
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

    w_rsi = 30
    w_iftrsi = 40
    w_macd = 30
    indicator = (w_rsi * rsi_data + w_iftrsi * iftrsi_data + w_macd * macd_data)
    indicator_sma = sma(indicator, 10)
    return indicator, indicator_sma


stock_list = create_Stocks()
stock_1 = stock_list[0]

START_TIME = 1637355600000
END_TIME = 1642712400000
RESOLUTION = '15'
stock_1.initialize_dfs(START_TIME, END_TIME, RESOLUTION)
print(stock_1.df.tail())


# msg = 'BIST tracker baslatildi.'
# try:
#     MyBot.send_message(chat_id=ChatID, text=msg)
# except Exception as e:
#     print(str(e))
#     try:
#         time.sleep(20)
#         MyBot.send_message(chat_id=ChatID, text=msg)
#     except:
#         print("Telegram send error...")

def plot():
    df = stock_1.df
    for i in range(1, len(df['FUN(8)'])):
        buy_cond_1 = df['FUN(8)'][i] < df['FUN(8)'][i - 1] and df['FUN(8)'][i - 2] < df['FUN(8)'][i - 1]
        buy_cond_2 = df['FUN(8)'][i] > 70
        buy_cond_3 = df['FUN(8)'][i] > 70 and df['FUN(8)'][i - 1] > 70 and df['FUN(8)'][i - 2] > 70
        if buy_cond_1 and buy_cond_2 and buy_cond_3:
            df.loc[i, 'BUY'] = 1
        else:
            df.loc[i, 'BUY'] = np.nan
        sell_cond_1 = df['FUN(8)'][i] > df['FUN(8)'][i - 1] and df['FUN(8)'][i - 2] > df['FUN(8)'][i - 1]
        sell_cond_2 = df['FUN(8)'][i] < 30
        sell_cond_3 = df['FUN(8)'][i] < 30 and df['FUN(8)'][i - 1] < 30 and df['FUN(8)'][i - 2] < 30
        if sell_cond_1 and sell_cond_2 and sell_cond_3:
            df.loc[i, 'SELL'] = 1
        else:
            df.loc[i, 'SELL'] = np.nan

    back_idx = -600
    df = stock_1.df.copy()
    df = df.rename(columns={'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low', 'CLOSE': 'Close', 'VOLUME': 'Volume'})
    df['TIME'] = pd.to_datetime(df['TIME'], unit='ms')
    df = df.set_index('TIME')
    marksize = 100
    bollingers = df[['BOLL_H', 'BOLL_L', 'BOLL_M']]
    apd = []
    apd.append(mpf.make_addplot(bollingers[back_idx:], panel=0, y_on_right=False, secondary_y=False, type='line',
                                color='#8f0b0b', width=1.5))
    apd.append(mpf.make_addplot(df['FUN(8)'][back_idx:], panel=1, y_on_right=False, secondary_y=False, type='line',
                                color='blue', width=1.2))
    apd.append(mpf.make_addplot(70 * np.ones(len(df['FUN(8)'][back_idx:])), panel=1, color='white', secondary_y=False))
    apd.append(mpf.make_addplot(20 * np.ones(len(df['FUN(8)'][back_idx:])), panel=1, color='white', secondary_y=False))

    apd.append(mpf.make_addplot(df['BUY'][back_idx:] * df['Low'][back_idx:] * 0.995, scatter=True, secondary_y=False,
                                y_on_right=False, markersize=marksize, marker='^', color='g'))
    apd.append(mpf.make_addplot(df['SELL'][back_idx:] * df['High'][back_idx:] * 1.005, scatter=True, secondary_y=False,
                                y_on_right=False, markersize=marksize, marker='^', color='r'))

    mc = mpf.make_marketcolors(up='g', down='r', wick='w', edge='w')
    s = mpf.make_mpf_style(base_mpf_style='binance', base_mpl_style='dark_background',
                           gridcolor='#000000', facecolor='#000000', y_on_right=False, marketcolors=mc, gridstyle=':',
                           rc={
                               'axes.edgecolor': '#737378',
                               'axes.linewidth': 0.4,
                               'axes.labelsize': 'medium',
                               'axes.labelweight': 'semibold',
                               # 'lines.linewidth': 1.0,
                               'font.weight': 'medium',
                               'font.size': 8.0}, mavcolors=['#c3c90c', '#f08b11', '#11f0e1', '#117df0'])

    fig, axlist = mpf.plot(df[back_idx:], volume=False, type='candle', figsize=(16, 9), style=s, ylabel=("ETH_"),
                           addplot=apd, scale_width_adjustment=dict(candle=2), returnfig=True,
                           mav=(10, 50, 100, 200))
    axlist[0].legend(['MOV10', 'MOV50', 'MOV100', 'MOV200'])
    fig.text(0.1, 0.9, stock_1.name, size=10, fontweight='bold', color='w')
    fig.subplots_adjust(bottom=0.1, right=0.5, top=3.9, hspace=3)
    plt.show()

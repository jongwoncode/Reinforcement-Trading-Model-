import os
import warnings
import pandas as pd
import numpy as np
import utils
warnings.filterwarnings(action='ignore')

COLUMNS_CHART_DATA = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
COLUMNS_TRADING_DATA =['close_ema15_ratio', 'volume_ema15_ratio', 'close_ema33_ratio', 'volume_ema33_ratio', 
                       'close_ema56_ratio', 'volume_ema56_ratio', 'close_ema112_ratio', 'volume_ema112_ratio',
                       'open_lastclose_ratio', 'close_lastclose_ratio','volume_lastvolume_ratio', 
                       'high_close_ratio', 'close_low_ratio', 'max9_close_ratio',
                        'max26_close_ratio', 'max52_close_ratio', 'close_min9_ratio', 'close_min26_ratio', 
                        'close_min52_ratio', 'close_fspan1_ratio', 'close_fspan2_ratio',
                        'bbupper_close_ratio', 'close_bblower_ratio', 'std20_mean20_ratio']

def preprocess(data) :
    # 데이터 전처리
    '''
    [지수이평선] - [5, 15, 33, 56, 112]
    '''
    windows = [5, 15, 33, 56, 112]
    for window in windows :
        data[f'close_ema{window}'] = data['Close'].ewm(span = window).mean()
        data[f'volume_ema{window}'] = data['Volume'].ewm(span = window).mean()
        # ratio
        data[f'close_ema{window}_ratio'] = (data['Close']-data[f'close_ema{window}'])/data[f'close_ema{window}']
        data[f'volume_ema{window}_ratio'] = (data['Volume']-data[f'volume_ema{window}'])/data[f'volume_ema{window}']

    data['open_lastclose_ratio'] = np.zeros(len(data))      # 당일 시가(t) 전일 종가(t-1) 관계
    data['close_lastclose_ratio'] = np.zeros(len(data))     # 당일 종가(t) 전일 종가(t-1) 관계
    data['volume_lastvolume_ratio'] = np.zeros(len(data))   # 당일 거래량(t) 전일 거래량(t-1) 관계
    data.loc[1:, 'open_lastclose_ratio'] = (data['Open'][1:].values - data['Close'][:-1].values)/data['Close'][:-1].values
    data.loc[1:, 'close_lastclose_ratio'] = (data['Close'][1:].values - data['Close'][:-1].values) / data['Close'][:-1].values
    data.loc[1:, 'volume_lastvolume_ratio'] = ((data['Volume'][1:].values - data['Volume'][:-1].values) 
                                                / data['Volume'][:-1].replace(to_replace=0, method='ffill').replace(to_replace=0, method='bfill').values)

    '''
    [chart 비율]
    '''
    data['high_close_ratio'] = (data['High'].values - data['Close'].values)/data['Close'].values     # 당일 고가(t) 당일 종가(t)
    data['close_low_ratio'] = (data['Close'].values - data['Low'].values)/data['Low'].values        # 당일 종가(t) 당일 저가(t)

    '''
    [일목균형표] 
        1. short_period(9일), mid_period(26일), long_period(52일) 최대 최소, 
        2. 전환선(change_line), 기준선(base_line), 
        3. 선행스팬1,2 (fspan1, fspan2)
    '''
    short_period, mid_period, long_period = 9, 26, 52
    data[f'max{short_period}'] = data['High'].rolling(window = short_period).max()
    data[f'min{short_period}'] = data['Low'].rolling(window = short_period).min()
    data[f'max{mid_period}'] = data['High'].rolling(window = mid_period).max()
    data[f'min{mid_period}'] = data['Low'].rolling(window = mid_period).min()
    data[f'max{long_period}'] = data['High'].rolling(window = long_period).max()
    data[f'min{long_period}'] = data['Low'].rolling(window = long_period).min()
    data['change_line'] = (data[f'max{short_period}'] + data[f'min{short_period}'])/2
    data['base_line'] = (data[f'max{mid_period}'] + data[f'min{mid_period}'])/2

    # 선행 스팬 1
    data['fspan1'] = (data['base_line'] + data['change_line'])/2
    data['fspan1'] = data['fspan1'].shift(mid_period-1)

    # 선행 스팬 2 
    data['fspan2'] = (data[f'max{long_period}'] + data[f'min{long_period}'])/2
    data['fspan2'] = data['fspan2'].shift(mid_period-1)

    # n일 고가 당일 종가(t) 관계
    data[f'max{short_period}_close_ratio'] = (data[f'max{short_period}'].values - data['Close'].values)/data['Close'].values
    data[f'max{mid_period}_close_ratio'] = (data[f'max{mid_period}'].values - data['Close'].values)/data['Close'].values 
    data[f'max{long_period}_close_ratio'] = (data[f'max{long_period}'].values - data['Close'].values)/data['Close'].values
    # 당일 종가(t) n일 저가 관계
    data[f'close_min{short_period}_ratio'] = (data['Close'].values - data[f'min{short_period}'].values)/data[f'min{short_period}'].values
    data[f'close_min{mid_period}_ratio'] = (data['Close'].values - data[f'min{mid_period}'].values)/data[f'min{mid_period}'].values 
    data[f'close_min{long_period}_ratio'] = (data['Close'].values - data[f'min{long_period}'].values)/data[f'min{long_period}'].values 
    # 당일 종가(t) 선행 스팬1, 2 관계
    data['close_fspan1_ratio'] = (data['Close'].values - data['fspan1'].values)/data['fspan1'].values
    data['close_fspan2_ratio'] = (data['Close'].values - data['fspan2'].values)/data['fspan2'].values


    '''
    [Bollinger Band] 
        - 20일 단순 이평, 20일 표준편차*(0.945)*2
        - [upper band, lower band,]
    '''
    # Bollinger Band 계산
    period, deviation = 20, 2
    data['mean20'] = data['Close'].rolling(window = period).mean()
    data['std20'] = data['Close'].rolling(window = period).std()*0.945
    data['bbupper'] = data['mean20'] + deviation*data['std20']
    data['bblower'] = data['mean20'] - deviation*data['std20']

    # bbupper(t) 당일 종가(t) 관계, bblower(t) 당일 종가(t) 관계
    data['bbupper_close_ratio'] = (data['bbupper'].values - data['Close'].values)/data['Close'].values
    data['close_bblower_ratio'] = (data['Close'].values - data['bblower'].values)/data[f'bblower'].values 
    data['std20_mean20_ratio'] = data['std20']/data['mean20']
    return data


def make_sequence_data(chart, training, window_size=5) :
    chart_columns = chart.columns
    chart_list, seqeunce_list = [], []

    for i in range(len(training)-window_size) :
        seqeunce_list.append(np.array(training.iloc[i:i+window_size]))
        chart_list.append(np.array(chart.iloc[i+window_size-1]))
    
    df_chart = pd.DataFrame(chart_list, columns=chart_columns) 
    return df_chart, np.array(seqeunce_list)

    
def load_data(code, start_date, end_date, n_steps) :
    df = pd.read_csv(os.path.join(utils.BASE_DIR, 'data', f'{code}.csv'), thousands=',', converters={'Date' : lambda x : str(x)})
    # sorting Date and reset index
    df = df.sort_values(by='Date').reset_index(drop=True)
    df = preprocess(df)
    # change datetime notation
    df['Date'] = df['Date'].str.replace('-', '')
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    # df = df.fillna(method='ffill').reset_index(drop=True)
    
    # Remove NaN rows (Because of forward span(52-1+26-1))
    df = df.iloc[76:, :]
    
    # Split orginal / preprocess column
    chart_data = df[COLUMNS_CHART_DATA]
    training_data = df[COLUMNS_TRADING_DATA]

    # Make sequence set for LSTM Network input
    chart_data, training_data = make_sequence_data(chart_data, training_data, window_size=n_steps)
    return chart_data, training_data
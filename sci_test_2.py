import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

trade = pd.read_csv('./trade.csv')
quote = pd.read_csv('./quote.csv')

trade['recv_time'] = pd.to_datetime(trade['recv_time']/1000, unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai').dt.strftime('%H:%M:%S')
quote['recv_time'] = pd.to_datetime(quote['recv_time']/1000, unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai').dt.strftime('%H:%M:%S')

t_recv_time = trade['recv_time'].values
t_symbol = trade['symbol'].values
trade_price = trade['trade_price'].values
trade_qty = trade['trade_qty'].values

q_recv_time = quote['recv_time'].values
q_symbol = quote['symbol'].values
bid_price = quote['bid_price'].values
bid_size = quote['bid_size'].values
ask_price = quote['ask_price'].values
ask_size = quote['ask_size'].values

bid_price_list = []
ask_price_list = []
spread_list = []

for i in range(t_recv_time.shape[0]):
    if (t_symbol[i]=="000021.SZ"):
        recv_time = t_recv_time[i]
        for j in range(q_recv_time.shape[0]):
            if (q_symbol[j]=="000021.SZ" and q_recv_time[j] == recv_time):
                spread_list.append(ask_price[j]-bid_price[j])

print(spread_list[0:100])
print(len(spread_list))

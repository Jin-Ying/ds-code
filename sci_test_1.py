import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
trade = pd.read_csv('./trade.csv')
quote = pd.read_csv('./quote_sample.csv')

trade['recv_time'] = pd.to_datetime(trade['recv_time']/1000, unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai').dt.strftime('%H:%M:%S')
quote['recv_time'] = pd.to_datetime(quote['recv_time']/1000, unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai').dt.strftime('%H:%M:%S')

recv_time = trade['recv_time'].values
symbol = trade['symbol'].values
trade_price = trade['trade_price'].values
trade_qty = trade['trade_qty'].values

trade_qty_per_min = {}
for i in range(recv_time.shape[0]):
    if (symbol[i]=='000021.SZ'):
        trade_qty_per_min[recv_time[i][:-3]] = 0

for i in range(recv_time.shape[0]):
    if (symbol[i]=='000021.SZ'):
        trade_qty_per_min[recv_time[i][:-3]] = trade_qty_per_min[recv_time[i][:-3]] + trade_qty[i]

time_list = []
qty_list = []
i=0
for key in trade_qty_per_min.keys():
    if (i==0):
        i=1
        continue
    time_list.append(key)
    qty_list.append(trade_qty_per_min[key])

# 画图
plt.plot(time_list, qty_list)
plt.show()
print(iii)

qty_dict = {}
for i in range(recv_time.shape[0]):
    qty_dict[symbol[i]]=0

hh = 9
min_mm = 30
max_mm = 59
for i in range(recv_time.shape[0]):
    hour, minute = int(recv_time[i].split(":")[0]), int(recv_time[i].split(":")[1])
    # if (hour == hh and minute>=min_mm and minute<=max_mm):
    qty_dict[symbol[i]] = qty_dict[symbol[i]]+ trade_qty[i]
sorted_qty = sorted(qty_dict.items(), key=lambda x:x[1])
print(sorted_qty)

max_qty = 0
max_key = None
for key in qty_dict.keys():
    if qty_dict[key] > max_qty:
        max_qty = qty_dict[key]
        max_key = key
print(max_qty)
print(max_key)
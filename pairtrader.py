import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import *

from utils import *
from account import *


class PairTrader:
    def __init__(self, X, Y, z_crit, z_sl, z_tp, trainval_split, window, trade_size):
        assert len(Y) == len(X)
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.z_crit = z_crit
        self.z_sl = z_sl
        self.z_tp = z_tp
        self.trainval_split = trainval_split
        self.window = window
        self.trade_size = trade_size
        
    def fit_model(self):
        self.b1, self.b0 = linregress(x=self.X[:self.trainval_split], y=self.Y[:self.trainval_split])[0:2]
        self.spread = self.Y - self.b1*self.X
        self.ma = rollingMA(self.spread, self.window)
        self.sd = rollingSD(self.spread, self.window) # Note: if window is too small; sd may be 0
        self.z = (self.spread - self.ma)/self.sd
    
    def test_model(self):
        self.account = Account(self.trade_size)
        self.logs = []
        for i in range(self.trainval_split, len(self.Y)):
            n = self.trade_size/self.Y[i]
            if len(self.account.positions) == 0:
                if -self.z_crit - self.z_sl < self.z[i] < -self.z_crit: # long spread
                    self.account.update_position('Y', n, self.Y[i]) # buy Y
                    self.account.update_position('X', -n*self.b1, self.X[i]) # sell b1*X
                    stoploss = (-self.z_crit - self.z_sl)*self.sd[i] + self.ma[i]
                    takeprofit = (-self.z_crit + self.z_tp)*self.sd[i] + self.ma[i]
                    info = {'spread': self.spread[i], 'stoploss': stoploss, 'takeprofit': takeprofit}
                    self.logs.append((i, 'L', info))
                elif self.z_crit < self.z[i] < self.z_crit + self.z_sl: # short spread
                    self.account.update_position('Y', -n, self.Y[i]) # sell Y
                    self.account.update_position('X', n*self.b1, self.X[i]) # buy b1*X
                    stoploss = (self.z_crit + self.z_sl)*self.sd[i] + self.ma[i]
                    takeprofit = (self.z_crit - self.z_tp)*self.sd[i] + self.ma[i]
                    info = {'spread': self.spread[i], 'stoploss': stoploss, 'takeprofit': takeprofit}
                    self.logs.append((i, 'S', info))
            else:
                if self.logs[-1][1] == 'L':
                    if self.spread[i] < self.logs[-1][2]['stoploss']:
                        self.account.update_position('Y', 'close', self.Y[i]) # sell Y
                        self.account.update_position('X', 'close', self.X[i]) # buy b1*X
                        self.logs.append((i, 'SL', self.spread[i]))
                    elif self.spread[i] > self.logs[-1][2]['takeprofit']:
                        self.account.update_position('Y', 'close', self.Y[i]) # sell Y
                        self.account.update_position('X', 'close', self.X[i]) # buy b1*X
                        self.logs.append((i, 'TP', self.spread[i]))
                elif self.logs[-1][1] == 'S':
                    if self.spread[i] > self.logs[-1][2]['stoploss']:
                        self.account.update_position('Y', 'close', self.Y[i]) # buy Y
                        self.account.update_position('X', 'close', self.X[i]) # sell b1*X
                        self.logs.append((i, 'SL', self.spread[i]))
                    elif self.spread[i] < self.logs[-1][2]['takeprofit']:
                        self.account.update_position('Y', 'close', self.Y[i]) # buy Y
                        self.account.update_position('X', 'close', self.X[i]) # sell b1*X
                        self.logs.append((i, 'TP', self.spread[i]))
        return self.account, self.logs
    
    def plot(self, type, figsize=(20, 10), zoom=False, markersize=78):
        self.longs = [i[0] for i in self.logs if i[1] == 'L']
        self.shorts = [i[0] for i in self.logs if i[1] == 'S']
        self.stop_loss = [i[0] for i in self.logs if i[1] == 'SL']
        self.take_profit = [i[0] for i in self.logs if i[1] == 'TP']
        
        plot = plt.figure(figsize=figsize)
        plt.axvline(self.trainval_split, linestyle='--', color='black', label='Train/Val Split')
        
        if type.lower() == 'spread':
            y = self.spread
            plt.plot(y, label='Spread')
            plt.plot(self.ma, linestyle='--', linewidth=.5, color='black', label='{} Tick Moving Average'.format(self.window))
            plt.plot(self.ma+self.z_crit*self.sd, linestyle='--', linewidth=.5, color='black', label='MA + {}*sigma'.format(self.z_crit))
            plt.plot(self.ma-self.z_crit*self.sd, linestyle='--', linewidth=.5, color='black', label='MA - {}*sigma'.format(self.z_crit))
            
        if type.lower() == 'z':
            y = self.z
            plt.plot(y, label='Z Scores (Standarized Spread)')
            plt.axhline(self.z_crit, linestyle='--', linewidth=.5, color='black')
            plt.axhline(-self.z_crit, linestyle='--', linewidth=.5, color='black')
            
        plt.scatter(self.longs, y[self.longs], marker=6, s=markersize, color='black', label='Long Spread')
        plt.scatter(self.shorts, y[self.shorts], marker=7, s=markersize, color='black', label='Short Spread')
        plt.scatter(self.stop_loss, y[self.stop_loss], marker='x', s=markersize, color='red', label='Stop Loss')
        plt.scatter(self.take_profit, y[self.take_profit], marker='o', s=markersize, color='green', label='Take Profit')
        plt.legend(loc=3)
        
        if zoom:
            plt.xlim(self.trainval_split, len(y))
            
        return plot
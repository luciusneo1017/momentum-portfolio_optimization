import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

class MomentumVectorBacktest():

    def __init__(self,symbol,start,end,amount,tc):
        '''
        Class for vectorised backtesting of a Momentum based trading strategy
    
        '''
        self.symbol = symbol
        self.tc = tc # transaction costs in %
        self.amount = amount # amount to be invested
        self.start = start
        self.end = end
        self.results = None
        self.data = self.get_data()

    def get_data(self):
        '''
        Retrieves and prepares data in a DataFrame Object
        For now, data is retrieved from yfinance
        '''
        raw = yf.download(self.symbol, start = self.start, end = self.end)
        raw = raw.loc[:,('Close', self.symbol)].rename('price').to_frame()
        raw['return'] = np.log(raw['price']/raw['price'].shift(1))
        return raw
    
    def run_strategy(self, momentum):
        '''
        Backtests the trading strategy
        '''
        self.momentum = momentum
        data = self.data.copy().dropna()
        data['position'] = np.sign(data['return'].rolling(momentum).mean())
        data['strategy'] = data['position'].shift(1) * data['return']
        data.dropna(inplace = True)
        trades = data['position'].diff().fillna(0) != 0
        data.loc[trades,'strategy'] -= self.tc
        data['creturns'] =  self.amount * data['return'].cumsum().apply(np.exp)
        data['cstrategy'] = self.amount * data['strategy'].cumsum().apply(np.exp)
        self.results = data

        returns = data['creturns'].iloc[-1]
        grossperf = data['cstrategy'].iloc[-1] # absolute performance of strategy
        operf = grossperf - data['creturns'].iloc[-1] # out/under-performance of strategy
        
        return (round(returns,2),round(grossperf,2), round(operf,2))
    
    def plot_results(self):
        '''
        Plots the cumulative performance of the trading strategy
        '''
        if self.results is None:
            print('No results to plot yet. Run a strategy.')
        title = f"{self.symbol} | Momentum = {self.momentum} | TC = {self.tc} "
        self.results[['creturns', 'cstrategy']].plot(title = title, figsize = (10,6))
        plt.show()

    def max_drawdown(self): # Implementing Max Drawdown with Sliding Window
        l,r = 0,1
        max_dd = 0
        arr = self.results['cstrategy']
        while r < len(arr):
            if arr[l] > arr[r]:
                dd = arr[l] - arr[r]
                max_dd = max(max_dd,dd)
            else:
                l = r
            r += 1
        print(f'Max Drawdown:{max_dd}')
        return max_dd




K = MomentumVectorBacktest('EURUSD=X','2023-12-01','2025-06-30',10000,0.0)
K.run_strategy(3)
K.plot_results()
K.max_drawdown()
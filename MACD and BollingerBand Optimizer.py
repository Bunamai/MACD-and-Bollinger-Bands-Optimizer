import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from  _datetime import datetime as dt
import matplotlib.pyplot as plt

class FinancialData:
    # initialize function
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start = start_date
        self.end = end_date
        self.get_data(self.symbol, self.start, self.end)
        self.prepare_data()

    # function call to retrieve daily data
    def get_data(self, symbol, start, end):
        self.data = yf.download(symbol, start=start, end=end)

    # preparing data - adding daily returns and buy/hold returns column
    def prepare_data(self):
        self.data['Daily Returns'] = np.log(self.data['Adj Close'] / self.data['Adj Close'].shift())
        self.data['bnh_returns'] = self.data['Daily Returns'].cumsum()
        self.data.dropna(inplace=True)

    # function to plot a list of attributes in the pandas data frame
    def plot_data(self, attribute_list):
        self.data[attribute_list].plot()
        plt.show()

    # plotting strategy returns
    def plot_strategy_returns(self):
        self.plot_data(['bnh_returns', 'Strategy Returns'])

class MACDAndBollingerBandBackTester(FinancialData):
    # Prepare indicators for the strategy
    def prepare_indicators(self, window1, window2, window3, window4):
        self.data["ma1"] = self.data["Adj Close"].ewm(span=window1).mean()
        self.data["ma2"] = self.data["Adj Close"].ewm(span=window2).mean()
        self.data["ma3"] = self.data['Adj Close'].rolling(window=window3).mean()
        self.data['Moving std'] = self.data['Adj Close'].rolling(window=window3).std()
        self.data['Upper Band'] = self.data["ma3"] + 2 * self.data['Moving std']
        self.data['Lower Band'] = self.data["ma3"] - 2 * self.data['Moving std']
        self.data["MACD"] = self.data["ma1"] - self.data["ma2"]
        self.data["MACD_S"] = self.data["MACD"].ewm(span=window4).mean()

    # Get the strategy return with the given parameters
    def backtest_strategy(self, window1, window2, window3, window4, start=None):
        self.prepare_indicators(window1, window2, window3, window4)
        if start is None:
            start = np.max(np.array([window1, window2, window3, window4]))

        # BUY condition
        self.data["Signal"] = np.where(((self.data["MACD"] < self.data["MACD_S"]) & (self.data["MACD"].shift(1) >= self.data["MACD_S"])) & ((self.data['Adj Close'] < self.data['Lower Band']) & (self.data['Adj Close'].shift(1) >= self.data['Lower Band'])), 1, 0)

        # SELL condition
        self.data["Signal"] = np.where(((self.data["MACD"] > self.data["MACD_S"]) & (self.data["MACD"].shift(1) <= self.data["MACD_S"])) & ((self.data['Adj Close'] > self.data['Upper Band']) & (self.data['Adj Close'].shift(1) <= self.data['Upper Band'])), -1, self.data["Signal"])

        self.data["Position"] = [0] * len(self.data)
        self.data["Position"] = self.data["Signal"].replace(to_replace=0, method="ffill")
        self.data["Position"] = self.data["Position"].shift()

        self.data["Strategy Returns"] = self.data["Daily Returns"] * self.data["Position"]

        performance = self.data[["Daily Returns", "Strategy Returns"]].iloc[start:].sum()

        self.data["Strategy Returns"] = self.data["Strategy Returns"].cumsum()

        return performance

    # Find the best time frame of the indicators
    def optimize_MACD_and_bollinger_band_parameters(self, windows1, windows2, windows3, windows4):
        start = np.max(np.array([windows1, windows2, windows3, windows4]))
        self.results = pd.DataFrame()
        for window1 in windows1:
            for window2 in windows2:
                if window1 >= window2:
                    continue
                for window3 in windows3:
                    for window4 in windows4:
                        perf = self.backtest_strategy(window1=window1, window2=window2, window3=window3, window4=window4, start=start)
                        self.result = pd.DataFrame({"Window 1":window1, "Window 2": window2, "Window 3": window3, "Window 4": window4, "BnH Returns": perf["Daily Returns"], "Strategy Returns": perf["Strategy Returns"]}, index=[0, ])
                        self.results = self.results.append(self.result, ignore_index=True)
        self.results.sort_values(by="Strategy Returns", inplace=True, ascending=False)
        self.results = self.results.reset_index()
        self.results = self.results.drop("index", axis=1)
        print(self.results.head())

    def plot_MACD_and_bollinger_band_returns(self):
        if (len(self.results)) > 0:
            window1 = self.results.loc[0, "Window 1"]
            window2 = self.results.loc[0, "Window 2"]
            window3 = self.results.loc[0, "Window 3"]
            window4 = self.results.loc[0, "Window 4"]
            start = np.max(np.array([window1, window2, window3, window4]))
            print("Window 1:", window1, "Window 2:", window2, "Window 3:", window3, "Window 4:", window4)
            self.backtest_strategy(window1=window1, window2=window2, window3=window3, window4=window4)
            deduction = self.data.iloc[start, 7]
            self.data["bnh_returns"] = self.data["bnh_returns"] - deduction
            self.data[["bnh_returns", "Strategy Returns"]].iloc[start:].plot()
            plt.show()


# Sample run
ticker = "AAPL"
start = dt(2010, 1, 1)
end = dt.today()
TEST = MACDAndBollingerBandBackTester(ticker, start, end)
TEST.optimize_MACD_and_bollinger_band_parameters(range(1, 16), range(15, 30), range(15, 30), range(1, 16))
TEST.plot_MACD_and_bollinger_band_returns()
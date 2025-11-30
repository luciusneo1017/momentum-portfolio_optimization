#need close and open, signals use close, rebalancing trade at the next open
import yfinance as yf
import pandas as pd
import os
import sys
from pathlib import Path

def preprocess_data():

    cwd = Path.cwd()
    data_dir = cwd /'data'

    clean_data_dir = cwd / 'cleaned_data'
    clean_data_dir.mkdir(parents=True, exist_ok=True)

    for file in os.listdir(data_dir):
        df = pd.read_csv(os.path.join(data_dir,file))
        
        df.columns =['Date', 'Close', 'High', 'Low', 'Open', 'Volume'] #setting columns which occur in this order
        df = df.iloc[2:].reset_index(drop=True)
        df = df[['Date','Close','Open','Volume']]

    # forward fill missing price data, bfill() incase first row is nan
        df[['Close', 'Open']] = df[['Close', 'Open']].ffill().bfill()

        try:
            assert df['Close'].isna().sum() == 0, "There is missing data in 'Close' series"
            assert df['Open'].isna().sum() == 0, "There is missing data in 'Open' series"
        except AssertionError as e:
            print(f'File name {file} caught an assertion error {e}')

        
        df.to_csv(clean_data_dir/file, index = False)

    # catch for ensuring all files in data_dir are cleaned and passed to clean_data_dir 
    assert len(os.listdir(data_dir)) == len(os.listdir(clean_data_dir)), "# of files in data dir do not match # of files in clean data dir"

preprocess_data()
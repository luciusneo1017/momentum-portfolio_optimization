import pandas as pd
import yfinance as yf
import os

input_file = "HSI Composite List.xlsx"
path = os.path.join(os.getcwd(),'momentumhk', input_file)
HSIC = pd.read_excel(path)
tickers= HSIC['Stock Code']
mask = tickers[tickers.apply(lambda x: type(x)) != int]
tickers = tickers.drop(mask.index)
tickers.index = range(len(tickers))

#------------------- checksum----------------
HSIC_constituents_n = 503

assert len(tickers) == HSIC_constituents_n, \
    f"Expected {HSIC_constituents_n} constituents, got {len(tickers)}"

def create_tickers(ticker:int) -> str:
    ticker = str(ticker).strip()
    if len(ticker) == 4:
        ticker += '.HK'
    else:
        tick = [0] * 4 # fixed length array
        ticker_list = [int(k) for k in ticker]
        
        for i in range(-1,-(len(ticker_list)+1),-1):
            
            tick[i] = ticker_list[i]
           
        ticker = ''.join(str(j) for j in tick) + '.HK'

    return ticker

tickers = tickers.apply(lambda x: create_tickers(x)).to_list()

start_date = "2009-01-01"
end_date = "2025-10-31"
for t in tickers:
    try:
        df= yf.download(t, start=start_date, end=end_date, auto_adjust=True, progress=True, keepna=True)
    except Exception as e:
        print(f"Error downloading data for {t}: {e}")
        continue
    file_path = os.path.join(os.getcwd(), "data", f"{t}.csv")
    df.to_csv(file_path)

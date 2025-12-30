# Momentum + Portfolio Optimization Model

This repo contains a project which details a Momentum + Portfolio Optimization Strategy

# Brief summary of Model
The goal of this project is to build a momentum-based strategy that can outperform a passive benchmark index, with a specific focus on APAC equity markets. The initial approach was straightforward: periodically rank a broad APAC stock universe by time-series momentum, select the strongest performers, and rebalance on a fixed schedule. In reading practitioner and academic work, I found this setup aligns with how some large systematic fund managers implement momentum in practice, for example, a common baseline is 12–1 momentum (proxied by total return over prior 12 months excluding the last month) with quarterly rebalancing according to a specifc weighting scheme (e.g. AQR Momentum Indices implementation).

To extend beyond the baseline, I introduced a weighting scheme on top of the momentum filter to study how allocation choices affect risk and performance relative to an  momentum-constructed portfolio. In particular, I explored whether optimizing weights (rather than allocating 1/N (equal weight allocation)) meaningfully improves outcomes, and what trade-offs it introduces.

**Initial thoughts and key design choices**

1) How many stocks should the portfolio hold?
A first question was the portfolio size. I chose 40 stocks based on the idea that once you hold roughly 30+ names, most idiosyncratic risk is diversified away and the marginal benefit of adding more names becomes relatively small. A 40-stock portfolio typically should be large enough to diversify away single-name idiosyncratic risk while still allowing the strategy to express a momentum tilt. (Of course, the degree of this diversification benefit also depends on cross-asset correlations)

2) How should momentum be measured (and over what horizon)?
The next decision was the momentum signal definition and lookback window. A widely used specification is 12–1 momentum: rank assets by their total return over the past 12 months while skipping the most recent month. The skip-month is commonly used because very short-term returns tend to exhibit mean reversion, which can dilute a medium-term momentum signal.

There was also a practical constraint: since the strategy only holds 40 assets, the momentum lookback period needs to provide enough observations for stable covariance estimation when using mean–variance optimization (MVO). The window should be long enough to avoid an overly noisy covariance matrix that can lead to unstable weights.



# Design and Implementation
This section walks through the design choices and implementation details.

## Retrieving data - get_data.py
Price data is retrieved from yfinance api (future improvement: migrate to Databento). 
The Hong Kong stock universe is based on the **HSI Composite** constituents downloaded from the HKEX/HSI website https://www.hsi.com.hk/ (PDF converted to XLSX).

**load_constituents()**
- Reads the constituents list from the XLSX file using *pd.read_excel()* into a DataFrame.
- Extracts the **Stock Code** column into a list *tickers*.
- Cleans the list by removing non-integer entries (e.g., notes, whitespace, invalid rows).

**create_ticker()**
For HK stocks, yfinance api requires ticker code to be in 'xxxx.HK' format where xxxx is the stock code. 
- If a stock code has 4 digits, return *{code}.HK*.
- Otherwise, left-pad with zeros to 4 digits (e.g., 1 -> *0001.HK*) before appending *.HK*.
**main()**
- Calls *load_constituents()* to obtain the cleaned ticker list.
- Performs validation checks to verify number of tickers we have is the same as in our composite list
- For each ticker, formats it using *create_ticker()*.
- Downloads OHLCV data from yfinance using the configured *start/end dates* and parameters such as *auto_adjust*, *progress*, and *keepna*.
- Saves each ticker’s data to data/raw/<TICKER>.csv

## Preprocessing data - preprocess_data.py
**preprocess_data()**
Cleans the raw CSV files in data/raw/ and writes processed files to data/interim/
- Drops the first 2 rows (due to the structure of the data we get from yfinance) an 
- Stnadardised column names to  *'Date', 'Close', 'High', 'Low', 'Open', 'Volume'* respectively. 
- Apply forward fill data to eradicate missing entries and backfill to remove leading empty entries, making sure the first (or first few rows) are not Nan. 
- We then save the data to data/interim

A limitation of forward/backward filling is that if a ticker has large gaps in its price series, then forward filling would create a non existent 'flat' price path that might affect the filtering based on momentum rankings and the portfolio weights at rebalancing dates. In this project, I chose forward/backward filling as a pragmatic simplification to keep the backtesting pipeline consistent to focus on researching the effects of different weighting schemes.

**build_wide_and_save()**
Reads the per-ticker CSV files from data/interim/ and constructs a wide panel DataFrame by concatenating all tickers side-by-side into a MultiIndex column structure (field, ticker). We then extract the specific matrices used in the backtest and save them to data/cleaned/.

*prices.csv* contains daily closing daily prices across all tickers
*open_px.csv* contains daily open prices across all tickers 
*volume.csv* contains daily traded volume across all tickers 
*rets_close.csv* contains daily returns computed from closing prices across all tickers
*adv20.csv* contains the daily 20-day average daily volume prices across all tickers

The data pipeline is split into three stages to keep each step auditable and easy to debug:

- *data/raw* for raw input files from yfinance api
- *data/interim* for cleaned per-ticker files (standardized columns, missing data handling + validation)
- 


*data/cleaned* for final wide-format datasets used directly by the backtest

This staged structure provides more granular control over data cleaning and makes it easier to modify or troubleshoot individual steps, compared to transforming raw files directly into final backtest inputs with no intermediate outputs.

## Single generation (Momentum filter) - signals.py
**quarter_end_dates()** *Helper function for *quarterly_rebalance_dates()* *
Takes a pd.Series of dates (type: DatetimeIndex). groups them by quarter and takes the last date of each quarter.

**quarterly_rebalance_dates()**
Given a series of dates (for a vectorised backtest), returns the trading days that are at the end of each quarter. These dates are used as the rebalancing dates.

**_month_shift_index** *Helper function for /momentum score functions*

**momentum_12m_minus_1()**

**select_topN()**

**blened_rank_6_12()**

**blended_rank_3_6_12()**
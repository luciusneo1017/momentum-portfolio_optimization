#need close and open, signals use close, rebalancing trade at the next open
import pandas as pd
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path when executed as a script
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Prefer package-relative import; fall back to absolute when run as a script
try:
    from .paths import RAW, INTERIM, CLEANED
except ImportError:  # script execution without -m
    from src.momentumhk.paths import RAW, INTERIM, CLEANED

def preprocess_data():

    INTERIM_data_dir = INTERIM
    INTERIM_data_dir.mkdir(parents=True, exist_ok=True)

    for file in os.listdir(RAW):
        df = pd.read_csv(RAW / file)
        
        df.columns =['Date', 'Close', 'High', 'Low', 'Open', 'Volume'] # setting columns which occur in this order
        df = df.iloc[2:].reset_index(drop=True)
        # df = df[['Date','Close','Open','Volume']]

    # forward fill missing price data, bfill() incase first row is nan
        df[['Close','High','Low','Open','Volume']] = df[['Close','High','Low','Open','Volume']].ffill().bfill()

        try:
            assert df['Close'].isna().sum() == 0, "There is missing data in 'Close' series"
            assert df['High'].isna().sum() == 0, "There is missing data in 'High' series"
            assert df['Low'].isna().sum() == 0, "There is missing data in 'Low' series"
            assert df['Open'].isna().sum() == 0, "There is missing data in 'Open' series"
            assert df['Volume'].isna().sum() == 0, "There is missing data in 'Volume' series"

        except AssertionError as e:
            print(f'File name {file} caught an assertion error {e}')

        
        df.to_csv(INTERIM_data_dir / file, index = False)

    # catch for ensuring all files in data_dir are cleaned and passed to clean_data_dir 
    assert len(os.listdir(RAW)) == len(os.listdir(INTERIM_data_dir)), "# of files in data dir do not match # of files in clean data dir"




# ---------- build wide outputs from INTERIM and save to CLEANED ----------

def build_wide_and_save():

    source = INTERIM
    destination = CLEANED
    destination.mkdir(parents=True, exist_ok=True)

    parts = {}  # per-ticker OHLCV with Date index
    for f in sorted(source.glob("*.csv")):
        ticker = f.stem
        df = pd.read_csv(f, parse_dates=["Date"])
        if df.empty:
            continue

        # to validate that we only retrive the columns we need
        # df = df[["Date", "Close", "High", "Low", "Open", "Volume"]]
        # drop rows with id data is nan (if any), sort them and set them as the dataframe index
        df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
        # remove duplicate index entries and keep the row with the last one, drops earlier entries
        df = df[~df.index.duplicated(keep="last")]

        # the columns we need
        parts[ticker] = df[["Open", "Close", "Volume"]]

    if not parts:
        raise FileNotFoundError(f"No cleaned CSVs found in {source}")

    # concat side-by-side : (ticker, field), then flip to (field, ticker)
    wide = pd.concat(parts, axis=1).sort_index()
    wide = wide.swaplevel(axis=1).sort_index(axis=1)
    wide.columns = wide.columns.set_names(["field", "ticker"])

    prices  = wide["Close"]
    open_px = wide["Open"].reindex_like(prices)  # ensure we have the same index as prices as reference
    volume  = wide["Volume"].reindex_like(prices)

    # Derived tables
    rets_close = prices.pct_change()
    adv20 = (prices * volume).rolling(20, min_periods=10).mean()

    # Save CSVs (dates as index named 'date')
    def _to_csv(df: pd.DataFrame, path: Path):
        out = df.copy()
        out.index.name = "date"
        out.to_csv(path)

    _to_csv(prices, destination / "prices.csv")
    _to_csv(open_px, destination / "open.csv")
    _to_csv(volume, destination / "volume.csv")
    _to_csv(rets_close, destination / "rets_close.csv")
    _to_csv(adv20, destination / "adv20.csv")

    # universe snapshot
    pd.DataFrame({"ticker": prices.columns}).to_csv(destination / "universe_current.csv", index=False)

    print(f"Built wide panel from {source} and wrote cleaned CSVs to {destination}")

if __name__ == "__main__":
    preprocess_data()
    build_wide_and_save()
# only execute when i run this file directly
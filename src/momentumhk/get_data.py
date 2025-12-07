import os
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

# Ensure repo root is on sys.path when executed as a script
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Prefer package-relative import; fall back to absolute when run as a script
try:
    from .paths import RAW
except ImportError:
    from src.momentumhk.paths import RAW


def load_constituents(file_name: str = "HSI Composite List.xlsx") -> pd.Series:
    excel_path = Path(__file__).resolve().parent / file_name
    hsic = pd.read_excel(excel_path)
    tickers = hsic["Stock Code"]
    # Drop non-int codes (e.g., blanks/notes)
    tickers = tickers[tickers.apply(lambda x: isinstance(x, int))]
    tickers.index = range(len(tickers))
    return tickers


def create_ticker(code: int) -> str:
    """Format numeric code to zero-padded 4-digit HK ticker."""
    ticker = str(code).strip()
    if len(ticker) == 4:
        return f"{ticker}.HK"

    padded = ticker.zfill(4)
    return f"{padded}.HK"


def main() -> None:
    tickers = load_constituents()

    HSIC_constituents_n = 503
    assert len(tickers) == HSIC_constituents_n, (
        f"Expected {HSIC_constituents_n} constituents, got {len(tickers)}"
    )

    tickers = tickers.apply(create_ticker).to_list()

    start_date = "2009-01-01"
    end_date = "2025-10-31"

    RAW.mkdir(parents=True, exist_ok=True)

    for t in tickers:
        try:
            df = yf.download(
                t,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=True,
                keepna=True,
            )
        except Exception as e:
            print(f"Error downloading data for {t}: {e}")
            continue

        file_path = RAW / f"{t}.csv"
        df.to_csv(file_path)


if __name__ == "__main__":
    main()

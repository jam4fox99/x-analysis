import pandas as pd
import yfinance as yf
from datetime import timedelta
from alpha_vantage.timeseries import TimeSeries
import os
from dotenv import load_dotenv

load_dotenv()

def fetch_intraday_data(symbol, start_date, end_date):
    """
    Fetches 1-minute interval data using Alpha Vantage.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        api_key = input("Please enter your Alpha Vantage API key: ")
        with open(".env", "w") as f:
            f.write(f"ALPHA_VANTAGE_API_KEY={api_key}\n")

    ts = TimeSeries(key=api_key, output_format='pandas')
    
    try:
        data, meta_data = ts.get_intraday(symbol=symbol.upper(), interval='1min', outputsize='full')
        data = data.rename(columns={'4. close': 'close'})
        data.index = pd.to_datetime(data.index)
        data = data.tz_localize('America/New_York').tz_convert('UTC')
        data.sort_index(inplace=True)
        return data[['close']]
    except Exception as e:
        print(f"Could not download data for {symbol}. Error: {e}")
        return None

def main():
    """
    Calculates ROI for trades based on entry/exit signals.
    """
    try:
        signals_df = pd.read_csv("signals.csv", parse_dates=['created_at'])
    except FileNotFoundError:
        print("Error: signals.csv not found. Please run main.py to generate it.")
        return

    signals_df['created_at'] = pd.to_datetime(signals_df['created_at'])
    signals_df.sort_values(by='created_at', inplace=True)

    price_data_cache = {}
    completed_trades = []
    open_positions = {}

    for index, signal in signals_df.iterrows():
        ticker = signal['ticker'].replace('$', '')
        action = signal['action']
        signal_time = signal['created_at']

        if ticker not in price_data_cache:
            print(f"Fetching intraday data for {ticker}...")
            price_data_cache[ticker] = fetch_intraday_data(ticker, None, None)
        
        price_data = price_data_cache.get(ticker)
        if price_data is None:
            continue

        if action == 'entry':
            if ticker not in open_positions:
                open_positions[ticker] = {'entry_time': signal_time}
                print(f"Opened position for {ticker} at {signal_time}")
        
        elif action == 'exit' and ticker in open_positions:
            entry_info = open_positions.pop(ticker)
            entry_time = entry_info['entry_time']
            
            entry_price_series = price_data.asof(entry_time)
            exit_price_series = price_data.asof(signal_time)

            entry_price = entry_price_series.iloc[0] if isinstance(entry_price_series, pd.Series) and not entry_price_series.empty else None
            exit_price = exit_price_series.iloc[0] if isinstance(exit_price_series, pd.Series) and not exit_price_series.empty else None

            if pd.notna(entry_price) and pd.notna(exit_price):
                roi = ((exit_price - entry_price) / entry_price) * 100 if entry_price != 0 else 0
                completed_trades.append({
                    'Ticker': ticker,
                    'Entry Time (UTC)': entry_time,
                    'Entry Price': entry_price,
                    'Exit Time (UTC)': signal_time,
                    'Exit Price': exit_price,
                    'ROI (%)': roi
                })
                print(f"  -> Trade completed for {ticker}. ROI: {roi:.2f}%")
            else:
                print(f"  -> Could not find price data for trade on {ticker} between {entry_time} and {signal_time}.")

    for ticker, entry_info in open_positions.items():
        price_data = price_data_cache.get(ticker)
        if price_data is not None and not price_data.empty:
            entry_time = entry_info['entry_time']
            entry_price_series = price_data.asof(entry_time)
            entry_price = entry_price_series.iloc[0] if isinstance(entry_price_series, pd.Series) and not entry_price_series.empty else None

            if pd.notna(entry_price):
                last_timestamp = price_data.index[-1]
                last_price = price_data.iloc[-1]['close']
                roi = ((last_price - entry_price) / entry_price) * 100 if entry_price != 0 else 0
                completed_trades.append({
                    'Ticker': ticker,
                    'Entry Time (UTC)': entry_time,
                    'Entry Price': entry_price,
                    'Exit Time (UTC)': last_timestamp,
                    'Exit Price': last_price,
                    'ROI (%)': roi
                })
                print(f"  -> Open position for {ticker} analyzed with holding ROI: {roi:.2f}%")

    if completed_trades:
        results_df = pd.DataFrame(completed_trades)
        results_df.sort_values(by='Entry Time (UTC)', inplace=True)
        results_df.to_csv("roi_analysis.csv", index=False)
        print("\nROI analysis complete. Results saved to 'roi_analysis.csv'.")
    else:
        print("\nNo completed trades to analyze.")

if __name__ == '__main__':
    main() 
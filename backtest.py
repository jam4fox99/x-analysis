import os
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv
from collections import deque

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

def fetch_intraday_data(symbol):
    """Fetches 1-minute interval data using Alpha Vantage."""
    if not ALPHA_VANTAGE_API_KEY:
        print("Error: ALPHA_VANTAGE_API_KEY not set.")
        return None

    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    
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

def run_backtest(signals_df, mode='all_entries'):
    """
    Runs a backtest simulation based on the specified mode.
    - 'all_entries': Treats every entry as a new trade (FIFO).
    - 'first_entry_only': Considers only the first entry for each ticker.
    """
    print(f"\n--- Running backtest in '{mode}' mode ---")
    
    price_data_cache = {}
    completed_trades = []
    
    # Use a deque for FIFO logic in 'all_entries' mode
    open_positions = {ticker.replace('$', ''): deque() for ticker in signals_df['ticker'].unique()}
    
    # For 'first_entry_only' mode
    processed_first_entry = set()

    for _, signal in signals_df.iterrows():
        ticker = signal['ticker'].replace('$', '') # Clean the ticker symbol FIRST
        action = signal['action']
        signal_time = signal['created_at']

        # Fetch data if not cached
        if ticker not in price_data_cache:
            print(f"Fetching intraday data for {ticker}...")
            price_data_cache[ticker] = fetch_intraday_data(ticker)
        
        price_data = price_data_cache.get(ticker)
        if price_data is None or price_data.empty:
            continue

        if action == 'entry':
            if mode == 'all_entries':
                open_positions[ticker].append(signal_time)
                print(f"  -> Opened new position for {ticker} at {signal_time}")
            elif mode == 'first_entry_only':
                if ticker not in processed_first_entry:
                    open_positions[ticker].append(signal_time)
                    processed_first_entry.add(ticker)
                    print(f"  -> Opened first and only position for {ticker} at {signal_time}")

        elif action == 'exit' and open_positions.get(ticker):
            entry_time = open_positions[ticker].popleft() # FIFO
            
            entry_price_series = price_data.asof(entry_time)
            exit_price_series = price_data.asof(signal_time)
            
            entry_price = entry_price_series.iloc[0] if isinstance(entry_price_series, pd.Series) and not entry_price_series.empty else None
            exit_price = exit_price_series.iloc[0] if isinstance(exit_price_series, pd.Series) and not exit_price_series.empty else None

            if pd.notna(entry_price) and pd.notna(exit_price) and entry_price != 0:
                roi = ((exit_price - entry_price) / entry_price) * 100
                completed_trades.append({
                    'Ticker': ticker, 'Entry Time': entry_time, 'Entry Price': entry_price,
                    'Exit Time': signal_time, 'Exit Price': exit_price, 'ROI (%)': roi
                })
                print(f"  -> Trade completed for {ticker}. ROI: {roi:.2f}%")

    # For any remaining open positions, calculate holding ROI
    for ticker, entries in open_positions.items():
        price_data = price_data_cache.get(ticker)
        if price_data is not None and not price_data.empty:
            for entry_time in entries:
                entry_price_series = price_data.asof(entry_time)
                entry_price = entry_price_series.iloc[0] if isinstance(entry_price_series, pd.Series) and not entry_price_series.empty else None

                if pd.notna(entry_price) and entry_price != 0:
                    last_price = price_data['close'].iloc[-1]
                    roi = ((last_price - entry_price) / entry_price) * 100
                    completed_trades.append({
                        'Ticker': ticker, 'Entry Time': entry_time, 'Entry Price': entry_price,
                        'Exit Time': price_data.index[-1], 'Exit Price': last_price, 'ROI (%)': roi
                    })
                    print(f"  -> Open position for {ticker} analyzed with holding ROI: {roi:.2f}%")
                    
    return pd.DataFrame(completed_trades)

def main():
    """
    Main function to load signals and run both backtesting simulations.
    """
    try:
        signals_df = pd.read_csv("signals.csv", parse_dates=['created_at'])
        signals_df.sort_values(by='created_at', inplace=True)
    except FileNotFoundError:
        print("Error: signals.csv not found. Please run main.py to generate signals first.")
        return

    # --- Run Simulation 1: All Entries (FIFO) ---
    all_entries_results = run_backtest(signals_df, mode='all_entries')
    if not all_entries_results.empty:
        output_filename = "roi_analysis_all_entries.csv"
        all_entries_results.sort_values(by='Entry Time', inplace=True)
        all_entries_results.to_csv(output_filename, index=False)
        print(f"\n'All Entries' simulation complete. Results saved to '{output_filename}'.")

    # --- Run Simulation 2: First Entry Only ---
    first_entry_results = run_backtest(signals_df, mode='first_entry_only')
    if not first_entry_results.empty:
        output_filename = "roi_analysis_first_entry_only.csv"
        first_entry_results.sort_values(by='Entry Time', inplace=True)
        first_entry_results.to_csv(output_filename, index=False)
        print(f"\n'First Entry Only' simulation complete. Results saved to '{output_filename}'.")

if __name__ == '__main__':
    main() 
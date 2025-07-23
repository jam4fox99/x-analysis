import os
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv
from collections import deque
from datetime import timedelta

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

def fetch_daily_data(symbol):
    """Fetches daily adjusted data using Alpha Vantage for a robust backtest period."""
    if not ALPHA_VANTAGE_API_KEY:
        print("Error: ALPHA_VANTAGE_API_KEY not set.")
        return None

    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    
    try:
        data, meta_data = ts.get_daily_adjusted(symbol=symbol.upper(), outputsize='full')
        data = data.rename(columns={'4. close': 'close', '1. open': 'open'})
        data.index = pd.to_datetime(data.index)
        data = data.tz_localize('America/New_York').tz_convert('UTC')
        data.sort_index(inplace=True)
        return data[['close']]
    except Exception as e:
        print(f"Could not download daily data for {symbol}. Error: {e}")
        return None

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

def run_backtest(signals_df, mode='all_entries', max_hold_days=45):
    """
    Runs a backtest simulation based on the specified mode.
    - 'all_entries': Treats every entry as a new trade (FIFO).
    - 'first_entry_only': Considers only the first entry for each ticker.
    - Includes a time-stop exit for trades open longer than max_hold_days.
    """
    print(f"\n--- Running backtest in '{mode}' mode with a {max_hold_days}-day time stop ---")
    
    price_data_cache = {}
    completed_trades = []
    
    # Ensure all tickers are strings before creating the dictionary
    unique_tickers = [str(ticker).replace('$', '') for ticker in signals_df['ticker'].unique()]
    # Each ticker will have a deque to store open positions (long or short)
    open_positions = {ticker: deque() for ticker in unique_tickers}
    
    for _, signal in signals_df.iterrows():
        # Ensure the ticker from the signal is also treated as a string
        ticker = str(signal['ticker']).replace('$', '')
        action = signal['action']
        signal_time = signal['created_at']

        # Fetch data if not cached
        if ticker not in price_data_cache:
            print(f"Fetching daily data for {ticker}...")
            price_data_cache[ticker] = fetch_daily_data(ticker)
        
        price_data = price_data_cache.get(ticker)
        if price_data is None or price_data.empty:
            continue

        position_type_map = {'entry': 'long', 'short': 'short'}
        if action in position_type_map:
            if mode == 'all_entries':
                open_positions[ticker].append({'time': signal_time, 'type': position_type_map[action]})
                print(f"  -> Opened new {position_type_map[action]} position for {ticker} at {signal_time}")
            elif mode == 'first_entry_only' and not open_positions.get(ticker):
                open_positions[ticker].append({'time': signal_time, 'type': position_type_map[action]})
                print(f"  -> Opened first and only {position_type_map[action]} position for {ticker} at {signal_time}")

        elif action == 'exit' and open_positions.get(ticker):
            opened_position = open_positions[ticker].popleft() # FIFO
            entry_time = opened_position['time']
            position_type = opened_position['type']
            
            entry_price_series = price_data.asof(entry_time)
            exit_price_series = price_data.asof(signal_time)
            
            entry_price = entry_price_series.iloc[0] if isinstance(entry_price_series, pd.Series) and not entry_price_series.empty else None
            exit_price = exit_price_series.iloc[0] if isinstance(exit_price_series, pd.Series) and not exit_price_series.empty else None

            if pd.notna(entry_price) and pd.notna(exit_price) and entry_price != 0:
                if position_type == 'long':
                    roi = ((exit_price - entry_price) / entry_price) * 100
                else: # Short position
                    roi = ((entry_price - exit_price) / entry_price) * 100
                
                completed_trades.append({
                    'Ticker': ticker, 'Position Type': position_type,
                    'Entry Time': entry_time, 'Entry Price': entry_price,
                    'Exit Time': signal_time, 'Exit Price': exit_price, 'ROI (%)': roi
                })
                print(f"  -> {position_type.capitalize()} trade completed for {ticker}. ROI: {roi:.2f}%")

    # For any remaining open positions, calculate holding ROI based on the time-stop
    print(f"\n--- Analyzing open positions with a {max_hold_days}-day time-stop exit ---")
    for ticker, entries in open_positions.items():
        price_data = price_data_cache.get(ticker)
        if price_data is not None and not price_data.empty:
            for open_pos in entries:
                entry_time = open_pos['time']
                position_type = open_pos['type']
                
                entry_price_series = price_data.asof(entry_time)
                entry_price = entry_price_series.iloc[0] if isinstance(entry_price_series, pd.Series) and not entry_price_series.empty else None

                if pd.notna(entry_price) and entry_price != 0:
                    # Calculate the mandatory exit date based on the holding period
                    time_stop_exit_date = entry_time + pd.Timedelta(days=max_hold_days)
                    
                    # Find the price at the time-stop date
                    exit_price_series = price_data.asof(time_stop_exit_date)
                    
                    if exit_price_series is not None and not exit_price_series.empty:
                        exit_price = exit_price_series.iloc[0]
                        exit_time = exit_price_series.name
                    else:
                        # Fallback to the last known price if no data exists up to the time-stop date
                        exit_price = price_data['close'].iloc[-1]
                        exit_time = price_data.index[-1]

                    if position_type == 'long':
                        roi = ((exit_price - entry_price) / entry_price) * 100
                    else: # Short position
                        roi = ((entry_price - exit_price) / entry_price) * 100

                    completed_trades.append({
                        'Ticker': ticker, 'Position Type': position_type, 
                        'Entry Time': entry_time, 'Entry Price': entry_price,
                        'Exit Time': exit_time, 'Exit Price': exit_price, 'ROI (%)': roi
                    })
                    print(f"  -> Open {position_type} position for {ticker} time-stopped after {max_hold_days} days. ROI: {roi:.2f}%")
                    
    return pd.DataFrame(completed_trades)

def main():
    """
    Main function to load signals and run both backtesting simulations.
    """
    try:
        signals_df = pd.read_csv("signals.csv", parse_dates=['created_at'], dtype={'ticker': str})
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
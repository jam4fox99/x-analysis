import os
import pandas as pd
from dotenv import load_dotenv
from alpha_vantage.timeseries import TimeSeries
from tqdm import tqdm
from datetime import datetime, timedelta
from collections import deque
import numpy as np

# Load environment variables
load_dotenv()

# --- Configuration ---
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
MAX_HOLD_DAYS = 45

# --- Data Caching ---
price_data_cache = {}
failed_tickers = set()

def fetch_daily_data(symbol):
    symbol = str(symbol).replace('$', '')
    if symbol in price_data_cache:
        return price_data_cache[symbol]
    if symbol in failed_tickers:
        return None
    
    if not ALPHA_VANTAGE_API_KEY:
        print("Error: ALPHA_VANTAGE_API_KEY not set.")
        return None

    print(f"Fetching daily data for {symbol}...")
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    
    try:
        data, _ = ts.get_daily_adjusted(symbol=symbol.upper(), outputsize='full')
        data.rename(columns={
            '1. open': 'open', '2. high': 'high', 
            '3. low': 'low', '4. close': 'close',
            '5. adjusted close': 'adjusted_close'
        }, inplace=True)
        data.index = pd.to_datetime(data.index)
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC')
        price_data_cache[symbol] = data.sort_index()
        return price_data_cache[symbol]
    except Exception as e:
        print(f"Could not download daily data for {symbol}. Error: {e}")
        failed_tickers.add(symbol)
        return None

def run_standard_backtest(signals_df):
    completed_trades = []
    open_positions = {ticker: deque() for ticker in signals_df['ticker'].unique()}

    for _, signal in tqdm(signals_df.iterrows(), total=len(signals_df), desc="Running Standard Backtest"):
        ticker = signal['ticker']
        action = signal['action']
        signal_time = signal['created_at']

        price_data = fetch_daily_data(ticker)
        if price_data is None: continue
        
        position_type_map = {'entry': 'long', 'short': 'short'}
        if action in position_type_map:
            entry_price_series = price_data.asof(signal_time)
            if not entry_price_series.empty:
                entry_price = entry_price_series['close']
                open_positions[ticker].append({
                    'Entry Time': signal_time, 
                    'Entry Price': entry_price,
                    'Position Type': position_type_map[action]
                })

        elif action == 'exit' and open_positions[ticker]:
            opened_position = open_positions[ticker].popleft()
            exit_price_series = price_data.asof(signal_time)
            if not exit_price_series.empty:
                exit_price = exit_price_series['close']
                if opened_position['Position Type'] == 'long':
                    roi = ((exit_price - opened_position['Entry Price']) / opened_position['Entry Price']) * 100
                else:
                    roi = ((opened_position['Entry Price'] - exit_price) / opened_position['Entry Price']) * 100
                completed_trades.append({
                    'Ticker': ticker, **opened_position,
                    'Exit Time': signal_time, 'Exit Price': exit_price, 'ROI (%)': roi
                })

    # Handle remaining open positions with a time-stop
    for ticker, entries in open_positions.items():
        price_data = price_data_cache.get(ticker)
        if price_data is None: continue
        for open_pos in entries:
            entry_time = open_pos['Entry Time']
            exit_time = entry_time + timedelta(days=MAX_HOLD_DAYS)
            exit_price_series = price_data.asof(exit_time)
            if not exit_price_series.empty:
                exit_price = exit_price_series['close']
                if open_pos['Position Type'] == 'long':
                    roi = ((exit_price - open_pos['Entry Price']) / open_pos['Entry Price']) * 100
                else:
                    roi = ((open_pos['Entry Price'] - exit_price) / open_pos['Entry Price']) * 100
                completed_trades.append({
                    'Ticker': ticker, **open_pos,
                    'Exit Time': exit_time, 'Exit Price': exit_price, 'ROI (%)': roi
                })
    return pd.DataFrame(completed_trades)

def remove_outliers(df, column='ROI (%)'):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df, outliers

def print_performance_summary(df, title):
    print(f"\n--- {title} ---")
    if df.empty:
        print("No trades to analyze.")
        return
    
    avg_roi = df['ROI (%)'].mean()
    win_rate = (df['ROI (%)'] > 0).sum() / len(df) * 100
    total_trades = len(df)
    
    print(f"Total Trades: {total_trades}")
    print(f"Average ROI: {avg_roi:.2f}%")
    print(f"Win Rate: {win_rate:.2f}%")

def main():
    try:
        signals_df = pd.read_csv("signals.csv", parse_dates=['created_at'], dtype={'ticker': str}).sort_values(by='created_at')
    except FileNotFoundError:
        print("Error: signals.csv not found."); return

    all_trades = run_standard_backtest(signals_df)
    
    print_performance_summary(all_trades, "Performance Before Outlier Removal")
    
    filtered_trades, removed_outliers = remove_outliers(all_trades)
    
    print(f"\nRemoved {len(removed_outliers)} outliers.")
    
    print_performance_summary(filtered_trades, "Performance After Outlier Removal")
    
    all_trades.to_csv("full_backtest_results.csv", index=False)
    filtered_trades.to_csv("filtered_backtest_results.csv", index=False)
    
    print("\nSaved full results to 'full_backtest_results.csv'")
    print("Saved filtered results to 'filtered_backtest_results.csv'")

if __name__ == "__main__":
    main() 
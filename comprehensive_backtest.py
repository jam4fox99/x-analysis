import os
import pandas as pd
from dotenv import load_dotenv
from alpha_vantage.timeseries import TimeSeries
from tqdm import tqdm
from datetime import datetime
import numpy as np
from collections import deque

# Load environment variables
load_dotenv()

# --- Configuration ---
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
TP_GRID = np.arange(0.05, 2.05, 0.05)
SL_GRID = np.arange(0.05, 1.05, 0.05)

# --- Data Caching ---
price_data_cache = {}
failed_tickers = set()

def fetch_daily_data(symbol):
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

def process_signals_to_trades(signals_df):
    completed_trades = []
    open_positions = {ticker: deque() for ticker in signals_df['ticker'].unique()}

    for _, signal in tqdm(signals_df.iterrows(), total=len(signals_df), desc="Processing Signals into Trades"):
        ticker = signal['ticker']
        action = signal['action']
        signal_time = signal['created_at']

        price_data = fetch_daily_data(ticker)
        if price_data is None:
            continue
        
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
            opened_position = open_positions[ticker].popleft() # FIFO
            exit_price_series = price_data.asof(signal_time)
            if not exit_price_series.empty:
                exit_price = exit_price_series['close']
                
                if opened_position['Position Type'] == 'long':
                    roi = ((exit_price - opened_position['Entry Price']) / opened_position['Entry Price']) * 100
                else: # Short
                    roi = ((opened_position['Entry Price'] - exit_price) / opened_position['Entry Price']) * 100

                completed_trades.append({
                    'Ticker': ticker,
                    **opened_position,
                    'Exit Time': signal_time,
                    'Exit Price': exit_price,
                    'ROI (%)': roi
                })

    open_trades = []
    for ticker, positions in open_positions.items():
        for pos in positions:
            open_trades.append({'Ticker': ticker, **pos})

    return pd.DataFrame(completed_trades), pd.DataFrame(open_trades)

def run_grid_search(trades_df):
    best_avg_roi = -np.inf
    optimal_params = (None, None)
    
    param_grid = [(tp, sl) for tp in TP_GRID for sl in SL_GRID]

    for tp, sl in tqdm(param_grid, desc="Optimizing TP/SL"):
        total_roi = 0
        for _, trade in trades_df.iterrows():
            price_data = fetch_daily_data(trade['Ticker'])
            if price_data is None:
                total_roi += trade['ROI (%)']
                continue
            
            entry_price = trade['Entry Price']
            entry_time = pd.to_datetime(trade['Entry Time'])
            exit_time = pd.to_datetime(trade['Exit Time'])
            position_type = trade['Position Type']

            sim_period = price_data[entry_time:exit_time]
            
            hit_tp, hit_sl = False, False
            for _, row in sim_period.iterrows():
                if position_type == 'long':
                    if row['high'] >= entry_price * (1 + tp / 100):
                        total_roi += tp; hit_tp = True; break
                    if row['low'] <= entry_price * (1 - sl / 100):
                        total_roi -= sl; hit_sl = True; break
                elif position_type == 'short':
                    if row['low'] <= entry_price * (1 - tp / 100):
                        total_roi += tp; hit_tp = True; break
                    if row['high'] >= entry_price * (1 + sl / 100):
                        total_roi -= sl; hit_sl = True; break
            
            if not hit_tp and not hit_sl:
                total_roi += trade['ROI (%)']
        
        current_avg_roi = total_roi / len(trades_df) if not trades_df.empty else 0
        if current_avg_roi > best_avg_roi:
            best_avg_roi = current_avg_roi
            optimal_params = (tp, sl)
            
    return optimal_params

def backtest_open_trades(open_trades_df, optimal_tp, optimal_sl):
    results = []
    for _, trade in tqdm(open_trades_df.iterrows(), total=len(open_trades_df), desc="Backtesting Open Trades"):
        price_data = fetch_daily_data(trade['Ticker'])
        if price_data is None: continue
        
        entry_price = trade['Entry Price']
        entry_time = pd.to_datetime(trade['Entry Time'])
        position_type = trade['Position Type']

        sim_period = price_data[entry_time:]
        
        exit_reason, exit_price, exit_time, roi = "Still Open", None, None, None

        for idx, row in sim_period.iterrows():
            if position_type == 'long':
                if row['high'] >= entry_price * (1 + optimal_tp / 100):
                    exit_price = entry_price * (1 + optimal_tp / 100); exit_time = idx; roi = optimal_tp; exit_reason = "Take Profit Hit"; break
                if row['low'] <= entry_price * (1 - optimal_sl / 100):
                    exit_price = entry_price * (1 - optimal_sl / 100); exit_time = idx; roi = -optimal_sl; exit_reason = "Stop Loss Hit"; break
            elif position_type == 'short':
                if row['low'] <= entry_price * (1 - optimal_tp / 100):
                    exit_price = entry_price * (1 - optimal_tp / 100); exit_time = idx; roi = optimal_tp; exit_reason = "Take Profit Hit"; break
                if row['high'] >= entry_price * (1 + optimal_sl / 100):
                    exit_price = entry_price * (1 + optimal_sl / 100); exit_time = idx; roi = -optimal_sl; exit_reason = "Stop Loss Hit"; break

        if exit_reason == "Still Open" and not sim_period.empty:
            last_price = sim_period.iloc[-1]['close']
            roi = ((last_price - entry_price) / entry_price) * 100 if position_type == 'long' else ((entry_price - last_price) / entry_price) * 100
            exit_price, exit_time = last_price, sim_period.index[-1]

        results.append({
            'Ticker': trade['Ticker'], 'Position Type': position_type,
            'Entry Time': trade['Entry Time'], 'Entry Price': entry_price,
            'Exit Time': exit_time, 'Exit Price': exit_price, 'ROI (%)': roi,
            'Exit Reason': exit_reason
        })
    return pd.DataFrame(results)

def main():
    try:
        signals_df = pd.read_csv("signals.csv", parse_dates=['created_at'], dtype={'ticker': str}).sort_values(by='created_at')
    except FileNotFoundError:
        print("Error: signals.csv not found."); return

    completed_trades, open_trades = process_signals_to_trades(signals_df)
    
    if completed_trades.empty:
        print("No completed trades found to optimize parameters."); return

    first_entry_completed = completed_trades.loc[completed_trades.groupby('Ticker')['Entry Time'].idxmin()]
    
    optimal_tp, optimal_sl = run_grid_search(first_entry_completed)
    print(f"\nOptimal Parameters Found: TP={optimal_tp:.2f}%, SL={optimal_sl:.2f}%")

    optimized_open_trades = backtest_open_trades(open_trades, optimal_tp, optimal_sl)
    
    final_report = pd.concat([completed_trades, optimized_open_trades], ignore_index=True).sort_values(by='Entry Time')
    
    final_report.to_csv("comprehensive_roi_analysis.csv", index=False)
    print("\nComprehensive backtest complete. Results saved to 'comprehensive_roi_analysis.csv'.")

if __name__ == "__main__":
    main() 
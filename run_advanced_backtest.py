import os
import pandas as pd
from dotenv import load_dotenv
from alpha_vantage.timeseries import TimeSeries
from tqdm import tqdm
from datetime import datetime, timedelta
import numpy as np

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
        price_data_cache[symbol] = data
        return data
    except Exception as e:
        print(f"Could not download daily data for {symbol}. Error: {e}")
        failed_tickers.add(symbol)
        return None

def separate_trades_by_exit_type(input_file):
    df = pd.read_csv(input_file)
    df['Entry Time'] = pd.to_datetime(df['Entry Time'])
    df['Exit Time'] = pd.to_datetime(df['Exit Time'])
    df['Holding Period'] = df['Exit Time'] - df['Entry Time']
    
    user_exited_trades = df[df['Holding Period'] < timedelta(days=45)]
    timed_out_trades = df[df['Holding Period'] >= timedelta(days=45)]
    
    return user_exited_trades, timed_out_trades

def run_grid_search_on_user_exits(trades_df):
    best_avg_roi = -np.inf
    optimal_params = (None, None)
    
    param_grid = [(tp, sl) for tp in TP_GRID for sl in SL_GRID]

    for tp, sl in tqdm(param_grid, desc="Optimizing TP/SL on User Exits"):
        total_roi = 0
        trade_count = len(trades_df)

        for _, trade in trades_df.iterrows():
            price_data = fetch_daily_data(trade['Ticker'])
            if price_data is None:
                total_roi += trade['ROI (%)']
                continue
            
            entry_price = trade['Entry Price']
            entry_time = pd.to_datetime(trade['Entry Time'])
            exit_time = pd.to_datetime(trade['Exit Time'])
            position_type = trade['Position Type']

            simulation_period = price_data[entry_time:exit_time]
            
            hit_tp = False
            hit_sl = False

            for _, row in simulation_period.iterrows():
                if position_type == 'long':
                    if row['high'] >= entry_price * (1 + tp / 100):
                        total_roi += tp
                        hit_tp = True
                        break
                    if row['low'] <= entry_price * (1 - sl / 100):
                        total_roi -= sl
                        hit_sl = True
                        break
                elif position_type == 'short':
                    if row['low'] <= entry_price * (1 - tp / 100):
                        total_roi += tp
                        hit_tp = True
                        break
                    if row['high'] >= entry_price * (1 + sl / 100):
                        total_roi -= sl
                        hit_sl = True
                        break
            
            if not hit_tp and not hit_sl:
                total_roi += trade['ROI (%)']
        
        current_avg_roi = total_roi / trade_count if trade_count > 0 else 0
        if current_avg_roi > best_avg_roi:
            best_avg_roi = current_avg_roi
            optimal_params = (tp, sl)
            
    return optimal_params

def run_final_backtest(timed_out_trades_df, optimal_tp, optimal_sl):
    final_results = []

    for _, trade in tqdm(timed_out_trades_df.iterrows(), total=len(timed_out_trades_df), desc="Backtesting Timed-Out Trades"):
        price_data = fetch_daily_data(trade['Ticker'])
        if price_data is None:
            continue
        
        entry_price = trade['Entry Price']
        entry_time = pd.to_datetime(trade['Entry Time'])
        position_type = trade['Position Type']

        simulation_period = price_data[entry_time:]
        
        exit_reason = "Still Open"
        exit_price = None
        exit_time = None
        roi = None

        for idx, row in simulation_period.iterrows():
            if position_type == 'long':
                if row['high'] >= entry_price * (1 + optimal_tp / 100):
                    exit_price = entry_price * (1 + optimal_tp / 100)
                    exit_time = idx
                    roi = optimal_tp
                    exit_reason = "Take Profit Hit"
                    break
                if row['low'] <= entry_price * (1 - optimal_sl / 100):
                    exit_price = entry_price * (1 - optimal_sl / 100)
                    exit_time = idx
                    roi = -optimal_sl
                    exit_reason = "Stop Loss Hit"
                    break
            elif position_type == 'short':
                if row['low'] <= entry_price * (1 - optimal_tp / 100):
                    exit_price = entry_price * (1 - optimal_tp / 100)
                    exit_time = idx
                    roi = optimal_tp
                    exit_reason = "Take Profit Hit"
                    break
                if row['high'] >= entry_price * (1 + optimal_sl / 100):
                    exit_price = entry_price * (1 + optimal_sl / 100)
                    exit_time = idx
                    roi = -optimal_sl
                    exit_reason = "Stop Loss Hit"
                    break

        if exit_reason == "Still Open":
            last_price = simulation_period.iloc[-1]['close']
            if position_type == 'long':
                roi = ((last_price - entry_price) / entry_price) * 100
            else: # Short
                roi = ((entry_price - last_price) / entry_price) * 100
            exit_price = last_price
            exit_time = simulation_period.index[-1]

        final_results.append({
            'Ticker': trade['Ticker'],
            'Position Type': position_type,
            'Entry Time': trade['Entry Time'],
            'Entry Price': entry_price,
            'Exit Time': exit_time,
            'Exit Price': exit_price,
            'ROI (%)': roi,
            'Exit Reason': exit_reason
        })

    return pd.DataFrame(final_results)

def main():
    # Step 1: Separate trades
    user_exited_trades, timed_out_trades = separate_trades_by_exit_type('roi_analysis_first_entry_only.csv')
    
    # Step 2: Find optimal parameters
    optimal_tp, optimal_sl = run_grid_search_on_user_exits(user_exited_trades)
    
    # Step 3: Run final backtest
    final_results_df = run_final_backtest(timed_out_trades, optimal_tp, optimal_sl)
    
    # Step 4: Save results
    if not final_results_df.empty:
        final_results_df.to_csv("optimized_timed_out_trades.csv", index=False)
        print("\nAdvanced backtest complete. Results saved to 'optimized_timed_out_trades.csv'.")

if __name__ == "__main__":
    main() 
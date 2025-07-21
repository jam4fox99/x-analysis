import os
import pandas as pd
from dotenv import load_dotenv
from alpha_vantage.timeseries import TimeSeries
import numpy as np
from tqdm import tqdm

# Load environment variables
load_dotenv()

# --- Configuration ---
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
FORWARD_DAYS = 60  # Number of days to look forward for TP/SL

# --- Grid Search Parameters ---
TP_GRID = np.arange(0.05, 2.05, 0.05)  # 5% to 200%
SL_GRID = np.arange(0.05, 1.05, 0.05)  # 5% to 100%

# --- Data Caching ---
price_data_cache = {}
failed_tickers = set() # New set to track failed tickers

def fetch_daily_data(symbol):
    """Fetches daily historical data for a given symbol, with caching for failures."""
    if symbol in price_data_cache:
        return price_data_cache[symbol]
    if symbol in failed_tickers:
        return None # Immediately return None for known bad tickers
    
    if not ALPHA_VANTAGE_API_KEY:
        print("Error: ALPHA_VANTAGE_API_KEY not set.")
        return None

    print(f"Fetching daily data for {symbol}...")
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    
    try:
        data, _ = ts.get_daily(symbol=symbol.upper(), outputsize='full')
        data.rename(columns={
            '1. open': 'open', '2. high': 'high', 
            '3. low': 'low', '4. close': 'close'
        }, inplace=True)
        data.index = pd.to_datetime(data.index)
        price_data_cache[symbol] = data
        return data
    except Exception as e:
        print(f"Could not download daily data for {symbol}. Error: {e}")
        failed_tickers.add(symbol) # Add to our set of failed tickers
        return None

def run_simulation(signals_df, tp, sl):
    """
    Runs a single simulation for a given TP/SL combination.
    Returns average ROI, win rate, and the list of trades.
    """
    trades = []
    
    for _, signal in signals_df.iterrows():
        ticker = signal['ticker'].replace('$', '')
        entry_time = signal['created_at']
        
        daily_data = fetch_daily_data(ticker)
        if daily_data is None:
            continue
            
        forward_data = daily_data[daily_data.index.date >= entry_time.date()].head(FORWARD_DAYS)
        if forward_data.empty or len(forward_data) < 2: # Need at least entry and potential exit
            continue
            
        entry_price = forward_data['open'].iloc[0]
        if entry_price == 0: continue

        tp_price = entry_price * (1 + tp)
        sl_price = entry_price * (1 - sl)
        
        exit_price = None
        exit_reason = f"Held for {FORWARD_DAYS} days"
        exit_time = forward_data.index[-1]
        
        for idx, day in forward_data.iterrows():
            if day['high'] >= tp_price:
                exit_price = tp_price
                exit_reason = "Take-Profit Hit"
                exit_time = idx
                break
            if day['low'] <= sl_price:
                exit_price = sl_price
                exit_reason = "Stop-Loss Hit"
                exit_time = idx
                break
        
        if exit_price is None:
            exit_price = forward_data['close'].iloc[-1]
            
        trade_roi = ((exit_price - entry_price) / entry_price) * 100
        
        trades.append({
            'Ticker': ticker, 'Entry Time': entry_time, 'Entry Price': entry_price,
            'Exit Time': exit_time, 'Exit Price': exit_price, 'ROI (%)': trade_roi,
            'Exit Reason': exit_reason
        })
        
    if not trades:
        return 0, 0, []
        
    results_df = pd.DataFrame(trades)
    avg_roi = results_df['ROI (%)'].mean()
    win_rate = (results_df['ROI (%)'] > 0).sum() / len(results_df) * 100 if not results_df.empty else 0
    
    return avg_roi, win_rate, trades

def run_full_grid_search(signals_df):
    """Runs the grid search on a given dataframe and saves the results."""
    best_roi = -np.inf
    best_roi_params = (None, None)
    best_roi_trades = []

    best_win_rate = -1
    best_win_rate_params = (None, None)
    best_win_rate_trades = []
    
    param_grid = [(tp, sl) for tp in TP_GRID for sl in SL_GRID]
    
    for tp, sl in tqdm(param_grid, desc="Optimizing"):
        avg_roi, win_rate, trades = run_simulation(signals_df, tp, sl)
        
        if avg_roi > best_roi:
            best_roi = avg_roi
            best_roi_params = (tp, sl)
            best_roi_trades = trades
            
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            best_win_rate_params = (tp, sl)
            best_win_rate_trades = trades

    return best_roi_params, best_roi_trades, best_win_rate_params, best_win_rate_trades

def main():
    """Main function to run the grid search for dual objectives."""
    try:
        signals_df = pd.read_csv("signals.csv", parse_dates=['created_at'], dtype={'ticker': str})
        # Filter for entries and only the first one per ticker
        signals_df = signals_df[signals_df['action'] == 'entry']
        signals_df = signals_df.loc[signals_df.groupby('ticker')['created_at'].idxmin()]
        signals_df.sort_values(by='created_at', inplace=True) # Ensure chronological order
        print(f"Loaded {len(signals_df)} unique first-entry signals.")
    except FileNotFoundError:
        print("Error: signals.csv not found. Please run main.py first.")
        return

    # --- 1. Run Grid Search on the ENTIRE dataset (as before) ---
    print("\n--- Running Grid Search on ENTIRE dataset ---")
    full_best_roi_params, full_best_roi_trades, _, _ = run_full_grid_search(signals_df)
    
    if full_best_roi_trades:
        print(f"\nOptimal ROI Params (Full Dataset): TP={full_best_roi_params[0]:.2%}, SL={full_best_roi_params[1]:.2%}")
        pd.DataFrame(full_best_roi_trades).to_csv("optimal_roi_trades.csv", index=False)
        print("Saved full dataset optimal ROI trades to 'optimal_roi_trades.csv'")
    else:
        print("\nNo trades were completed in the full dataset simulation.")

    # --- 2. Split data for In-Sample / Out-of-Sample testing ---
    split_index = len(signals_df) // 2
    in_sample_df = signals_df.iloc[:split_index]
    out_of_sample_df = signals_df.iloc[split_index:]
    print(f"\n--- Splitting data for Out-of-Sample Test ---")
    print(f"  -> In-sample (first half): {len(in_sample_df)} signals")
    print(f"  -> Out-of-sample (second half): {len(out_of_sample_df)} signals")
    
    # --- 3. Run Grid Search on the FIRST HALF (In-Sample) to find parameters ---
    print("\n--- Running Grid Search on IN-SAMPLE data (first half) ---")
    in_sample_best_params, _, _, _ = run_full_grid_search(in_sample_df)
    
    if in_sample_best_params[0] is None:
        print("\nCould not determine optimal parameters from the in-sample data. Halting.")
        return
        
    print(f"\nOptimal ROI Params (In-Sample): TP={in_sample_best_params[0]:.2%}, SL={in_sample_best_params[1]:.2%}")

    # --- 4. Run ONE simulation on the SECOND HALF (Out-of-Sample) using the found parameters ---
    print("\n--- Running ONE simulation on OUT-OF-SAMPLE data (second half) ---")
    oos_tp, oos_sl = in_sample_best_params
    oos_avg_roi, oos_win_rate, oos_trades = run_simulation(out_of_sample_df, oos_tp, oos_sl)

    if oos_trades:
        print(f"\nOut-of-Sample Results: Avg ROI={oos_avg_roi:.2f}%, Win Rate={oos_win_rate:.2f}%")
        pd.DataFrame(oos_trades).to_csv("out_of_sample_analysis.csv", index=False)
        print("Saved out-of-sample analysis trades to 'out_of_sample_analysis.csv'")
    else:
        print("\nNo trades were completed in the out-of-sample simulation.")

if __name__ == "__main__":
    main() 
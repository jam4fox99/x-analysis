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
    win_rate = (results_df['ROI (%)'] > 0).mean()
    
    return avg_roi, win_rate, trades

def main():
    """Main function to run the grid search for dual objectives."""
    try:
        signals_df = pd.read_csv("signals.csv", parse_dates=['created_at'], dtype={'ticker': str})
        # Filter for entries and only the first one per ticker
        signals_df = signals_df[signals_df['action'] == 'entry']
        signals_df = signals_df.loc[signals_df.groupby('ticker')['created_at'].idxmin()]
        print(f"Loaded {len(signals_df)} unique first-entry signals.")
    except FileNotFoundError:
        print("Error: signals.csv not found. Please run main.py first.")
        return

    best_roi = -np.inf
    best_roi_params = (None, None)
    best_roi_trades = []

    best_win_rate = -1
    best_win_rate_params = (None, None)
    best_win_rate_trades = []
    
    param_grid = [(tp, sl) for tp in TP_GRID for sl in SL_GRID]
    
    print("\nStarting grid search for optimal Take-Profit, Stop-Loss, and Win Rate...")
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
            
    print("\n--- Grid Search Complete ---")
    
    # --- Results for Best ROI ---
    if best_roi_params[0] is not None:
        print("\n--- Best Average ROI Strategy ---")
        print(f"Optimal Take-Profit: {best_roi_params[0]:.0%}")
        print(f"Optimal Stop-Loss: {best_roi_params[1]:.0%}")
        print(f"Best Average ROI: {best_roi:.2f}%")
        
        roi_df = pd.DataFrame(best_roi_trades)
        roi_filename = "optimal_roi_trades.csv"
        roi_df.to_csv(roi_filename, index=False)
        print(f"Saved detailed trades for this strategy to '{roi_filename}'")
    else:
        print("Could not determine optimal ROI parameters.")

    # --- Results for Best Win Rate ---
    if best_win_rate_params[0] is not None:
        print("\n--- Highest Win Rate Strategy ---")
        print(f"Take-Profit for Best Win Rate: {best_win_rate_params[0]:.0%}")
        print(f"Stop-Loss for Best Win Rate: {best_win_rate_params[1]:.0%}")
        print(f"Highest Win Rate: {best_win_rate:.2%}")
        
        win_rate_df = pd.DataFrame(best_win_rate_trades)
        win_rate_filename = "optimal_win_rate_trades.csv"
        win_rate_df.to_csv(win_rate_filename, index=False)
        print(f"Saved detailed trades for this strategy to '{win_rate_filename}'")
    else:
        print("Could not determine optimal win rate parameters.")

if __name__ == "__main__":
    main() 
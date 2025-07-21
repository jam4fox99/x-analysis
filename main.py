import os
import re
import json
import requests # Use requests for HTTP calls
import pandas as pd
import openai
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import time # Add time for polling

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TARGET_USERNAME = "SarwanJohn"
TWEET_FETCH_LIMIT = 10000 # Increased limit
GPT_MODEL = "gpt-4.1-nano"
MAX_WORKERS = 10 # Number of threads for parallel processing

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Tweet Fetching (Apify Asynchronous Version) ---
def fetch_tweets_with_apify(username, limit):
    """
    Fetches tweets using the Apify Tweet Scraper API asynchronously
    to handle long-running scrapes.
    """
    if not APIFY_API_TOKEN:
        print("Error: APIFY_API_TOKEN is not set in your .env file.")
        return []

    print(f"Starting to fetch {limit} tweets for user '{username}' via Apify...")
    
    start_url = f"https://api.apify.com/v2/acts/kaitoeasyapi~twitter-x-data-tweet-scraper-pay-per-result-cheapest/runs?token={APIFY_API_TOKEN}"
    
    headers = {"Content-Type": "application/json"}
    
    data = {
        "from": username,
        "maxItems": limit,
        "queryType": "Latest",
        "lang": "en",
        "include:nativeretweets": False
    }
    
    try:
        # 1. Start the run
        print("  -> Starting the scraping job on Apify...")
        response = requests.post(start_url, headers=headers, json=data)
        response.raise_for_status()
        run_info = response.json().get('data', {})
        run_id = run_info.get('id')
        dataset_id = run_info.get('defaultDatasetId')

        if not run_id or not dataset_id:
            print("  -> Failed to start the Apify run.")
            return []
            
        print(f"  -> Job started with Run ID: {run_id}")

        # 2. Poll for completion
        status_url = f"https://api.apify.com/v2/acts/kaitoeasyapi~twitter-x-data-tweet-scraper-pay-per-result-cheapest/runs/{run_id}?token={APIFY_API_TOKEN}"
        while True:
            print("  -> Checking job status...")
            status_response = requests.get(status_url)
            status_response.raise_for_status()
            status_data = status_response.json().get('data', {})
            status = status_data.get('status')
            
            if status == 'SUCCEEDED':
                print("  -> Job succeeded!")
                break
            elif status in ['FAILED', 'ABORTED', 'TIMED_OUT']:
                print(f"  -> Job failed with status: {status}")
                return []
            
            time.sleep(10) # Wait 10 seconds before checking again

        # 3. Fetch the results
        print("  -> Fetching results from dataset...")
        results_url = f"https://api.apify.com/v2/datasets/{dataset_id}/items?token={APIFY_API_TOKEN}"
        results_response = requests.get(results_url)
        results_response.raise_for_status()
        
        tweets = results_response.json()
        print(f"  -> Fetched {len(tweets)} tweets from Apify.")
        
        adapted_tweets = [{'created_at': pd.to_datetime(t.get('createdAt')), 'text': t.get('text', '')} for t in tweets]
        return adapted_tweets
        
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while calling the Apify API: {e}")
        return []

# --- Helper Functions ---
def save_tweets_to_csv(tweets, filename="tweets.csv"):
    """Saves a list of tweets to a CSV file."""
    if not tweets:
        print("No tweets to save.")
        return
    df = pd.DataFrame(tweets)
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} tweets to {filename}")

def load_tweets_from_csv(filename="tweets.csv"):
    """Loads tweets from a CSV file."""
    if not os.path.exists(filename):
        return None
    df = pd.read_csv(filename, parse_dates=['created_at'])
    return df

def group_tweets_by_ticker(tweets_df):
    """Groups tweets by ticker symbol."""
    ticker_groups = {}
    for _, row in tweets_df.iterrows():
        tickers = re.findall(r'\$[A-Za-z]+', row['text'])
        for ticker in tickers:
            ticker = ticker.upper()
            if ticker not in ticker_groups:
                ticker_groups[ticker] = []
            ticker_groups[ticker].append(row.to_dict())
    
    for ticker in ticker_groups:
        ticker_groups[ticker].sort(key=lambda x: x['created_at'])
        
    return ticker_groups

# --- Analysis Function ---
def analyze_ticker_batch(ticker, tweet_batch):
    """Analyzes a batch of tweets for a single ticker."""
    print(f"\nAnalyzing batch for {ticker}...")
    
    formatted_tweets = "\n".join([f"- {t['created_at']}: {t['text']}" for t in tweet_batch])
    
    prompt = f"""
    You are a financial analyst specializing in interpreting social media signals for stock trading.
    Your task is to carefully analyze the following sequence of tweets for the ticker {ticker}, which are provided in strict chronological order.
    Focus on detecting explicit or strongly implied trading signals based on common trading language and context.
    Be conservative in your classifications: only label a tweet as a signal if it clearly meets the criteria below, avoiding over-interpretation of vague, neutral, or unrelated content.
    Consider the overall narrative flow across tweets—e.g., an entry might build on prior hype, but don't assume signals where none are evident.

    Definitions for actions (only use these; do not invent new ones):
    - 'entry': Clear buy signals, such as explicit calls to "buy now," "get in," "loading up," mentions of "moon," "breakout," "undervalued," or positive catalysts like "news incoming" with bullish intent. Must indicate initiating or adding to a position.
    - 'exit': Clear sell signals, such as "taking profits," "dumping," "sell now," "topping out," warnings of "crash" or "resistance," or negative catalysts with bearish intent. Must indicate closing or reducing a position.
    - 'hold': Explicit advice to maintain a position, like "hold strong," "diamond hands," "not selling yet," or reassurances during dips/volatility.
    - 'update': Neutral status updates without buy/sell intent, such as price/volume reports ("up 20% today"), news without bias ("earnings tomorrow"), or general observations ("watching closely").

    Tweets to Analyze:
    {formatted_tweets}

    Return a JSON object with a "signals" key, containing a list of all detected signals.
    Each signal should be a JSON object with "created_at", "ticker", "action", and "original_text".
    """
    
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        result_text = response.choices[0].message.content
        signals_data = json.loads(result_text)
        return signals_data.get("signals", [])
    except Exception as e:
        print(f"  -> Could not analyze batch for {ticker}. Error: {e}")
        return []

# --- Main Orchestration ---
def main():
    """Main function to orchestrate the process."""
    INCLUDE_RETWEETS = False

    if os.path.exists("tweets.csv"):
        print("Loading tweets from existing file: tweets.csv")
        tweets_df = load_tweets_from_csv("tweets.csv")
    else:
        print("No local tweet file found. Fetching from API...")
        all_tweets = fetch_tweets_with_apify(TARGET_USERNAME, TWEET_FETCH_LIMIT)
        if all_tweets:
            save_tweets_to_csv(all_tweets)
            tweets_df = pd.DataFrame(all_tweets)
        else:
            tweets_df = pd.DataFrame()

    if tweets_df.empty:
        print("No tweets available to analyze. Exiting.")
        return

    if not INCLUDE_RETWEETS:
        original_count = len(tweets_df)
        tweets_df = tweets_df[~tweets_df['text'].str.startswith("RT @")]
        print(f"  -> Filtering out retweets. Kept {len(tweets_df)} of {original_count} tweets.")

    if tweets_df['created_at'].dt.tz is None:
        tweets_df['created_at'] = tweets_df['created_at'].dt.tz_localize('UTC')
    else:
        tweets_df['created_at'] = tweets_df['created_at'].dt.tz_convert('UTC')
        
    ticker_groups = group_tweets_by_ticker(tweets_df)
    print(f"\nGrouped tweets into {len(ticker_groups)} ticker-specific batches.")
    print(f"Starting analysis using {MAX_WORKERS} threads...")

    all_signals = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_ticker = {executor.submit(analyze_ticker_batch, ticker, batch): ticker for ticker, batch in ticker_groups.items()}
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                signals = future.result()
                if signals:
                    all_signals.extend(signals)
                    print(f"  -> Completed: {ticker} ({len(signals)} signals found)")
            except Exception as exc:
                print(f"  -> Error processing {ticker}: {exc}")

    if all_signals:
        signals_df = pd.DataFrame(all_signals)
        signals_df['created_at'] = pd.to_datetime(signals_df['created_at'])
        signals_df.sort_values(by='created_at', inplace=True)
        signals_df.to_csv("signals.csv", index=False)
        print(f"\nSaved {len(signals_df)} signals to signals.csv")
    else:
        print("\nNo signals were generated.")

if __name__ == "__main__":
    main() 
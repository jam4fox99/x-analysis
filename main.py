import os
import re
import json
import time
import tweepy
import pandas as pd
import openai
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TARGET_USERNAME = "SarwanJohn"
TWEET_FETCH_LIMIT = 200 # Max number of tweets to fetch

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Functions from previous steps (get_user_id, fetch_tweets) ---
# ... (These will be re-added here) ...

def get_user_id(client, username):
    """Fetches the user ID for a given username."""
    try:
        response = client.get_user(username=username)
        if response.data:
            return response.data.id
        else:
            print(f"Error: User '{username}' not found.")
            return None
    except Exception as e:
        print(f"An error occurred while fetching user ID: {e}")
        return None

def fetch_tweets(client, user_id, limit=1000):
    """
    Fetches up to 1000 tweets (10 API calls).
    """
    print(f"Starting to fetch {limit} tweets...")
    all_tweets = []
    pagination_token = None

    try:
        while len(all_tweets) < limit:
            # The API requires max_results to be between 5 and 100.
            # We'll fetch 100 and truncate later if needed.
            max_results = min(100, limit - len(all_tweets))
            if max_results == 0:
                break
            
            # Ensure max_results is at least 5 if we are not on the last page
            if max_results < 5:
                # If we need less than 5, it's better to fetch 5 and truncate
                # or just stop. Since we have most tweets, we can stop here.
                print(f"  -> Nearing fetch limit. Collected {len(all_tweets)} tweets.")
                break

            response = client.get_users_tweets(
                id=user_id,
                max_results=max_results,
                pagination_token=pagination_token,
                tweet_fields=["created_at", "text"]
            )

            if response.data:
                all_tweets.extend(response.data)
                print(f"  -> Fetched {len(response.data)} tweets. Total collected: {len(all_tweets)}")

            pagination_token = response.meta.get('next_token')
            if not pagination_token:
                print("  -> Reached the end of the user's tweets.")
                break
                
    except tweepy.errors.TooManyRequests:
        print("Rate limit exceeded. Please wait 15 minutes before trying again.")
    except Exception as e:
        print(f"An unexpected error occurred during fetching: {e}")
    
    print(f"\nFinished fetching. Total tweets collected: {len(all_tweets)}")
    return all_tweets[:limit]

def save_tweets_to_csv(tweets, filename="tweets.csv"):
    """Saves a list of tweets to a CSV file with the correct data."""
    if not tweets:
        print("No tweets to save.")
        return
    # --- DEFINITIVE FIX ---
    # The 'created_at' attribute is directly on the tweet object.
    tweet_data = [
        {"id": t.id, "text": t.text, "created_at": t.created_at} 
        for t in tweets
    ]
    # --- END FIX ---
    df = pd.DataFrame(tweet_data)
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} tweets to {filename}")

def load_tweets_from_csv(filename="tweets.csv"):
    """Loads tweets from a CSV file, correctly parsing dates."""
    if not os.path.exists(filename):
        return None
    print(f"Loading tweets from existing file: {filename}")
    # This is the correct and robust way to parse the dates
    df = pd.read_csv(filename, parse_dates=['created_at'])
    return df

def group_tweets_by_ticker(tweets_df):
    """
    Groups tweets by ticker symbol.
    """
    ticker_groups = {}
    for index, row in tweets_df.iterrows():
        tickers = re.findall(r'\$[A-Za-z]+', row['text'])
        for ticker in tickers:
            ticker = ticker.upper()
            if ticker not in ticker_groups:
                ticker_groups[ticker] = []
            ticker_groups[ticker].append(row.to_dict())
    
    # Sort each group by date
    for ticker in ticker_groups:
        ticker_groups[ticker].sort(key=lambda x: x['created_at'])
        
    return ticker_groups

def analyze_ticker_batch(ticker, tweet_batch):
    """
    Analyzes a batch of tweets for a single ticker.
    """
    print(f"\nAnalyzing batch for {ticker}...")
    
    # Format the batch for the AI
    formatted_tweets = "\n".join([f"- {t['created_at']}: {t['text']}" for t in tweet_batch])
    
    prompt = f"""
    You are a financial analyst. Analyze the following sequence of tweets for the ticker {ticker}
    and identify all trading signals. The tweets are in chronological order.

    Tweets:
    {formatted_tweets}

    Return a JSON object with a "signals" key, containing a list of all detected signals.
    Each signal should be a JSON object with "created_at", "ticker", "action", and "original_text".
    Actions can be 'entry', 'exit', 'hold', or 'update'.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        
        result_text = response.choices[0].message.content
        signals_data = json.loads(result_text)
        return signals_data.get("signals", [])
    except Exception as e:
        print(f"  -> Could not analyze batch for {ticker}. Error: {e}")
        return []

def main():
    """
    Main function to orchestrate the tweet fetching, analysis, and signal generation.
    """
    # --- Step 1: Fetch Tweets ---
    if os.path.exists("tweets.csv"):
        print("Loading tweets from existing file: tweets.csv")
        tweets_df = pd.read_csv("tweets.csv", parse_dates=['created_at'])
        # Ensure timezone-awareness
        if tweets_df['created_at'].dt.tz is None:
            tweets_df['created_at'] = tweets_df['created_at'].dt.tz_localize('UTC')
        else:
            tweets_df['created_at'] = tweets_df['created_at'].dt.tz_convert('UTC')
        
        print(f"\nSuccessfully loaded {len(tweets_df)} tweets.")
        # When loading from file, we don't clear signals, we append.
        
    else:
        print("No local tweet file found. Fetching from API...")
        tweepy_client = tweepy.Client(X_BEARER_TOKEN)
        
        user_id = get_user_id(tweepy_client, TARGET_USERNAME)

        if user_id:
            print(f"Fetching last {TWEET_FETCH_LIMIT} tweets for user {TARGET_USERNAME}...")
            all_tweets = fetch_tweets(tweepy_client, user_id, limit=TWEET_FETCH_LIMIT)
            
            # Save the collected tweets to a file
            if all_tweets:
                tweets_df = pd.DataFrame([
                    {'created_at': t.created_at, 'text': t.text} 
                    for t in all_tweets
                ])
                tweets_df.to_csv("tweets.csv", index=False)
                print(f"Saved {len(all_tweets)} tweets to tweets.csv")
            else:
                tweets_df = pd.DataFrame()


            # When we fetch new tweets, we should start with a clean slate for signals.
            if os.path.exists("signals.csv"):
                print("Cleared old signals.csv to start a fresh analysis.")
                os.remove("signals.csv")

    if tweets_df.empty:
        print("No tweets available to analyze. Exiting.")
        return

    # --- Group Tweets by Ticker ---
    ticker_groups = group_tweets_by_ticker(tweets_df)
    print(f"\nGrouped tweets into {len(ticker_groups)} ticker-specific batches.")

    # --- Analyze Batches and Save Signals ---
    all_signals = []
    for ticker, batch in ticker_groups.items():
        signals = analyze_ticker_batch(ticker, batch)
        if signals:
            all_signals.extend(signals)

    if all_signals:
        signals_df = pd.DataFrame(all_signals)
        signals_df['created_at'] = pd.to_datetime(signals_df['created_at'])
        signals_df.sort_values(by='created_at', inplace=True)
        signals_df.to_csv("signals.csv", index=False)
        print(f"\nSaved {len(signals_df)} signals to signals.csv")
    else:
        print("\nNo signals were generated from the analysis.")


if __name__ == "__main__":
    main() 
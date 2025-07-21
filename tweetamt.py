import os
import requests
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")
# This is the endpoint to look up a user by their username
BASE_URL = "https://api.twitter.com/2/users/by/username/"

def get_total_tweet_count(username):
    """
    Gets the total public tweet count for a user by fetching their profile.
    This count includes original tweets, replies, and retweets.
    
    Args:
        username (str): The X user handle.

    Returns:
        int: The total tweet count, or None if an error occurs.
    """
    if not BEARER_TOKEN:
        print("Error: X_BEARER_TOKEN is not set in your .env file.")
        return None

    # We need to specify which fields we want in the response.
    # 'public_metrics' contains the tweet_count.
    params = {
        "user.fields": "public_metrics"
    }
    
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}"
    }
    
    url = f"{BASE_URL}{username}"
    
    print(f"Querying X API for user profile: '{username}'")
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json().get("data", {})
        tweet_count = data.get("public_metrics", {}).get("tweet_count")
        
        return tweet_count
        
    except requests.exceptions.HTTPError as e:
        print(f"\nError calling the X API: {e.response.status_code} {e.response.reason}")
        error_details = e.response.json()
        print(f"  -> Title: {error_details.get('title')}")
        print(f"  -> Detail: {error_details.get('detail')}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"An unexpected network error occurred: {e}")
        return None

def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(
        description="Get the total public tweet count for a given X user (includes tweets, replies, and retweets)."
    )
    parser.add_argument("username", type=str, help="The X username to look up (e.g., 'elonmusk').")
    
    args = parser.parse_args()
    
    count = get_total_tweet_count(args.username)
    
    if count is not None:
        print("\n--- Counting Complete ---")
        print(f"Total posts for '{args.username}': {count:,}")
    else:
        print("\nCould not retrieve the tweet count.")

if __name__ == "__main__":
    main() 
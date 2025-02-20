# fetch_data.py
import logging
# Set up logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    force=True
)

import time
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional, Union
from config import SUPABASE_URL, DATA_DIR
from utils import save_pickle
from dateutil.parser import parse
from datetime import datetime, timedelta
from supabase import create_client, Client
from tqdm import tqdm

# Load environment variables and set up logging
load_dotenv()
API_TOKEN = os.getenv('API_TOKEN')

class SupabaseClient:
    def __init__(self):
        self.client: Client = create_client(SUPABASE_URL, API_TOKEN)

class AccountFetcher(SupabaseClient):
    def fetch_batch(self, offset: int = 0, limit: int = 1000) -> List[Dict]:
        try:
            response = self.client.table('account').select('*').range(offset, offset + limit - 1).execute()
            return response.data
        except Exception as e:
            logging.error(f"Error fetching accounts batch: {str(e)}")
            return []

    def fetch_all(self) -> List[Dict]:
        all_accounts = []
        offset = 0
        batch_size = 1000

        while True:
            logging.info(f"Fetching accounts {offset} to {offset + batch_size}...")
            batch = self.fetch_batch(offset, batch_size)
            
            if not batch:
                break
            
            all_accounts.extend(batch)
            offset += batch_size
            
            if len(batch) < batch_size:
                break
            
            time.sleep(0.1)  # To avoid hitting rate limits

        logging.info(f"Total accounts fetched: {len(all_accounts)}")
        return all_accounts

class TweetFetcher(SupabaseClient):
    def fetch_batch(self, account_id: Optional[int] = None, offset: int = 0, limit: int = 1000, 
                    start_date: Optional[datetime] = None, 
                    end_date: Optional[datetime] = None,
                    keywords: Optional[List[str]] = None) -> List[Dict]:
        query = self.client.table('tweets').select('*').order('created_at', desc=True).range(offset, offset + limit - 1)
        
        if account_id is not None:
            query = query.eq('account_id', account_id)
        
        if start_date and end_date:
            query = query.gte('created_at', start_date.isoformat()).lte('created_at', end_date.isoformat())

        if keywords:
            keyword_string = ' | '.join(keywords)  # Join keywords with OR operator
            query = query.text_search('full_text', keyword_string)

        try:
            logging.info(f"Making request to fetch tweets")
            response = query.execute()
            logging.info(f"Received {len(response.data)} tweets")
            return response.data
        except Exception as e:
            logging.error(f"Error fetching tweets: {str(e)}")
            logging.error(f"Query details: {query._query}")  # Log the query details for debugging
            return []

    def fetch_all(self, account_id: Optional[int] = None, 
                  start_date: Optional[Union[str, datetime]] = None, 
                  end_date: Optional[Union[str, datetime]] = None,
                  keywords: Optional[List[str]] = None) -> List[Dict]:
        all_tweets = []
        offset = 0
        batch_size = 1000
   
        while True:
            logging.info(f"Fetching tweets {offset} to {offset + batch_size}...")
            batch = self.fetch_batch(account_id, offset, batch_size, start_date, end_date, keywords)
           
            if not batch:
                break
           
            all_tweets.extend(batch)
            offset += batch_size
           
            if len(batch) < batch_size:
                break
           
            time.sleep(0.01)  # To avoid hitting rate limits
   
        logging.info(f"Total tweets fetched: {len(all_tweets)}")
        return all_tweets

    async def fetch_user_tweets(self, user_id: str, limit: int = 5) -> List[Dict]:
        """Fetch most recent tweets for a user"""
        try:
            response = self.client.table('tweets')\
                .select('*')\
                .eq('account_id', user_id)\
                .order('created_at', desc=True)\
                .limit(limit)\
                .execute()
            
            tweets = response.data
            if tweets:
                logging.info(f"Found {len(tweets)} tweets for user {user_id}")
            return tweets
        except Exception as e:
            logging.error(f"Error fetching tweets for user {user_id}: {str(e)}")
            return []

class FollowFetcher(SupabaseClient):
    def fetch_followers_batch(self, account_id: str, offset: int = 0, limit: int = 1000) -> List[Dict]:
        try:
            response = self.client.table('followers') \
                .select('follower_account_id') \
                .eq('account_id', account_id) \
                .range(offset, offset + limit - 1) \
                .execute()
            return response.data
        except Exception as e:
            logging.error(f"Error fetching followers batch: {str(e)}")
            return []

    def fetch_following_batch(self, account_id: str, offset: int = 0, limit: int = 1000) -> List[Dict]:
        try:
            response = self.client.table('following') \
                .select('following_account_id') \
                .eq('account_id', account_id) \
                .range(offset, offset + limit - 1) \
                .execute()
            return response.data
        except Exception as e:
            logging.error(f"Error fetching following batch: {str(e)}")
            return []

    def fetch_all_followers(self, account_id: str) -> List[str]:
        all_followers = set()
        offset = 0
        batch_size = 1000

        while True:
            batch = self.fetch_followers_batch(account_id, offset, batch_size)
            
            if not batch:
                break
            
            new_followers = {item['follower_account_id'] for item in batch}
            all_followers.update(new_followers)
            
            if len(batch) < batch_size:
                break
            
            offset += batch_size
            time.sleep(0.1)

        return list(all_followers)

    def fetch_all_following(self, account_id: str) -> List[str]:
        all_following = set()
        offset = 0
        batch_size = 1000

        while True:
            batch = self.fetch_following_batch(account_id, offset, batch_size)
            
            if not batch:
                break
            
            new_following = {item['following_account_id'] for item in batch}
            all_following.update(new_following)
            
            if len(batch) < batch_size:
                break
            
            offset += batch_size
            time.sleep(0.1)

        return list(all_following)

def save_data(data: List[Dict], filename: str):
    filepath = os.path.join(DATA_DIR, filename)
    save_pickle(data, filepath)
    logging.info(f"Data saved to {filepath}")

def fetch_data_main(args):
    account_fetcher = AccountFetcher()
    tweet_fetcher = TweetFetcher()
    follow_fetcher = FollowFetcher()

    # Fetch and save accounts
    accounts = account_fetcher.fetch_all()
    if not accounts:
        logging.warning(f"Error fetching accounts.")
        return None

    account_map = {str(account['username']): account['account_id'] for account in accounts}

    tweets_dict = {}
    follows_dict = {}
    
    # Fetch and save tweets and follow data for each username
    for username in args.usernames:
        account_id = account_map.get(username)
        if account_id is None:
            logging.warning(f"Unknown username: {username}. Skipping...")
            continue

        logging.info(f"Fetching tweets for {username}")
        user_tweets = tweet_fetcher.fetch_all(account_id, args.start_date, args.end_date, args.keywords)
        tweets_dict[username] = user_tweets

        logging.info(f"Fetching followers and following for {username}")
        followers = follow_fetcher.fetch_all_followers(str(account_id))
        following = follow_fetcher.fetch_all_following(str(account_id))
        follows_dict[username] = {
            'followers': followers,
            'following': following
        }

    return tweets_dict, follows_dict

async def test_tweet_fetch():
    """Test function to verify tweet fetching"""
    fetcher = TweetFetcher()
    # Test with a known user ID that should have tweets
    test_user_id = "1464483769222680582"  # Replace with a known active user ID
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    tweets = await fetcher.fetch_user_tweets(test_user_id, start_date, end_date)
    
    print(f"\nTest Results:")
    print(f"User ID: {test_user_id}")
    print(f"Tweets found: {len(tweets)}")
    if tweets:
        print("\nFirst tweet sample:")
        print(f"ID: {tweets[0]['tweet_id']}")
        print(f"Created: {tweets[0]['created_at']}")
        print(f"Text: {tweets[0]['full_text'][:100]}...")

if __name__ == "__main__":
    import argparse
    import asyncio
    
    # Check if we're running a test
    import sys
    if len(sys.argv) == 1:  # No arguments provided
        # Just run the test
        asyncio.run(test_tweet_fetch())
    else:
        # Run the normal argument parser
        parser = argparse.ArgumentParser(description="Fetch account, tweet, and follow data for multiple accounts")
        parser.add_argument("usernames", nargs='+', help="Twitter usernames to fetch data for")
        parser.add_argument("--start_date", help="Start date for tweet fetch (YYYY-MM-DD)")
        parser.add_argument("--end_date", help="End date for tweet fetch (YYYY-MM-DD)")
        parser.add_argument("--keywords", nargs='*', help="Keywords to filter tweets (optional)")
        args = parser.parse_args()
        tweets_dict, follows_dict = fetch_data_main(args)
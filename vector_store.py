import logging
# Set up logging FIRST, before other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Simplified format for cleaner output
    force=True  # Force update the root logger
)

import os
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from fetch_data import TweetFetcher, AccountFetcher, ProfileFetcher
import json
import asyncio
import time
from tenacity import retry, wait_exponential, stop_after_attempt
from concurrent.futures import ThreadPoolExecutor
from functools import partial

class VectorStore:
    def __init__(self, data_dir: str = "data", days: int = None, max_concurrent: int = 3):
        self.dimension = 768  # ModernBERT embedding dimension
        self.data_dir = data_dir
        self.embeddings_path = os.path.join(data_dir, 'user_embeddings.json')
        self.profiles_path = os.path.join(data_dir, 'user_profiles.json')
        self.checkpoint_path = os.path.join(data_dir, 'checkpoint.json')
        self.days = days  # Store days parameter
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize model
        self.model = SentenceTransformer("nomic-ai/modernbert-embed-base")
        
        # Initialize fetchers
        self.tweet_fetcher = TweetFetcher()
        self.account_fetcher = AccountFetcher()
        self.profile_fetcher = ProfileFetcher()
        
        # Load existing embeddings and profiles if any
        self.user_embeddings = self.load_embeddings()
        self.profiles = self.load_profiles()
        self.retry_count = 0
        self.max_retries = 5
        self.max_concurrent = max_concurrent  # Control parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.processed_users = self.load_checkpoint()

    def load_embeddings(self) -> Dict:
        """Load existing embeddings if available and filter by age if days is set"""
        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, 'r') as f:
                embeddings = json.load(f)
                
                # If days filter is set, filter embeddings by age
                if self.days is not None:
                    cutoff_date = datetime.now() - timedelta(days=self.days)
                    filtered_embeddings = {}
                    
                    for user_id, data in embeddings.items():
                        last_updated = datetime.fromisoformat(data['metadata']['last_updated'])
                        if last_updated >= cutoff_date:
                            filtered_embeddings[user_id] = data
                    
                    logging.info(f"Filtered embeddings from {len(embeddings)} to {len(filtered_embeddings)} "
                               f"(last {self.days} days)")
                    return filtered_embeddings
                
                return embeddings
        return {}

    def load_profiles(self) -> Dict:
        """Load existing profiles if available and filter to match embeddings"""
        if os.path.exists(self.profiles_path):
            with open(self.profiles_path, 'r') as f:
                profiles = json.load(f)
                
                # If we have filtered embeddings, only keep matching profiles
                if self.days is not None:
                    filtered_profiles = {
                        user_id: profile 
                        for user_id, profile in profiles.items() 
                        if user_id in self.user_embeddings
                    }
                    logging.info(f"Filtered profiles from {len(profiles)} to {len(filtered_profiles)} "
                               f"to match embeddings from last {self.days} days")
                    return filtered_profiles
                
                return profiles
        return {}

    def load_checkpoint(self) -> set:
        """Load set of already processed users"""
        try:
            if os.path.exists(self.checkpoint_path):
                with open(self.checkpoint_path, 'r') as f:
                    return set(json.load(f))
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}")
        return set()

    def save_checkpoint(self):
        """Save current progress"""
        try:
            with open(self.checkpoint_path, 'w') as f:
                json.dump(list(self.processed_users), f)
        except Exception as e:
            logging.error(f"Error saving checkpoint: {e}")

    def chunk_text(self, text: str, max_tokens: int = 512) -> List[str]:
        """Split text into chunks that won't exceed token limit"""
        # Approximate token count (words + some extra for special tokens)
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word.split())
            if current_length + word_length > max_tokens:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        logging.info(f"Split text into {len(chunks)} chunks")
        return chunks

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using SentenceTransformer"""
        try:
            chunks = self.chunk_text(text)
            all_embeddings = []
            
            # Process chunks in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                chunk_embeddings = list(executor.map(
                    lambda chunk: self.model.encode([f"search_document: {chunk}"])[0],
                    chunks
                ))
                all_embeddings.extend(chunk_embeddings)
            
            if len(all_embeddings) > 1:
                final_embedding = np.mean(all_embeddings, axis=0)
                return final_embedding.tolist()
            else:
                return all_embeddings[0].tolist()
            
        except Exception as e:
            logging.error(f"Embedding generation failed: {str(e)}")
            raise

    async def process_user_tweets(self, user_id: str, tweet_limit: int = None) -> Dict:
        """Fetch tweets for a user and generate weekly embeddings"""
        try:
            # Check what weeks we already have for this user
            existing_weeks = {}
            if user_id in self.user_embeddings:
                existing_weeks = self.user_embeddings[user_id]

            # Calculate the date range we need
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.days) if self.days else None

            # Fetch tweets
            tweets = await self.tweet_fetcher.fetch_user_tweets(user_id, limit=None, days=self.days)
            
            if not tweets:
                logging.info(f"No tweets found for user {user_id}")
                return existing_weeks  # Return existing data if no new tweets

            logging.info(f"Processing {len(tweets)} tweets for user {user_id}")

            # Process tweets and dates...
            weekly_tweets = {}
            skipped_tweets = 0
            for tweet in tweets:
                # Parse date...
                tweet_date = self.parse_tweet_date(tweet['created_at'])
                if not tweet_date:
                    skipped_tweets += 1
                    continue

                # Skip if tweet is outside our date range
                if start_date and tweet_date < start_date:
                    skipped_tweets += 1
                    continue

                # Get week key
                week_start = tweet_date - timedelta(days=tweet_date.weekday())
                week_key = week_start.strftime('%Y-%m-%d')
                
                # Skip if we already have this week and it's not too old
                if week_key in existing_weeks:
                    last_updated = datetime.fromisoformat(existing_weeks[week_key]['metadata']['last_updated'])
                    if (datetime.now() - last_updated).days < (self.days or 365):
                        skipped_tweets += 1
                        continue

                if week_key not in weekly_tweets:
                    weekly_tweets[week_key] = []
                weekly_tweets[week_key].append(tweet['full_text'])

            logging.info(f"Grouped into {len(weekly_tweets)} weeks (skipped {skipped_tweets} tweets)")

            # Generate embeddings only for new or outdated weeks
            weekly_embeddings = existing_weeks.copy()
            for week, texts in weekly_tweets.items():
                combined_text = " ".join(texts)
                if combined_text.strip():
                    embedding = self.generate_embedding(combined_text)
                    weekly_embeddings[week] = {
                        'embedding': embedding,
                        'tweet_count': len(texts),
                        'metadata': {
                            'tweets': texts,
                            'last_updated': datetime.now().isoformat()
                        }
                    }

            logging.info(f"Generated embeddings for {len(weekly_embeddings)} weeks")
            return weekly_embeddings

        except Exception as e:
            logging.error(f"Error processing user {user_id}: {str(e)}")
            logging.error(f"Full error: ", exc_info=True)  # Print full traceback
            return None

    async def process_users_batch(self, accounts: List[Dict], profile_lookup: Dict):
        """Process a batch of users concurrently"""
        tasks = []
        for account in accounts:
            user_id = str(account['account_id'])
            if user_id in self.processed_users:
                logging.info(f"Skipping already processed user {user_id}")
                continue

            username = account['username']
            logging.info(f"Queueing user @{username}")
            
            # Update profile
            profile = profile_lookup.get(user_id, {})
            if profile:
                self.profiles[user_id] = {
                    'username': username,
                    'avatar_url': profile.get('avatar_media_url', ''),
                    'bio': profile.get('bio', ''),
                    'location': profile.get('location', ''),
                    'website': profile.get('website', '')
                }
            
            task = asyncio.create_task(self.process_user_tweets(user_id))
            tasks.append((user_id, task))
        
        # Process each task
        for user_id, task in tasks:
            try:
                weekly_embeddings = await task
                if weekly_embeddings:
                    self.user_embeddings[user_id] = weekly_embeddings
                    self.processed_users.add(user_id)
                    # Save after each user
                    self.save_embeddings()
                    self.save_checkpoint()
            except Exception as e:
                logging.error(f"Failed to process user {user_id}: {str(e)}")

    def parse_tweet_date(self, date_str: str) -> Optional[datetime]:
        """Parse tweet date handling multiple formats"""
        try:
            return datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%fZ')
        except ValueError:
            try:
                return datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S+00:00')
            except ValueError:
                logging.error(f"Could not parse date: {date_str}")
                return None

    async def process_all_users(self, limit: int = None, tweet_limit: int = None):
        """Process all users with concurrent batches"""
        accounts = self.account_fetcher.fetch_all()
        profiles = self.profile_fetcher.fetch_all()
        profile_lookup = {p['account_id']: p for p in profiles}
        
        if limit:
            accounts = accounts[:limit]
        
        # If days filter is set, only process users whose embeddings are too old or missing
        if self.days is not None:
            cutoff_date = datetime.now() - timedelta(days=self.days)
            filtered_accounts = []
            
            for account in accounts:
                user_id = str(account['account_id'])
                user_data = self.user_embeddings.get(user_id, {})
                if not user_data or 'metadata' not in user_data:
                    filtered_accounts.append(account)
                    continue
                
                # Check if any weekly embedding is too old
                needs_update = False
                for week_data in user_data.values():
                    if 'metadata' in week_data:
                        last_updated = datetime.fromisoformat(week_data['metadata']['last_updated'])
                        if last_updated < cutoff_date:
                            needs_update = True
                            break
                
                if needs_update:
                    filtered_accounts.append(account)
            
            accounts = filtered_accounts
            logging.info(f"Found {len(accounts)} accounts needing update (older than {self.days} days)")
        
        total_accounts = len(accounts)
        logging.info(f"\nStarting processing of {total_accounts} accounts...")
        
        # Process in batches
        batch_size = self.max_concurrent
        for i in range(0, len(accounts), batch_size):
            batch = accounts[i:i + batch_size]
            logging.info(f"Processing batch {i//batch_size + 1}/{(total_accounts + batch_size - 1)//batch_size}")
            
            await self.process_users_batch(batch, profile_lookup)
            
            # Save periodically
            if (i + batch_size) % (batch_size * 2) == 0:
                self.save_embeddings()
                self.save_profiles()
                logging.info(f"Saved progress after {i + batch_size} users")

        # Final save
        self.save_embeddings()
        self.save_profiles()

    def save_embeddings(self):
        """Save embeddings to file"""
        try:
            with open(self.embeddings_path, 'w') as f:
                json.dump(self.user_embeddings, f)
            print(f"Saved {len(self.user_embeddings)} embeddings to {self.embeddings_path}")
            # Print first user as sample
            if self.user_embeddings:
                first_user = next(iter(self.user_embeddings.items()))
                print(f"Sample user data structure: {first_user[0]}: {list(first_user[1].keys())}")
                # Print sample week data
                sample_week = next(iter(first_user[1].values()))
                print(f"Sample week data: {list(sample_week.keys())}")  # Should show ['embedding', 'tweet_count', 'metadata']
        except Exception as e:
            print(f"Error saving embeddings: {str(e)}")
            print(f"First user data: {json.dumps(next(iter(self.user_embeddings.items())), indent=2)}")

    def save_profiles(self):
        """Save profiles to file"""
        try:
            with open(self.profiles_path, 'w') as f:
                json.dump(self.profiles, f)
            print(f"Saved {len(self.profiles)} profiles to {self.profiles_path}")
        except Exception as e:
            print(f"Error saving profiles: {e}")

async def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Generate embeddings for user tweets")
    parser.add_argument("--limit", type=int, help="Limit number of users to process")
    parser.add_argument("--tweets", type=int, default=5, help="Number of tweets per user")
    parser.add_argument("--profiles-only", action="store_true", help="Only fetch profiles, skip embeddings")
    parser.add_argument("--days", type=int, help="Only load/update embeddings from last N days")
    parser.add_argument("--concurrent", type=int, default=3, help="Number of users to process concurrently")
    args = parser.parse_args()

    # Initialize vector store with days and concurrent parameters
    store = VectorStore(days=args.days, max_concurrent=args.concurrent)
    
    if args.profiles_only:
        # Just fetch and save profiles
        accounts = store.account_fetcher.fetch_all()
        if args.limit:
            accounts = accounts[:args.limit]
            
        profiles = store.profile_fetcher.fetch_all()
        profile_lookup = {p['account_id']: p for p in profiles}
        
        for account in accounts:
            user_id = str(account['account_id'])
            username = account['username']
            
            profile = profile_lookup.get(user_id, {})
            if profile:
                store.profiles[user_id] = {
                    'username': username,
                    'avatar_url': profile.get('avatar_media_url', ''),
                    'bio': profile.get('bio', ''),
                    'location': profile.get('location', ''),
                    'website': profile.get('website', '')
                }
        
        store.save_profiles()
        logging.info(f"Saved profiles for {len(store.profiles)} users")
    else:
        # Process embeddings and profiles
        await store.process_all_users(limit=args.limit, tweet_limit=args.tweets)

if __name__ == "__main__":
    asyncio.run(main()) 
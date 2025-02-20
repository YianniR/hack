import logging
# Set up logging FIRST, before other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Simplified format for cleaner output
    force=True  # Force update the root logger
)

import os
import faiss
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import openai
from dotenv import load_dotenv
from fetch_data import TweetFetcher, AccountFetcher
import json
import asyncio
import time
from tenacity import retry, wait_exponential, stop_after_attempt

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
print(f"Using OpenAI API key: {openai.api_key[:8]}...")  # Print first 8 chars of API key

class VectorStore:
    def __init__(self, data_dir: str = "data"):
        self.dimension = 1536  # OpenAI ada-002 embedding dimension
        self.data_dir = data_dir
        self.embeddings_path = os.path.join(data_dir, 'user_embeddings.json')
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize fetchers
        self.tweet_fetcher = TweetFetcher()
        self.account_fetcher = AccountFetcher()
        
        # Load existing embeddings if any
        self.user_embeddings = self.load_embeddings()
        self.retry_count = 0
        self.max_retries = 5

    def load_embeddings(self) -> Dict:
        """Load existing embeddings if available"""
        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, 'r') as f:
                return json.load(f)
        return {}

    def chunk_text(self, text: str, max_tokens: int = 6000) -> List[str]:
        """Split text into chunks that won't exceed token limit"""
        # More conservative approximation: 1 token ≈ 3 chars for English text
        chars_per_chunk = max_tokens * 3
        chunks = []
        
        while text:
            if len(text) <= chars_per_chunk:
                chunks.append(text)
                break
            
            # Find the last period or newline before the chunk limit
            split_point = text[:chars_per_chunk].rfind('.')
            if split_point == -1:
                split_point = text[:chars_per_chunk].rfind('\n')
            if split_point == -1:
                split_point = chars_per_chunk
            
            chunks.append(text[:split_point])
            text = text[split_point:].strip()
        
        logging.info(f"Split text into {len(chunks)} chunks (approx. {len(text)/3:.0f} tokens)")
        return chunks

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=10),
        stop=stop_after_attempt(5)
    )
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI's API with retry logic"""
        try:
            # Split text into chunks if needed
            chunks = self.chunk_text(text)
            if len(chunks) > 1:
                logging.info(f"Text split into {len(chunks)} chunks")
            
            # Generate embeddings for each chunk
            all_embeddings = []
            client = openai.OpenAI()
            
            for i, chunk in enumerate(chunks):
                response = client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=chunk
                )
                all_embeddings.append(response.data[0].embedding)
                if len(chunks) > 1:
                    logging.info(f"Processed chunk {i+1}/{len(chunks)}")
                time.sleep(0.5)  # Small delay between chunks
            
            # Average the embeddings if we had multiple chunks
            if len(all_embeddings) > 1:
                final_embedding = np.mean(all_embeddings, axis=0)
                return final_embedding.tolist()
            else:
                return all_embeddings[0]
            
        except Exception as e:
            self.retry_count += 1
            logging.warning(f"API call failed (attempt {self.retry_count}): {str(e)}")
            if "quota" in str(e).lower():
                logging.error("OpenAI API quota exceeded. Please check your billing.")
                raise Exception("API quota exceeded") from e
            raise  # Re-raise other exceptions for retry

    async def process_user_tweets(self, user_id: str, tweet_limit: int = 5) -> Dict:
        """Fetch most recent tweets for a user and generate embedding"""
        try:
            # Fetch tweets
            tweets = await self.tweet_fetcher.fetch_user_tweets(
                user_id,
                limit=tweet_limit  # Get last N tweets instead of using date range
            )
            
            if not tweets:
                logging.info(f"No tweets found for user {user_id}")
                return None

            # Combine all tweet text
            tweet_texts = [tweet['full_text'] for tweet in tweets if tweet.get('full_text')]
            
            if not tweet_texts:
                logging.info(f"No valid tweet texts found for user {user_id}")
                return None
            
            combined_text = " ".join(tweet_texts)
            
            if not combined_text.strip():
                logging.info(f"Empty combined text for user {user_id}")
                return None
            
            # Add delay between API calls
            time.sleep(1)  # Wait 1 second between calls
            
            # Generate embedding and metadata
            try:
                embedding = self.generate_embedding(combined_text)
                return {
                    'embedding': embedding,
                    'tweet_count': len(tweets),
                    'metadata': {
                        'tweets': tweet_texts,  # Store the actual tweets for reference
                        'last_updated': datetime.now().isoformat()
                    }
                }
            except Exception as e:
                if "quota" in str(e).lower():
                    logging.error("Stopping due to API quota limit")
                    raise SystemExit(1)
                raise
                
        except Exception as e:
            logging.error(f"Error processing user {user_id}: {str(e)}")
            return None

    async def process_all_users(self, limit: int = None, tweet_limit: int = 5):
        """Process all users and generate embeddings"""
        # Fetch all accounts
        accounts = self.account_fetcher.fetch_all()
        if limit:
            accounts = accounts[:limit]
        
        total_accounts = len(accounts)
        logging.info(f"\nStarting processing of {total_accounts} accounts (using {tweet_limit} tweets per user)...")
        
        # Process each account
        processed = 0
        successful = 0
        
        for account in accounts:
            user_id = str(account['account_id'])
            username = account['username']
            processed += 1
            
            logging.info(f"\n[{processed}/{total_accounts}] Processing @{username} (ID: {user_id})")
            
            try:
                # Generate embedding from recent tweets
                result = await self.process_user_tweets(user_id, tweet_limit=tweet_limit)
                
                if result is None:
                    logging.info(f"  ↳ No tweets found for @{username}")
                    continue
                
                # Store embedding and metadata
                self.user_embeddings[user_id] = {
                    'username': username,
                    'embedding': result['embedding'],
                    'metadata': {
                        'tweet_count': result['tweet_count'],
                        'tweets': result['metadata']['tweets'],
                        'last_updated': result['metadata']['last_updated']
                    }
                }
                
                successful += 1
                logging.info(f"  ↳ Success! Found {result['tweet_count']} tweets")
                
                # Save every 10 accounts
                if successful % 10 == 0:
                    self.save_embeddings()
                    logging.info(f"\nProgress: {successful} accounts processed successfully")
                
            except Exception as e:
                logging.error(f"  ↳ Error processing @{username}: {str(e)}")
        
        # Final save
        self.save_embeddings()
        logging.info(f"\nProcessing complete!")
        logging.info(f"Total accounts processed: {total_accounts}")
        logging.info(f"Successful embeddings: {successful}")
        logging.info(f"Failed/no tweets: {total_accounts - successful}")

    def save_embeddings(self):
        """Save embeddings to file"""
        with open(self.embeddings_path, 'w') as f:
            json.dump(self.user_embeddings, f)
        print(f"Saved {len(self.user_embeddings)} embeddings to {self.embeddings_path}")
        # Print first user as sample
        if self.user_embeddings:
            first_user = next(iter(self.user_embeddings.items()))
            print(f"Sample user data: {first_user[0]}: {list(first_user[1].keys())}")

async def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Generate embeddings for user tweets")
    parser.add_argument("--limit", type=int, help="Limit number of users to process")
    parser.add_argument("--tweets", type=int, default=5, help="Number of tweets per user")
    args = parser.parse_args()

    # Initialize vector store
    store = VectorStore()
    
    # Process all users
    await store.process_all_users(limit=args.limit, tweet_limit=args.tweets)
    
    logging.info("Embedding generation complete!")
    logging.info(f"Processed {len(store.user_embeddings)} users")

if __name__ == "__main__":
    asyncio.run(main()) 
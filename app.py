from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import os
import requests
import json
from openai import OpenAI
from dotenv import load_dotenv
import asyncio

app = Flask(__name__)
CORS(app)

# Load the model globally but lazily
model = None

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

load_dotenv()

CHUNK_SIZE = 6000  # About 1500 tokens
CHUNK_OVERLAP = 500  # Overlap between chunks

def get_model():
    global model
    if model is None:
        model = SentenceTransformer("nomic-ai/modernbert-embed-base")
    return model

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'}), 200

@app.route('/')
def serve_visualizer():
    return send_from_directory('.', 'visualizer.html')

@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory('data', filename)

@app.route('/get_embedding')
def get_embedding():
    text = request.args.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        # Load model only when needed
        model = get_model()
        embedding = model.encode([f"search_document: {text}"])[0].tolist()
        return jsonify({'embedding': embedding})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/network_map.graphml')
def serve_graphml():
    return send_from_directory('.', 'network_map.graphml')

@app.route('/api/user-archive/<username>')
def get_user_archive(username):
    url = f'https://fabxmporizzqflnftavs.supabase.co/storage/v1/object/public/archives/{username.lower()}/archive.json'
    print(f"Fetching data for {username} from URL: {url}")
    response = requests.get(url)
    data = response.json()
    print(f"\nData received for {username}:")
    print("Keys in data:", data.keys())
    print("\nExample tweet structure:")
    if data.get('tweets') and len(data['tweets']) > 0:
        print("First tweet full structure:", json.dumps(data['tweets'][0], indent=2))
    return jsonify(data)

@app.route('/public/<path:filename>')
def serve_public(filename):
    return send_from_directory('public', filename)

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start_index = 0
    
    while start_index < len(text):
        end_index = min(start_index + chunk_size, len(text))
        chunk = text[start_index:end_index]
        chunks.append(chunk)
        
        if end_index == len(text):
            break
            
        start_index = end_index - overlap
        
        # If remaining text is too small for a meaningful chunk, break
        if len(text) - start_index < chunk_size / 2:
            break
    
    return chunks

def summarize_chunk(client, chunk, user1, user2, chunk_index):
    """Generate summary for a single chunk of tweets."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": "Analyze the relationship between two Twitter users based on their liked tweets. Focus on identifying patterns, shared interests, and interaction styles."
            }, {
                "role": "user",
                "content": f"Analyze this set of tweets between {user1} and {user2} (part {chunk_index + 1}):\n\n{chunk}"
            }],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error summarizing chunk {chunk_index}: {str(e)}")
        raise

def merge_summaries(client, summaries, user1, user2):
    """Merge multiple chunk summaries into a final summary."""
    try:
        summaries_text = "\n\n".join([f"Part {i+1}:\n{s}" for i, s in enumerate(summaries)])
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "system",
                "content": "Create two concise summaries of the relationship between two Twitter users based on their liked tweets. Each summary should focus on what one user appreciates about the other's content."
            }, {
                "role": "user",
                "content": f"Based on these analysis parts, create two separate summaries:\n\n{summaries_text}\n\nProvide two summaries:\n1. What {user2} appreciates about {user1}'s content\n2. What {user1} appreciates about {user2}'s content"
            }],
            temperature=0.3
        )
        
        summaries = response.choices[0].message.content.split('\n\n')
        return {
            'summary1': summaries[0].strip(),
            'summary2': summaries[1].strip() if len(summaries) > 1 else ''
        }
    except Exception as e:
        print(f"Error merging summaries: {str(e)}")
        raise

@app.route('/api/summarize-relationship', methods=['POST'])
def summarize_relationship():
    try:
        data = request.json
        tweets1 = data.get('tweets1', [])
        tweets2 = data.get('tweets2', [])
        user1 = data.get('user1')
        user2 = data.get('user2')

        # Prepare tweet texts
        tweets1_text = "\n".join([t['tweet']['full_text'] for t in tweets1 if t.get('tweet', {}).get('full_text')])
        tweets2_text = "\n".join([t['tweet']['full_text'] for t in tweets2 if t.get('tweet', {}).get('full_text')])
        
        # Combine both sets of tweets
        combined_text = f"Tweets by {user1} liked by {user2}:\n{tweets1_text}\n\nTweets by {user2} liked by {user1}:\n{tweets2_text}"
        
        # Split into chunks if text is too long
        chunks = chunk_text(combined_text)
        print(f"Split text into {len(chunks)} chunks")

        if not os.getenv('OPENAI_API_KEY'):
            return jsonify({'error': 'OpenAI API key not configured'}), 500

        # Process chunks
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            try:
                summary = summarize_chunk(client, chunk, user1, user2, i)
                chunk_summaries.append(summary)
            except Exception as e:
                print(f"Error processing chunk {i}: {str(e)}")
                return jsonify({'error': f'Error processing chunk {i}: {str(e)}'}), 500

        # If we only have one chunk, parse it directly
        if len(chunk_summaries) == 1:
            summaries = chunk_summaries[0].split('\n\n')
            return jsonify({
                'summary1': summaries[0].strip(),
                'summary2': summaries[1].strip() if len(summaries) > 1 else ''
            })
        
        # Merge summaries if we have multiple chunks
        try:
            final_summaries = merge_summaries(client, chunk_summaries, user1, user2)
            return jsonify(final_summaries)
        except Exception as e:
            print(f"Error in final merge: {str(e)}")
            return jsonify({'error': f'Error merging summaries: {str(e)}'}), 500

    except Exception as e:
        print(f"Error in summarize_relationship: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port) 
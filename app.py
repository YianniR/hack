from flask import Flask, request, jsonify, send_from_directory
from vector_store import VectorStore
import os
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

vector_store = VectorStore()

@app.route('/')
def serve_visualizer():
    return send_from_directory('.', 'visualizer.html')

@app.route('/network_map.graphml')
def serve_network_map():
    return send_from_directory('.', 'network_map.graphml')

@app.route('/data/<path:filename>')
def serve_data(filename):
    try:
        filepath = os.path.join('data', filename)
        print(f"\nTrying to serve: {filepath}")
        
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return "File not found", 404
            
        with open(filepath, 'r') as f:
            if filename.endswith('.json'):
                data = json.load(f)
                print(f"Successfully loaded JSON with {len(data)} entries")
                print("First entry:", next(iter(data.items())))
                
        return send_from_directory('data', filename)
    except Exception as e:
        print(f"Error serving {filename}: {e}")
        return str(e), 500

@app.route('/get_embedding', methods=['GET'])
def get_embedding():
    text = request.args.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        print(f"Generating embedding for text: {text}")
        embedding = vector_store.generate_embedding(text)
        print(f"Generated embedding length: {len(embedding)}")
        return jsonify({'embedding': embedding})
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Make sure the data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Print available routes
    print("\nAvailable routes:")
    for rule in app.url_map.iter_rules():
        print(f"{rule.endpoint}: {rule.rule}")
    
    # Check if embeddings file exists
    embeddings_path = os.path.join('data', 'user_embeddings.json')
    if not os.path.exists(embeddings_path):
        print(f"\nWarning: {embeddings_path} not found!")
    else:
        print(f"\nFound embeddings file: {embeddings_path}")
    
    app.run(debug=True, port=5000) 
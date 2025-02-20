from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)
CORS(app)

# Load the model globally but lazily
model = None

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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port) 
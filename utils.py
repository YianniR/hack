import pickle
import json

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {filename}")

def convert_to_string(value):
    if isinstance(value, dict):
        return json.dumps(value)
    return str(value)

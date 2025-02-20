# build_graph.py

import networkx as nx
import json
import pickle
import xml.etree.ElementTree as ET
from xml.dom import minidom

from .utils import load_pickle, save_pickle

def load_data(filename):
    return load_pickle(filename)

def build_graph(tweets):
    G = nx.DiGraph()
    
    for tweet in tweets:
        tweet_id = tweet['tweet_id']
        reply_to_id = tweet['reply_to_tweet_id']
        
        G.add_node(tweet_id, **tweet)  # Add all tweet data as node attributes
        
        if reply_to_id:
            G.add_edge(reply_to_id, tweet_id)
    
    return G

from config import TWEET_GRAPH_FILE, OUTPUT_DIR, DATA_DIR
import os

def save_graph_pickle(G, filename=TWEET_GRAPH_FILE):
    save_pickle(G, filename)
    print(f"Graph saved in pickle format to {filename}")
    print("You can now run the thread explorer with: python main.py visualise_threads")

def save_graph_graphml_with_subgraphs(G, filename=os.path.join(OUTPUT_DIR, 'tweet_graph.graphml')):
    def convert_to_string(value):
        if isinstance(value, dict):
            return json.dumps(value)
        return str(value)

    root = ET.Element('graphml')
    root.set('xmlns', 'http://graphml.graphdrawing.org/xmlns')
    
    # Add key definitions
    keys = set()
    for _, data in G.nodes(data=True):
        keys.update(data.keys())
    
    for key in keys:
        elem = ET.SubElement(root, 'key')
        elem.set('id', key)
        elem.set('for', 'node')
        elem.set('attr.name', key)
        elem.set('attr.type', 'string')

    # Create separate graphs for each weakly connected component
    for i, component in enumerate(nx.weakly_connected_components(G)):
        subgraph = G.subgraph(component)
        graph_elem = ET.SubElement(root, 'graph')
        graph_elem.set('id', f'g{i}')
        graph_elem.set('edgedefault', 'directed')

        for node, data in subgraph.nodes(data=True):
            node_elem = ET.SubElement(graph_elem, 'node')
            node_elem.set('id', str(node))
            for key, value in data.items():
                data_elem = ET.SubElement(node_elem, 'data')
                data_elem.set('key', key)
                data_elem.text = convert_to_string(value)

        for u, v in subgraph.edges():
            edge_elem = ET.SubElement(graph_elem, 'edge')
            edge_elem.set('source', str(u))
            edge_elem.set('target', str(v))

    # Create a pretty-printed XML string
    xml_string = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")

    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(xml_string)

    print(f"Graph saved in GraphML format with separate subgraphs to {filename}")

def main(args):
    print("Starting graph building process...")
    # Load tweets
    input_file = args.input if hasattr(args, 'input') and args.input else 'whole_archive_tweets.pkl'
    tweets = load_data(input_file)
    print(f"Loaded {len(tweets)} tweets from {input_file}.")
    
    # Build graph
    graph = build_graph(tweets)
    print(f"Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    
    # Save graph in both formats
    output_file = args.output if hasattr(args, 'output') and args.output else TWEET_GRAPH_FILE
    save_graph_pickle(graph, output_file)
    save_graph_graphml_with_subgraphs(graph)
    print("Graph building process completed.")

if __name__ == "__main__":
    main()

import argparse
import pickle

def build_graph(tweets):
    G = nx.DiGraph()
    
    for tweet in tweets:
        tweet_id = tweet['tweet_id']
        reply_to_id = tweet['reply_to_tweet_id']
        
        G.add_node(tweet_id, **tweet)  # Add all tweet data as node attributes
        
        if reply_to_id:
            G.add_edge(reply_to_id, tweet_id)
    
    return G

def save_graph(graph, filename):
    with open(filename, 'wb') as f:
        pickle.dump(graph, f)
    print(f"Graph saved to {filename}")

def load_tweets(filename):
    filepath = os.path.join(DATA_DIR, filename)
    try:
        with open(filepath, 'rb') as f:
            tweets = pickle.load(f)
        print(f"Loaded {len(tweets)} tweets from {filepath}")
        return tweets
    except FileNotFoundError:
        print(f"Error: File not found: {filepath}")
        return []

def main(args):
    input_file = args.input if args.input else 'whole_archive_tweets.pkl'
    output_file = args.output if args.output else 'graph.pkl'

    tweets = load_tweets(input_file)
    graph = build_graph(tweets)
    save_graph(graph, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build graph from tweet data")
    parser.add_argument("--input", help="Input data filename (default: whole_archive_tweets.pkl)")
    parser.add_argument("--output", help="Output graph filename (default: graph.pkl)")
    args = parser.parse_args()
    main(args)

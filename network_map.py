import os
import logging
import json
from typing import Dict, List, Set
from fetch_data import AccountFetcher, FollowFetcher
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class NetworkMapper:
    def __init__(self, limit: int = None):
        """
        Initialize the network mapper
        Args:
            limit: Maximum number of accounts to map (None for all accounts)
        """
        self.account_fetcher = AccountFetcher()
        self.follow_fetcher = FollowFetcher()
        self.limit = limit
        self.nodes: List[Dict] = []
        self.edges: List[Dict] = []
        self.processed_accounts: Set[str] = set()

    def fetch_accounts(self) -> List[Dict]:
        """Fetch all accounts up to the limit"""
        accounts = self.account_fetcher.fetch_all()
        if self.limit:
            accounts = accounts[:self.limit]
            logger.info(f"Limited to {self.limit} accounts")
        return accounts

    def build_network(self):
        """Build the network graph"""
        accounts = self.fetch_accounts()
        total_accounts = len(accounts)
        logger.info(f"Building network for {total_accounts} accounts")

        # Create nodes for all accounts
        account_ids = set()  # Keep track of valid account IDs
        for account in accounts:
            account_id = str(account['account_id'])
            account_ids.add(account_id)
            self.nodes.append({
                'id': account_id,
                'username': account['username'],
                'display_name': account.get('display_name', ''),
                'description': account.get('description', '')
            })

        # Create edges for follow relationships
        for account in accounts:
            account_id = str(account['account_id'])

            # Fetch followers and following
            followers = self.follow_fetcher.fetch_all_followers(account_id)
            following = self.follow_fetcher.fetch_all_following(account_id)

            # Add follower edges (only if both nodes exist)
            for follower_id in followers:
                str_follower_id = str(follower_id)
                if str_follower_id in account_ids:
                    self.edges.append({
                        'source': str_follower_id,
                        'target': account_id,
                        'type': 'follows'
                    })

            # Add following edges (only if both nodes exist)
            for following_id in following:
                str_following_id = str(following_id)
                if str_following_id in account_ids:
                    self.edges.append({
                        'source': account_id,
                        'target': str_following_id,
                        'type': 'follows'
                    })

            self.processed_accounts.add(account_id)

    def export_graphml(self, output_file: str):
        """
        Export the network as a GraphML file
        Args:
            output_file: Path to save the GraphML file
        """
        logger.info("Generating GraphML export...")
        
        # Create GraphML header
        graphml = """<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
    <key id="username" for="node" attr.name="username" attr.type="string"/>
    <key id="display_name" for="node" attr.name="display_name" attr.type="string"/>
    <key id="description" for="node" attr.name="description" attr.type="string"/>
    <graph id="G" edgedefault="directed">
"""

        # Add nodes
        for node in self.nodes:
            graphml += f"""        <node id="{node['id']}">
                <data key="username">{node['username']}</data>
                <data key="display_name">{node.get('display_name', '')}</data>
                <data key="description">{node.get('description', '')}</data>
            </node>
"""

        # Add edges
        for edge in self.edges:
            graphml += f'        <edge source="{edge["source"]}" target="{edge["target"]}"/>\n'

        # Close the graph
        graphml += """    </graph>
</graphml>"""

        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(graphml)
        logger.info(f"GraphML file saved to {output_file}")

    def export_json(self, output_file: str):
        """
        Export the network as JSON
        Args:
            output_file: Path to save the JSON file
        """
        logger.info("Generating JSON export...")
        network = {
            'nodes': self.nodes,
            'edges': self.edges
        }
        with open(output_file, 'w') as f:
            json.dump(network, f, indent=2)
        logger.info(f"Network JSON saved to {output_file}")

def main(args):
    mapper = NetworkMapper(limit=args.limit)
    mapper.build_network()
    
    # Export in both formats
    mapper.export_graphml(args.output + '.graphml')
    mapper.export_json(args.output + '.json')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build a social network map")
    parser.add_argument("--limit", type=int, help="Maximum number of accounts to process", default=None)
    parser.add_argument("--output", type=str, help="Output file path (without extension)", default="network_map")
    args = parser.parse_args()
    main(args) 
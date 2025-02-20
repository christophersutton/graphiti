"""
A minimal example script based on the Graphiti README for benchmark testing.
"""

import asyncio
from datetime import datetime, timezone

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

# Simple test episodes about Kamala Harris
episodes = [
    "Kamala Harris is the Attorney General of California. She was previously "
    "the district attorney for San Francisco.",
    "As AG, Harris was in office from January 3, 2011 – January 3, 2017",
]

async def main():
    # Initialize Graphiti with default localhost settings
    client = Graphiti("bolt://localhost:7687", "neo4j", "password")
    
    try:
        # Clear existing data and setup indices
        await clear_data(client.driver)
        await client.build_indices_and_constraints()

        # Add episodes
        for i, episode in enumerate(episodes):
            await client.add_episode(
                name=f"Freakonomics Radio {i}",
                episode_body=episode,
                source=EpisodeType.text,
                source_description="podcast",
                reference_time=datetime.now(timezone.utc)
            )

        # Do a simple search to verify everything works
        results = await client.search('Who was the California Attorney General?')
        
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(main()) 
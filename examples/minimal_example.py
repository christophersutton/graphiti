"""
A minimal example script based on the Graphiti README for benchmark testing.
"""

import asyncio
import logging
from datetime import datetime, timezone
import re

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

# Get logger but don't configure it (configuration handled by bench_runner.py)
logger = logging.getLogger(__name__)

# Expose graphiti_instance at module level for bench_runner.py to access
graphiti_instance = None

# Expose search_validation_result and answers at module level for bench_runner to access
search_validation_result = None
answer = None

# Simple test episodes about Kamala Harris
episodes = [
    "Kamala Harris is the Attorney General of California. She was previously ",
    "the district attorney for San Francisco.",
    "As AG, Harris was in office from January 3, 2011 – January 3, 2017",
]

def extract_fact(result_str):
    """Extract just the fact from a search result string."""
    # Look for fact='...' pattern
    match = re.search(r"fact='([^']*)'", result_str)
    if match:
        return match.group(1)
    return None

async def main():
    logger.info("Initializing Graphiti instance...")
    # Initialize Graphiti with default localhost settings
    global graphiti_instance, answer
    graphiti_instance = Graphiti("bolt://localhost:7687", "neo4j", "password")
    
    try:
        # Clear existing data and setup indices
        logger.info("Clearing existing data...")
        await clear_data(graphiti_instance.driver)
        logger.info("Building indices and constraints...")
        await graphiti_instance.build_indices_and_constraints()

        # Add episodes
        for i, episode in enumerate(episodes):
            logger.info(f"Adding episode {i}: {episode[:50]}...")
            await graphiti_instance.add_episode(
                name=f"Freakonomics Radio {i}",
                episode_body=episode,
                source=EpisodeType.text,
                source_description="podcast",
                reference_time=datetime.now(timezone.utc)
            )
            logger.debug(f"Episode {i} added successfully")

        # Do a simple search to verify everything works
        logger.info("Performing test search query...")
        results = await graphiti_instance.search('Who was the California Attorney General?')
        logger.info(f"Search results found: {len(results)} results")
        
        # Log and store all results
        all_facts = []
        logger.info("Search results:")
        for i, result in enumerate(results):
            fact = extract_fact(str(result))
            if fact:
                all_facts.append(fact)
                logger.info(f"Result {i+1}: {fact}")
            logger.debug(f"Full result {i+1}: {result}")
        
        # Store all facts as the answer
        answer = all_facts
        logger.info(f"All extracted facts: {answer}")
        
        # Validate search results contain Kamala Harris
        validation_result = any('Kamala Harris' in str(result) for result in results)
        logger.info(f"Search validation result: {'passed' if validation_result else 'failed'} - Looking for 'Kamala Harris' in results")
        
        # Store validation result at module level for bench_runner to access
        global search_validation_result
        search_validation_result = validation_result
        
    finally:
        logger.info("Closing Graphiti instance...")
        await graphiti_instance.close()

if __name__ == "__main__":
    asyncio.run(main()) 
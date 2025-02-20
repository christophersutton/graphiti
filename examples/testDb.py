from neo4j import GraphDatabase, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()

async def clear_data(driver):
    """Clear all data from the Neo4j database."""
    async with driver.session() as session:
        await session.run('MATCH (n) DETACH DELETE n')

async def build_indices_and_constraints(driver, delete_existing: bool = False):
    """Build indices and constraints in the Neo4j database."""
    if delete_existing:
        result = await driver.execute_query(
            """
            SHOW INDEXES YIELD name
            """,
            database_="neo4j",
        )
        index_names = [record['name'] for record in result.records]
        for name in index_names:
            await driver.execute_query(
                """DROP INDEX $name""",
                name=name,
                database_="neo4j",
            )

    range_indices = [
        'CREATE INDEX entity_uuid IF NOT EXISTS FOR (n:Entity) ON (n.uuid)',
        'CREATE INDEX episode_uuid IF NOT EXISTS FOR (n:Episodic) ON (n.uuid)',
        'CREATE INDEX community_uuid IF NOT EXISTS FOR (n:Community) ON (n.uuid)',
        'CREATE INDEX relation_uuid IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.uuid)',
        'CREATE INDEX mention_uuid IF NOT EXISTS FOR ()-[e:MENTIONS]-() ON (e.uuid)',
        'CREATE INDEX has_member_uuid IF NOT EXISTS FOR ()-[e:HAS_MEMBER]-() ON (e.uuid)',
        'CREATE INDEX entity_group_id IF NOT EXISTS FOR (n:Entity) ON (n.group_id)',
        'CREATE INDEX episode_group_id IF NOT EXISTS FOR (n:Episodic) ON (n.group_id)',
        'CREATE INDEX relation_group_id IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.group_id)',
        'CREATE INDEX mention_group_id IF NOT EXISTS FOR ()-[e:MENTIONS]-() ON (e.group_id)',
        'CREATE INDEX name_entity_index IF NOT EXISTS FOR (n:Entity) ON (n.name)',
        'CREATE INDEX created_at_entity_index IF NOT EXISTS FOR (n:Entity) ON (n.created_at)',
        'CREATE INDEX created_at_episodic_index IF NOT EXISTS FOR (n:Episodic) ON (n.created_at)',
        'CREATE INDEX valid_at_episodic_index IF NOT EXISTS FOR (n:Episodic) ON (n.valid_at)',
        'CREATE INDEX name_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.name)',
        'CREATE INDEX created_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.created_at)',
        'CREATE INDEX expired_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.expired_at)',
        'CREATE INDEX valid_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.valid_at)',
        'CREATE INDEX invalid_at_edge_index IF NOT EXISTS FOR ()-[e:RELATES_TO]-() ON (e.invalid_at)',
    ]

    fulltext_indices = [
        """CREATE FULLTEXT INDEX node_name_and_summary IF NOT EXISTS 
        FOR (n:Entity) ON EACH [n.name, n.summary, n.group_id]""",
        """CREATE FULLTEXT INDEX community_name IF NOT EXISTS 
        FOR (n:Community) ON EACH [n.name, n.group_id]""",
        """CREATE FULLTEXT INDEX edge_name_and_fact IF NOT EXISTS 
        FOR ()-[e:RELATES_TO]-() ON EACH [e.name, e.fact, e.group_id]""",
    ]

    index_queries = range_indices + fulltext_indices

    for query in index_queries:
        await driver.execute_query(
            query,
            database_="neo4j",
        )

async def async_test_neo4j_connection(clear_db=False):
    """
    Test connection to Neo4j and optionally clear the database and rebuild indices.
    
    Args:
        clear_db (bool): If True, clears all data and rebuilds indices
    """
    uri = os.environ.get('NEO4J_URI') or 'bolt://localhost:7687'
    user = os.environ.get('NEO4J_USER') or 'neo4j'
    password = os.environ.get('NEO4J_PASSWORD') or 'password'
    try:
        driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        # Verify connectivity
        await driver.verify_connectivity()
        print("✅ Successfully connected to Neo4j!")
        
        if clear_db:
            print("🗑️  Clearing database...")
            await clear_data(driver)
            print("🔄 Rebuilding indices...")
            await build_indices_and_constraints(driver)
            print("✅ Database cleared and indices rebuilt!")
        
        # Try a simple query
        result = await driver.execute_query("RETURN 1 AS num")
        print("✅ Successfully executed test query!")
            
        await driver.close()
        return True
        
    except ServiceUnavailable:
        print("❌ Failed to connect to Neo4j. Please check if:")
        print("  - Neo4j is running")
        print("  - The connection details are correct")
        print(f"  - The database is accessible at {uri}")
        return False
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        return False

def test_neo4j_connection(clear_db=False):
    """Synchronous wrapper for async_test_neo4j_connection"""
    return asyncio.run(async_test_neo4j_connection(clear_db))

if __name__ == "__main__":
    test_neo4j_connection(clear_db=True)  # Set clear_db=True to clear and rebuild 
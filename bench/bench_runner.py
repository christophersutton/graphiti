import argparse
import json
import time
import importlib.util
import asyncio
import os
import logging
from datetime import datetime
from pathlib import Path
from neo4j import GraphDatabase, AsyncGraphDatabase
from dotenv import load_dotenv
from graphiti_core.utils.maintenance.graph_data_operations import clear_data
import asyncpg

neo4j_uri = os.environ.get('NEO4J_URI') or 'bolt://localhost:7687'
neo4j_user = os.environ.get('NEO4J_USER') or 'neo4j'
neo4j_password = os.environ.get('NEO4J_PASSWORD') or 'password'
postgres_url = os.environ.get('DATABASE_URL') or 'postgresql://postgres:postgres@localhost:5432/litellm'
load_dotenv()

# Enhanced logging configuration
def setup_logging(debug=False):
    # First remove all handlers associated with the root logger object
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Set root logger to lowest level so child loggers control their own levels
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # Always set to DEBUG to allow all messages through
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Configure specific loggers
    for logger_name in [
        'graphiti_core',
        'graphiti_core.llm_client',
        'graphiti_core.llm_client.client',
        'graphiti_core.llm_client.openai_client'
    ]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = True
        logger.debug(f"Configured logger: {logger_name}")
    
    # Set Neo4j logging to WARN level
    neo4j_logger = logging.getLogger('neo4j')
    neo4j_logger.setLevel(logging.WARN)
    neo4j_logger.propagate = True
    
    # Also silence Neo4j driver's debug logs
    logging.getLogger('neo4j.io').setLevel(logging.WARN)
    logging.getLogger('neo4j.bolt').setLevel(logging.WARN)
    
    # Set httpx to INFO to reduce noise
    logging.getLogger('httpx').setLevel(logging.INFO)
    logging.getLogger('httpcore').setLevel(logging.INFO)

# Set up logging immediately
setup_logging(True)  # Always use debug logging for now
logger = logging.getLogger(__name__)


async def count_nodes_edges():
    """Counts nodes and edges in the Neo4j database and returns detailed information."""
    driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    try:
        async with driver.session() as session:
            # Get node counts by label
            node_result = await session.run("""
                MATCH (n)
                WITH labels(n) as labels, count(*) as count
                RETURN labels, count
            """)
            node_counts = {','.join(record['labels']) or 'unlabeled': record['count'] 
                         async for record in node_result}

            # Get relationship counts by type with source and target node types
            rel_result = await session.run("""
                MATCH (source)-[r]->(target)
                WITH 
                    type(r) as rel_type,
                    labels(source) as source_labels,
                    labels(target) as target_labels,
                    count(*) as count
                RETURN rel_type, source_labels, target_labels, count
            """)
            edge_counts = {record['rel_type']: record['count'] 
                         async for record in rel_result}
            
            # Get detailed edge paths with node names where available
            edge_paths_result = await session.run("""
                MATCH (source)-[r]->(target)
                RETURN 
                    CASE
                        WHEN source.name IS NOT NULL THEN source.name
                        ELSE labels(source)[0] + '_' + id(source)
                    END as source_name,
                    labels(source) as source_labels,
                    type(r) as relationship,
                    CASE
                        WHEN target.name IS NOT NULL THEN target.name
                        ELSE labels(target)[0] + '_' + id(target)
                    END as target_name,
                    labels(target) as target_labels
                ORDER BY source_name, relationship, target_name
            """)
            
            edge_paths = []
            async for record in edge_paths_result:
                edge_paths.append({
                    'source': {
                        'name': record['source_name'],
                        'labels': record['source_labels']
                    },
                    'relationship': record['relationship'],
                    'target': {
                        'name': record['target_name'],
                        'labels': record['target_labels']
                    },
                    'full_path': f"{record['source_name']} -[{record['relationship']}]-> {record['target_name']}"
                })

            # Get entity names
            entity_result = await session.run("""
                MATCH (n:Entity)
                RETURN n.name as name
                ORDER BY n.name
            """)
            entity_names = [record['name'] async for record in entity_result]

            # Get episodic details
            episodic_result = await session.run("""
                MATCH (n:Episodic)
                RETURN n.name as name, n.description as description
                ORDER BY n.name
            """)
            episodic_details = [{
                'name': record['name'],
                'description': record['description']
            } async for record in episodic_result]

            # Get total counts
            total_nodes = sum(node_counts.values())
            total_edges = sum(edge_counts.values())

        return {
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'node_types': node_counts,
            'edge_types': edge_counts,
            'edge_paths': edge_paths,
            'entity_names': entity_names,
            'episodic_details': episodic_details
        }
    finally:
        await driver.close()

async def get_llm_metrics(start_time: datetime, end_time: datetime) -> dict:
    """Get LLM usage metrics from Postgres between two timestamps."""
    try:
        conn = await asyncpg.connect(postgres_url)
        query = """
            SELECT 
                call_type,
                COUNT(*) as call_count,
                SUM(spend) as total_spend,
                SUM(prompt_tokens) as prompt_tokens,
                SUM(completion_tokens) as completion_tokens,
                SUM(total_tokens) as total_tokens
            FROM public."LiteLLM_SpendLogs"
            WHERE "startTime" >= $1 AND "startTime" <= $2
            GROUP BY call_type
        """
        rows = await conn.fetch(query, start_time, end_time)
        await conn.close()
        
        return [{
            'call_type': row['call_type'],
            'call_count': row['call_count'],
            'total_spend': float(row['total_spend']) if row['total_spend'] else 0.0,
            'prompt_tokens': row['prompt_tokens'],
            'completion_tokens': row['completion_tokens'],
            'total_tokens': row['total_tokens']
        } for row in rows]
    except Exception as e:
        logger.error(f"Error fetching LLM metrics: {e}")
        return []

async def run_benchmark(filepath, max_reflexion_iterations):
    """Runs the example file and benchmarks its performance."""
    logger.info(f"Starting benchmark run for {filepath}")
    logger.debug(f"Max reflexion iterations set to: {max_reflexion_iterations}")

    module_name = filepath.replace(os.sep, '.').rstrip('.py')
    logger.debug(f"Loading module: {module_name}")
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None:
        raise ImportError(f"Failed to import module from path: {filepath}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, 'main'):
        raise AttributeError(f"Module {filepath} does not have a main function.")

    main_func = getattr(module, 'main')
    if not asyncio.iscoroutinefunction(main_func):
        raise TypeError(f"Module {filepath} main function is not asynchronous.")

    # Clear the database before running the benchmark
    driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    try:
        logger.info("Clearing database before benchmark run...")
        await clear_data(driver)
        logger.debug("Database cleared successfully")
    finally:
        await driver.close()

    logger.info("Starting benchmark execution...")
    start_time = datetime.utcnow()
    initial_counts = await count_nodes_edges()
    logger.debug(f"Initial database state: {initial_counts}")
    
    await main_func() 
    
    end_time = datetime.utcnow()
    final_counts = await count_nodes_edges()
    logger.debug(f"Final database state: {final_counts}")
    elapsed_time = (end_time - start_time).total_seconds()
    logger.info(f"Benchmark execution completed in {elapsed_time:.2f} seconds")

    # Get LLM usage metrics
    llm_metrics = await get_llm_metrics(start_time, end_time)
    logger.info(f"LLM usage metrics: {llm_metrics}")

    # Get search validation result and answer if available
    search_validation_result = getattr(module, 'search_validation_result', None)
    answer = getattr(module, 'answer', None)
    if search_validation_result is not None:
        logger.info(f"Search validation result: {'passed' if search_validation_result else 'failed'}")
    if answer is not None:
        logger.info(f"Generated answer: {answer}")
    
    # Calculate node differences with detailed logging
    node_type_diff = {}
    for node_type in set(initial_counts['node_types'].keys()) | set(final_counts['node_types'].keys()):
        initial = initial_counts['node_types'].get(node_type, 0)
        final = final_counts['node_types'].get(node_type, 0)
        if final - initial > 0:
            node_type_diff[node_type] = final - initial
            logger.debug(f"Node type '{node_type}' increased by {final - initial}")

    # Calculate edge differences with detailed logging
    edge_type_diff = {}
    edge_details_diff = []
    
    # Track edge type changes
    for edge_type in set(initial_counts['edge_types'].keys()) | set(final_counts['edge_types'].keys()):
        initial = initial_counts['edge_types'].get(edge_type, 0)
        final = final_counts['edge_types'].get(edge_type, 0)
        if final - initial > 0:
            edge_type_diff[edge_type] = final - initial
            logger.debug(f"Edge type '{edge_type}' increased by {final - initial}")
    
    # Track detailed edge information
    final_edge_details = final_counts['edge_paths']
    initial_edge_types = {detail['relationship']: detail for detail in initial_counts['edge_paths']}
    
    for detail in final_edge_details:
        edge_type = detail['relationship']
        initial_detail = initial_edge_types.get(edge_type, {'full_path': ''})
        if detail['full_path'] != initial_detail['full_path']:
            edge_details_diff.append({
                'type': edge_type,
                'new_connections': 1,
                'source_type': ','.join(detail['source']['labels']) or 'unlabeled',
                'target_type': ','.join(detail['target']['labels']) or 'unlabeled',
                'full_path': detail['full_path']
            })

    return {
        "execution_time": elapsed_time,
        "nodes_created": final_counts['total_nodes'] - initial_counts['total_nodes'],
        "edges_created": final_counts['total_edges'] - initial_counts['total_edges'],
        "node_types_created": node_type_diff,
        "edge_types_created": edge_type_diff,
        "edge_details": edge_details_diff,
        "entity_names": final_counts['entity_names'],
        "episodic_details": final_counts['episodic_details'],
        "search_validation": {
            "performed": search_validation_result is not None,
            "passed": search_validation_result if search_validation_result is not None else None,
            "answer": answer if answer is not None else None
        },
        "llm_metrics": llm_metrics
    }

async def benchmark_runner(filepath, runs, max_reflexion_iterations, output_file=None):
    """Runs the benchmark multiple times and aggregates results."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Example file not found: {filepath}")

    if not filepath.endswith(".py"):
        raise ValueError(f"Invalid filepath: {filepath}. Must be a Python file (.py).")

    # Create output directory if it doesn't exist
    out_dir = Path("bench/out")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate default output filename if none provided
    if output_file is None:
        input_filename = Path(filepath).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        use_spacy = os.environ.get('USE_SPACY_PIPELINE', 'false').lower() == 'true'
        prefix = 'spacy_' if use_spacy else ''
        output_file = out_dir / f"{prefix}{input_filename}_r{runs}_{timestamp}.json"

    logger.info(f"Benchmarking example: {filepath}, runs: {runs}, MAX_REFLEXION_ITERATIONS: {max_reflexion_iterations}")

    run_metrics = []
    successful_runs = 0
    error_message = None

    # Initialize aggregated LLM metrics
    aggregated_llm_metrics = {}

    for i in range(runs):
        try:
            logger.info(f"Run {i+1}/{runs}...")
            metrics = await run_benchmark(filepath, max_reflexion_iterations)
            metrics["success"] = True
            run_metrics.append(metrics)
            successful_runs += 1

            # Aggregate LLM metrics
            for llm_metric in metrics.get("llm_metrics", []):
                call_type = llm_metric["call_type"]
                if call_type not in aggregated_llm_metrics:
                    aggregated_llm_metrics[call_type] = {
                        "total_calls": 0,
                        "total_spend": 0.0,
                        "total_prompt_tokens": 0,
                        "total_completion_tokens": 0,
                        "total_tokens": 0
                    }
                agg = aggregated_llm_metrics[call_type]
                agg["total_calls"] += llm_metric["call_count"]
                agg["total_spend"] += llm_metric["total_spend"]
                agg["total_prompt_tokens"] += llm_metric["prompt_tokens"]
                agg["total_completion_tokens"] += llm_metric["completion_tokens"]
                agg["total_tokens"] += llm_metric["total_tokens"]

        except Exception as e:
            error_msg = f"Error in run {i+1}: {str(e)}"
            logger.error(error_msg)
            run_metrics.append({
                "success": False,
                "error": error_msg,
                "run_number": i + 1
            })
            error_message = error_msg

    # Calculate aggregated metrics only for successful runs
    aggregated_metrics = {}
    if successful_runs > 0:
        successful_metrics = [m for m in run_metrics if m.get("success", False)]
        aggregated_metrics = {
            "average_time": sum(m["execution_time"] for m in successful_metrics) / successful_runs,
            "average_nodes": sum(m["nodes_created"] for m in successful_metrics) / successful_runs,
            "average_edges": sum(m["edges_created"] for m in successful_metrics) / successful_runs,
            "llm_metrics": {
                call_type: {
                    "average_calls": metrics["total_calls"] / successful_runs,
                    "average_spend": metrics["total_spend"] / successful_runs,
                    "average_prompt_tokens": metrics["total_prompt_tokens"] / successful_runs,
                    "average_completion_tokens": metrics["total_completion_tokens"] / successful_runs,
                    "average_tokens": metrics["total_tokens"] / successful_runs,
                    "total_calls": metrics["total_calls"],
                    "total_spend": metrics["total_spend"],
                    "total_prompt_tokens": metrics["total_prompt_tokens"],
                    "total_completion_tokens": metrics["total_completion_tokens"],
                    "total_tokens": metrics["total_tokens"]
                }
                for call_type, metrics in aggregated_llm_metrics.items()
            },
            "search_validation": {
                "successful_validations": sum(1 for m in successful_metrics if m.get("search_validation", {}).get("passed", False)),
                "total_validations": sum(1 for m in successful_metrics if m.get("search_validation", {}).get("performed", False)),
                "answers": [m.get("search_validation", {}).get("answer") for m in successful_metrics]
            }
        }

    output_json = {
        "filepath": filepath,
        "total_runs_attempted": runs,
        "successful_runs": successful_runs,
        "max_reflexion_iterations": max_reflexion_iterations,
        "per_run_metrics": run_metrics,
        "aggregated_metrics": aggregated_metrics,
        "overall_success": successful_runs == runs,
        "error": error_message
    }

    json_output = json.dumps(output_json, indent=4)

    # Always write to file (either user-specified or auto-generated path)
    try:
        with open(output_file, 'w') as f:
            f.write(json_output)
        logger.info(f"Benchmark results written to: {output_file}")
    except Exception as e:
        logger.error(f"Error writing to output file: {e}")
    
    # Also print to stdout for immediate feedback
    print(json_output)

    # If no successful runs, raise an exception
    if successful_runs == 0:
        raise RuntimeError(f"All benchmark runs failed. Last error: {error_message}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark runner for Graphiti examples.")
    parser.add_argument("filepath", help="Path to the example file to benchmark.")
    parser.add_argument("--runs", type=int, default=1, help="Number of benchmark runs (default: 1).")
    parser.add_argument(
        "--max_reflexion_iterations",
        type=int,
        default=2,
        help="Maximum number of reflexion iterations (default: 2).",
    )
    parser.add_argument(
        "--output_file",
        help="Optional filepath to write JSON output to. If not provided, output is printed to stdout.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()
    
    # Setup logging based on debug flag
    setup_logging(args.debug)

    try:
        asyncio.run(
            benchmark_runner(
                args.filepath, args.runs, args.max_reflexion_iterations, args.output_file
            )
        )
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
    except AttributeError as e:
        logger.error(f"Error: {e}")
    except TypeError as e:
        logger.error(f"Error: {e}")
    except ValueError as e:
        logger.error(f"Error: {e}")
    except ImportError as e:
        logger.error(f"Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
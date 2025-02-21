import json
import sys
from pathlib import Path
from typing import Dict, Any, Set, List, Tuple
import csv
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict

@dataclass
class Comparison:
    metric_name: str
    base_value: float  # non-spacy value
    spacy_value: float  # spacy value
    abs_diff: float
    pct_diff: float

@dataclass
class EdgeInfo:
    type: str
    source_type: str
    target_type: str
    full_path: str

def is_spacy_file(filepath: str) -> bool:
    return Path(filepath).stem.startswith('spacy_')

def order_files(file1: str, file2: str) -> Tuple[str, str]:
    """Returns (base_file, spacy_file)"""
    if is_spacy_file(file1):
        return file2, file1
    return file1, file2

def load_json(filepath: str) -> Dict[str, Any]:
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_diff(base_val: float, spacy_val: float) -> tuple[float, float]:
    """Calculate difference with spacy relative to base"""
    abs_diff = spacy_val - base_val
    pct_diff = ((spacy_val - base_val) / base_val * 100) if base_val != 0 else float('inf')
    return abs_diff, pct_diff

def extract_edges(data: Dict[str, Any]) -> List[EdgeInfo]:
    edges = []
    for run in data["per_run_metrics"]:
        for edge in run["edge_details"]:
            edges.append(EdgeInfo(
                type=edge["type"],
                source_type=edge["source_type"],
                target_type=edge["target_type"],
                full_path=edge["full_path"]
            ))
    return edges

def analyze_edges(base_edges: List[EdgeInfo], spacy_edges: List[EdgeInfo]) -> Dict[str, Any]:
    # Get unique edges and sort by type then path
    base_edges_unique = {e.full_path: e for e in base_edges}.values()
    spacy_edges_unique = {e.full_path: e for e in spacy_edges}.values()
    
    base_paths = sorted([e.full_path for e in base_edges_unique], 
                       key=lambda p: (next(e.type for e in base_edges_unique if e.full_path == p), p))
    spacy_paths = sorted([e.full_path for e in spacy_edges_unique],
                        key=lambda p: (next(e.type for e in spacy_edges_unique if e.full_path == p), p))
    
    # Count edge types
    base_types = defaultdict(int)
    spacy_types = defaultdict(int)
    for e in base_edges:
        base_types[e.type] += 1
    for e in spacy_edges:
        spacy_types[e.type] += 1
    
    # Get the maximum length for alignment
    max_edges = max(len(base_paths), len(spacy_paths))
    
    # Pad the shorter list with None to match lengths
    base_aligned = base_paths + [None] * (max_edges - len(base_paths))
    spacy_aligned = spacy_paths + [None] * (max_edges - len(spacy_paths))
        
    return {
        "base_edges": base_paths,
        "spacy_edges": spacy_paths,
        "aligned_edges": list(zip(base_aligned, spacy_aligned)),
        "base_edge_types": dict(base_types),
        "spacy_edge_types": dict(spacy_types),
        "common_edges": len(set(base_paths) & set(spacy_paths))
    }

def compare_metrics(base_data: Dict[str, Any], spacy_data: Dict[str, Any]) -> list[Comparison]:
    comparisons = []
    
    # Compare basic metrics
    metrics = {
        "execution_time": lambda x: x["aggregated_metrics"]["average_time"],
        "nodes_created": lambda x: x["aggregated_metrics"]["average_nodes"],
        "edges_created": lambda x: x["aggregated_metrics"]["average_edges"],
        "success_rate": lambda x: (x["successful_runs"] / x["total_runs_attempted"]) * 100,
    }
    
    # Compare LLM metrics
    for call_type in ["acompletion", "aembedding"]:
        metrics.update({
            f"{call_type}_calls": lambda x, ct=call_type: x["aggregated_metrics"]["llm_metrics"][ct]["average_calls"],
            f"{call_type}_spend": lambda x, ct=call_type: x["aggregated_metrics"]["llm_metrics"][ct]["average_spend"],
            f"{call_type}_tokens": lambda x, ct=call_type: x["aggregated_metrics"]["llm_metrics"][ct]["average_tokens"],
        })

    for metric_name, getter in metrics.items():
        try:
            base_val = getter(base_data)
            spacy_val = getter(spacy_data)
            abs_diff, pct_diff = calculate_diff(base_val, spacy_val)
            comparisons.append(Comparison(
                metric_name=metric_name,
                base_value=base_val,
                spacy_value=spacy_val,
                abs_diff=abs_diff,
                pct_diff=pct_diff
            ))
        except (KeyError, ZeroDivisionError):
            continue

    return comparisons

def write_csv_report(comparisons: list[Comparison], output_path: str):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Base Value', 'Spacy Value', 'Absolute Diff', 'Percent Diff'])
        for comp in comparisons:
            writer.writerow([
                comp.metric_name,
                f"{comp.base_value:.4f}",
                f"{comp.spacy_value:.4f}",
                f"{comp.abs_diff:.4f}",
                f"{comp.pct_diff:.2f}%"
            ])

def write_json_report(comparisons: list[Comparison], output_path: str):
    report = {comp.metric_name: {
        "base_value": comp.base_value,
        "spacy_value": comp.spacy_value,
        "absolute_diff": comp.abs_diff,
        "percent_diff": comp.pct_diff
    } for comp in comparisons}
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

def format_metric_table(comparisons: List[Comparison], metrics: List[str]) -> str:
    rows = []
    rows.append("| Metric | Base | Spacy | Absolute Diff | % Change |")
    rows.append("|--------|------|-------|---------------|----------|")
    
    for metric in metrics:
        for comp in comparisons:
            if comp.metric_name == metric:
                rows.append(f"| {comp.metric_name} | {comp.base_value:.4f} | {comp.spacy_value:.4f} | {comp.abs_diff:+.4f} | {comp.pct_diff:+.2f}% |")
    
    return "\n".join(rows)

def write_markdown_report(comparisons: list[Comparison], edge_analysis: Dict[str, Any], base_path: str, spacy_path: str, output_path: str):
    base_name = Path(base_path).stem
    spacy_name = Path(spacy_path).stem
    
    sections = []
    
    # Header
    sections.append(f"# Benchmark Comparison Report\n")
    sections.append(f"Comparing base (`{base_name}`) vs spaCy (`{spacy_name}`)\n")
    
    # Execution Metrics
    sections.append("## Execution Metrics\n")
    exec_metrics = ["execution_time", "success_rate"]
    sections.append(format_metric_table(comparisons, exec_metrics))
    sections.append("\n")
    
    # Graph Metrics
    sections.append("## Graph Structure Metrics\n")
    graph_metrics = ["nodes_created", "edges_created"]
    sections.append(format_metric_table(comparisons, graph_metrics))
    sections.append("\n")
    
    # Edge Analysis
    sections.append("## Edge Analysis\n")
    sections.append(f"Total unique edges - Base: {len(edge_analysis['base_edges'])}, spaCy: {len(edge_analysis['spacy_edges'])}")
    sections.append(f"Common edges between runs: {edge_analysis['common_edges']}\n")
    
    # Complete edge lists side by side
    sections.append("### Complete Edge Lists\n")
    sections.append("| Base | spaCy |")
    sections.append("|------|-------|")
    for base_edge, spacy_edge in edge_analysis['aligned_edges']:
        base_str = f"`{base_edge}`" if base_edge else ""
        spacy_str = f"`{spacy_edge}`" if spacy_edge else ""
        sections.append(f"| {base_str} | {spacy_str} |")
    sections.append("\n")
    
    # Edge Type Distribution
    sections.append("### Edge Type Distribution\n")
    sections.append("| Edge Type | Base | Spacy |")
    sections.append("|-----------|------|-------|")
    all_types = set(edge_analysis['base_edge_types'].keys()) | set(edge_analysis['spacy_edge_types'].keys())
    for edge_type in sorted(all_types):
        count1 = edge_analysis['base_edge_types'].get(edge_type, 0)
        count2 = edge_analysis['spacy_edge_types'].get(edge_type, 0)
        sections.append(f"| {edge_type} | {count1} | {count2} |")
    sections.append("\n")
    
    # Search Validation Results
    sections.append("## Search Validation Results\n")
    base_data = load_json(base_path)
    spacy_data = load_json(spacy_path)
    
    base_validation = base_data.get('aggregated_metrics', {}).get('search_validation', {})
    spacy_validation = spacy_data.get('aggregated_metrics', {}).get('search_validation', {})
    
    # Success rate
    base_success = f"{base_validation.get('successful_validations', 0)}/{base_validation.get('total_validations', 0)}"
    spacy_success = f"{spacy_validation.get('successful_validations', 0)}/{spacy_validation.get('total_validations', 0)}"
    sections.append(f"Success rate - Base: {base_success}, spaCy: {spacy_success}\n")
    
    # Answers by run
    sections.append("### Answers by Run\n")
    sections.append("| Run # | Base Answers | spaCy Answers |")
    sections.append("|-------|--------------|---------------|")
    
    base_answers = base_validation.get('answers', [])
    spacy_answers = spacy_validation.get('answers', [])
    max_runs = max(len(base_answers), len(spacy_answers))
    
    for i in range(max_runs):
        base_ans = base_answers[i] if i < len(base_answers) else []
        spacy_ans = spacy_answers[i] if i < len(spacy_answers) else []
        
        # Format each answer in the array with code blocks and line breaks
        base_formatted = "<br>".join(f"`{ans}`" for ans in (base_ans if isinstance(base_ans, list) else [base_ans]) if ans)
        spacy_formatted = "<br>".join(f"`{ans}`" for ans in (spacy_ans if isinstance(spacy_ans, list) else [spacy_ans]) if ans)
        
        sections.append(f"| {i + 1} | {base_formatted} | {spacy_formatted} |")
    sections.append("\n")
    
    # LLM Metrics
    sections.append("## LLM Usage Metrics\n")
    llm_metrics = [m for m in [c.metric_name for c in comparisons] if m.startswith(("acompletion", "aembedding"))]
    sections.append(format_metric_table(comparisons, llm_metrics))
    
    with open(output_path, 'w') as f:
        f.write("\n".join(sections))

def find_benchmark_files(directory: str) -> Tuple[str, str]:
    """Find and validate benchmark files in the given directory.
    Returns (base_file, spacy_file) if valid files are found.
    Raises ValueError if validation fails."""
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise ValueError(f"'{directory}' is not a valid directory")
        
    json_files = list(dir_path.glob("*.json"))
    if len(json_files) < 2:
        raise ValueError(f"Directory must contain at least 2 JSON files, found {len(json_files)}")
        
    spacy_files = [f for f in json_files if is_spacy_file(f.name)]
    non_spacy_files = [f for f in json_files if not is_spacy_file(f.name)]
    
    if len(spacy_files) != 1 or len(non_spacy_files) < 1:
        raise ValueError(f"Directory must contain exactly one spaCy benchmark file and at least one non-spaCy benchmark file")
        
    return str(non_spacy_files[0]), str(spacy_files[0])

def main():
    if len(sys.argv) != 2:
        print("Usage: python compare_benchmarks.py <benchmark_directory>")
        sys.exit(1)

    directory = sys.argv[1]
    try:
        base_path, spacy_path = find_benchmark_files(directory)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    base_data = load_json(base_path)
    spacy_data = load_json(spacy_path)
    
    comparisons = compare_metrics(base_data, spacy_data)
    edge_analysis = analyze_edges(extract_edges(base_data), extract_edges(spacy_data))
    
    # Generate output filenames in the same directory
    output_dir = Path(directory)
    base_output = output_dir / f"comparison_{Path(base_path).stem}_vs_{Path(spacy_path).stem}"
    
    write_csv_report(comparisons, f"{base_output}.csv")
    write_json_report(comparisons, f"{base_output}.json")
    write_markdown_report(comparisons, edge_analysis, base_path, spacy_path, f"{base_output}.md")
    
    print(f"Comparison reports generated:")
    print(f"- CSV: {base_output}.csv")
    print(f"- JSON: {base_output}.json") 
    print(f"- Markdown: {base_output}.md")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""Script to run multiple pipelines on a dataset."""
import argparse
from pathlib import Path
import re
import sys
import yaml

from .. import pipelines
from ..dataset import Dataset

def get_pipeline_map():
    """Dynamically create mapping of pipeline names to extractor classes.
    
    Converts CamelCase class names to snake_case pipeline names by:
    1. Removing 'Extractor' suffix
    2. Converting remaining CamelCase to snake_case
    """
    def camel_to_snake(name: str) -> str:
        # Remove 'Extractor' suffix
        name = re.sub('Extractor$', '', name)
        # Convert CamelCase to snake_case
        pattern = re.compile(r'(?<!^)(?=[A-Z])')
        return pattern.sub('_', name).lower()
    
    # Create mapping from snake_case name to class
    return {
        camel_to_snake(cls.__name__): cls
        for cls_name in pipelines.__all__
        if (cls := getattr(pipelines, cls_name))
    }

def load_yaml_config(config_path: Path, available_pipelines: set) -> list:
    """Load pipeline configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        available_pipelines: Set of valid pipeline names
        
    Returns:
        List of tuples (pipeline_name, pipeline_args) to execute
        
    Example YAML format:
    pipelines:
      - name: word_count
        args:
          min_word_length: 3
      - name: task
        args:
          model_name: gpt-4
    """
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        if not isinstance(config, dict) or "pipelines" not in config:
            raise ValueError("YAML file must contain a 'pipelines' list")
            
        pipelines_config = config["pipelines"]
        if not isinstance(pipelines_config, list):
            raise ValueError("'pipelines' must be a list")
            
        # Validate and extract pipeline configurations
        pipeline_configs = []
        for p in pipelines_config:
            if isinstance(p, str):
                # Simple pipeline name without args
                name = p
                args = {}
            elif isinstance(p, dict) and "name" in p:
                # Pipeline with args
                name = p["name"]
                args = p.get("args", {})
            else:
                raise ValueError(f"Invalid pipeline configuration: {p}")
                
            if name not in available_pipelines:
                raise ValueError(f"Invalid pipeline name: {name}. Available pipelines: {sorted(available_pipelines)}")
                
            pipeline_configs.append((name, args))
            
        return pipeline_configs
        
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")

def run_pipelines(dataset_path: Path, output_path: Path, pipeline_configs: list, pipeline_map: dict, num_workers: int = 1) -> None:
    """Run specified pipelines on dataset.
    
    Args:
        dataset_path: Path to ns-pond Dataset
        output_path: Path to save pipeline outputs
        pipeline_configs: List of tuples (pipeline_name, pipeline_args) to execute
        pipeline_map: Mapping of pipeline names to pipeline classes
        num_workers: Number of worker threads for parallel processing (default: 1)
    """
    # Initialize dataset
    try:
        dataset = Dataset(dataset_path)
    except Exception as e:
        print(f"Error loading dataset from {dataset_path}: {e}", file=sys.stderr)
        sys.exit(1)
        

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Run each pipeline
    for pipeline_name, pipeline_args in pipeline_configs:
        try:
            print(f"Running {pipeline_name} pipeline...")
            
            # Initialize pipeline with arguments
            pipeline_class = pipeline_map[pipeline_name]
            pipeline = pipeline_class(**pipeline_args)
            
            # Run pipeline
            pipeline_output_dir = output_path / pipeline_name
            pipeline.transform_dataset(dataset, pipeline_output_dir, num_workers=num_workers)
            
            print(f"Completed {pipeline_name} pipeline")
            
        except Exception as e:
            print(f"Error running {pipeline_name} pipeline: {e}", file=sys.stderr)
            sys.exit(1)

def main():
    # Get available pipelines
    pipeline_map = get_pipeline_map()
    available_pipelines = set(pipeline_map.keys())

    parser = argparse.ArgumentParser(description="Run pipelines on ns-pond dataset")
    parser.add_argument("dataset_path", type=Path, help="Path to ns-pond dataset")
    parser.add_argument("output_path", type=Path, help="Path to save pipeline outputs")
    parser.add_argument("--num-workers", type=int, default=1,
                      help="Number of worker threads for parallel processing (default: 1)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pipelines",
        nargs="+",
        choices=list(available_pipelines),
        help="List of pipelines to run in order"
    )
    group.add_argument(
        "--config",
        type=Path,
        help="YAML configuration file specifying pipelines to run"
    )

    args = parser.parse_args()

    # Get pipeline configurations
    if args.config:
        try:
            pipeline_configs = load_yaml_config(args.config, available_pipelines)
        except ValueError as e:
            print(f"Error loading config file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # For command line, use pipelines with default arguments
        pipeline_configs = [(name, {}) for name in args.pipelines]

    # Run pipelines
    run_pipelines(args.dataset_path, args.output_path, pipeline_configs, pipeline_map,
                 num_workers=args.num_workers)

if __name__ == "__main__":
    main()
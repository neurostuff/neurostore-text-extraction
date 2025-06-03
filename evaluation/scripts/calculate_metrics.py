#!/usr/bin/env python3
"""
Script to calculate evaluation metrics comparing extracted data against ground truth.
Implements field mapping, normalization, and detailed metric calculations.
"""
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Union
import re

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Normalization Functions
def normalize_text(text: str) -> str:
    """Normalize text fields by removing extra whitespace and converting to lowercase."""
    if pd.isna(text):
        return ""
    return re.sub(r'\s+', ' ', str(text).strip().lower())

def normalize_numeric(value: Union[str, float, int]) -> float:
    """Convert numeric values to float, handling various formats."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        # Remove non-numeric characters except decimal points
        cleaned = re.sub(r'[^\d.-]', '', value)
        return float(cleaned) if cleaned else np.nan
    return float(value)

def normalize_boolean(value: Union[str, bool]) -> bool:
    """Normalize boolean values, handling various formats."""
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', 'yes', '1', 't', 'y')
    return bool(value)

def normalize_age_range(age: Union[str, float]) -> Tuple[float, float]:
    """Convert age values to standardized range format."""
    if pd.isna(age):
        return (np.nan, np.nan)
    
    if isinstance(age, (int, float)):
        return (float(age), float(age))
    
    # Handle ranges like "18-25" or "18 to 25"
    range_match = re.search(r'(\d+)(?:\s*[-to]+\s*)(\d+)', str(age))
    if range_match:
        return (float(range_match.group(1)), float(range_match.group(2)))
    
    # Handle single numbers in string format
    number_match = re.search(r'(\d+)', str(age))
    if number_match:
        value = float(number_match.group(1))
        return (value, value)
    
    return (np.nan, np.nan)

def calculate_field_metrics(ground_truth: pd.Series, extracted: pd.Series, 
                          normalization_func=None, tolerance: float = 0.0) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score for a single field.
    
    Args:
        ground_truth: Series containing ground truth values
        extracted: Series containing extracted values
        normalization_func: Optional function to normalize values before comparison
        tolerance: Numeric tolerance for float comparisons
    
    Returns:
        Dictionary containing precision, recall, and F1 score
    """
    if normalization_func:
        ground_truth = ground_truth.apply(normalization_func)
        extracted = extracted.apply(normalization_func)

    total_ground_truth = len(ground_truth.dropna())
    total_extracted = len(extracted.dropna())
    
    # Handle numeric comparisons with tolerance
    if np.issubdtype(ground_truth.dtype, np.number):
        matches = sum(abs(g - e) <= tolerance 
                     for g, e in zip(ground_truth, extracted) 
                     if not (pd.isna(g) or pd.isna(e)))
    else:
        matches = sum(g == e 
                     for g, e in zip(ground_truth, extracted) 
                     if not (pd.isna(g) or pd.isna(e)))

    precision = matches / total_extracted if total_extracted > 0 else 0.0
    recall = matches / total_ground_truth if total_ground_truth > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'matches': matches,
        'total_ground_truth': total_ground_truth,
        'total_extracted': total_extracted
    }

def calculate_metrics(ground_truth_df: pd.DataFrame, extracted_df: pd.DataFrame, 
                     field_configs: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Calculate metrics for all fields based on their configurations.
    
    Args:
        ground_truth_df: DataFrame containing ground truth data
        extracted_df: DataFrame containing extracted data
        field_configs: Dictionary mapping field names to their configurations
                      (normalization function, comparison tolerance, etc.)
    
    Returns:
        Dictionary containing metrics for each field and overall aggregate metrics
    """
    results = {}
    aggregate_metrics = {'precision': [], 'recall': [], 'f1': []}

    for field, config in field_configs.items():
        if field not in ground_truth_df.columns or field not in extracted_df.columns:
            continue

        field_metrics = calculate_field_metrics(
            ground_truth_df[field],
            extracted_df[field],
            normalization_func=config.get('normalize'),
            tolerance=config.get('tolerance', 0.0)
        )
        
        results[field] = field_metrics
        
        # Add to aggregate calculations
        aggregate_metrics['precision'].append(field_metrics['precision'])
        aggregate_metrics['recall'].append(field_metrics['recall'])
        aggregate_metrics['f1'].append(field_metrics['f1'])

    # Calculate aggregate metrics
    results['aggregate'] = {
        'precision': np.mean(aggregate_metrics['precision']),
        'recall': np.mean(aggregate_metrics['recall']),
        'f1': np.mean(aggregate_metrics['f1'])
    }

    return results

def format_results(results: Dict[str, Dict]) -> str:
    """Format results into a readable string with tables."""
    output = []
    
    # Header
    output.append("Evaluation Metrics")
    output.append("=================")
    output.append("")

    # Field-specific metrics
    output.append("Per-Field Metrics:")
    output.append("-----------------")
    header = f"{'Field':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}"
    output.append(header)
    output.append("-" * len(header))
    
    for field, metrics in results.items():
        if field == 'aggregate':
            continue
        line = f"{field:<20} {metrics['precision']:>10.2%} {metrics['precision']:>10.2%} {metrics['f1']:>10.2%}"
        output.append(line)
    
    # Aggregate metrics
    output.append("\nAggregate Metrics:")
    output.append("-----------------")
    agg = results['aggregate']
    output.append(f"Overall Precision: {agg['precision']:.2%}")
    output.append(f"Overall Recall: {agg['recall']:.2%}")
    output.append(f"Overall F1 Score: {agg['f1']:.2%}")
    
    return "\n".join(output)

def main():
    """Main execution function."""
    # Define field configurations for each dataset type
    pd_field_configs = {
        'age': {'normalize': normalize_age_range, 'tolerance': 0.5},
        'diagnosis': {'normalize': normalize_text},
        'group': {'normalize': normalize_text},
        'n': {'normalize': normalize_numeric, 'tolerance': 0},
        'gender': {'normalize': normalize_text},
        'handedness': {'normalize': normalize_text},
    }

    nv_field_configs = {
        'diagnosis': {'normalize': normalize_text},
        'description': {'normalize': normalize_text},
        'has_disease': {'normalize': normalize_boolean},
        'sample_size': {'normalize': normalize_numeric, 'tolerance': 0},
    }

    # Load ground truth and extracted data
    ground_truth_path = os.path.join(PROJECT_ROOT, "evaluation", "data")
    
    # Process participant demographics
    pd_ground_truth = pd.read_csv(os.path.join(ground_truth_path, "participant_demographics_ground_truth.csv"))
    pd_extracted = pd.read_csv(os.path.join(ground_truth_path, "participant_demographics_extracted.csv"))
    pd_results = calculate_metrics(pd_ground_truth, pd_extracted, pd_field_configs)
    
    # Process NV task data
    nv_ground_truth = pd.read_csv(os.path.join(ground_truth_path, "nv_task_ground_truth.csv"))
    nv_extracted = pd.read_csv(os.path.join(ground_truth_path, "nv_task_extracted.csv"))
    nv_results = calculate_metrics(nv_ground_truth, nv_extracted, nv_field_configs)
    
    # Print formatted results
    print("\nParticipant Demographics Results:")
    print("================================")
    print(format_results(pd_results))
    
    print("\nNV Task Results:")
    print("===============")
    print(format_results(nv_results))

if __name__ == "__main__":
    main()

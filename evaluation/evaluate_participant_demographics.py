#!/usr/bin/env python3
"""Script to evaluate participant demographics extraction against ground truth."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any


def load_ground_truth(csv_path: str) -> pd.DataFrame:
    """Load ground truth data from CSV.
    
    Args:
        csv_path: Path to ground truth CSV file
        
    Returns:
        DataFrame containing ground truth data
    """
    df = pd.read_csv(csv_path)
    df['pmcid'] = df['pmcid'].astype(str)
    return df


def compare_numeric(extracted: float, truth: float, tolerance: float = 0.1) -> bool:
    """Compare numeric values within tolerance.
    
    Args:
        extracted: Extracted numeric value
        truth: Ground truth value
        tolerance: Relative tolerance for comparison
        
    Returns:
        True if values match within tolerance
    """
    if pd.isna(extracted) and pd.isna(truth):
        return True
    if pd.isna(extracted) or pd.isna(truth):
        return False
    return abs(extracted - truth) <= (tolerance * truth)


def compare_counts(extracted: Dict[str, Any], truth: Dict[str, Any]) -> Dict[str, bool]:
    """Compare extracted counts against ground truth.
    
    Args:
        extracted: Dictionary containing extracted counts
        truth: Dictionary containing ground truth counts
        
    Returns:
        Dictionary mapping count fields to match results
    """
    results = {}
    count_fields = ['count', 'male_count', 'female_count']
    
    for field in count_fields:
        extracted_val = extracted.get(field)
        truth_val = truth.get(field)
        
        if extracted_val is None and truth_val is None:
            results[field] = True
        elif extracted_val is None or truth_val is None:
            results[field] = False
        else:
            results[field] = compare_numeric(float(extracted_val), float(truth_val))
            
    return results


def compare_age_stats(extracted: Dict[str, Any], truth: Dict[str, Any]) -> Dict[str, bool]:
    """Compare extracted age statistics against ground truth.
    
    Args:
        extracted: Dictionary containing extracted age statistics
        truth: Dictionary containing ground truth age statistics
        
    Returns:
        Dictionary mapping age statistic fields to match results
    """
    results = {}
    age_fields = ['age_mean', 'age_minimum', 'age_maximum', 'age_median']
    
    for field in age_fields:
        extracted_val = extracted.get(field)
        truth_val = truth.get(field)
        
        if extracted_val is None and truth_val is None:
            results[field] = True
        elif extracted_val is None or truth_val is None:
            results[field] = False
        else:
            results[field] = compare_numeric(float(extracted_val), float(truth_val))
            
    return results


def evaluate_extraction(
    extractor_output: Dict[str, Any], 
    ground_truth: pd.DataFrame
) -> Dict[str, float]:
    """Evaluate extractor output against ground truth.
    
    Args:
        extractor_output: Dictionary containing extracted data for a study
        ground_truth: DataFrame with ground truth data
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Initialize counters
    field_matches = {
        'count': 0,
        'male_count': 0,
        'female_count': 0,
        'age_mean': 0,
        'age_minimum': 0, 
        'age_maximum': 0,
        'age_median': 0,
        'group_names': 0
    }
    
    field_totals = {
        'count': 0,
        'male_count': 0,
        'female_count': 0,
        'age_mean': 0,
        'age_minimum': 0,
        'age_maximum': 0,
        'age_median': 0,
        'group_names': 0
    }

    # Get ground truth rows for this study
    pmcid = extractor_output['pmcid']
    study_truth = ground_truth[ground_truth['pmcid'] == pmcid]
    
    # Compare each extracted group against ground truth
    for group in extractor_output.get('groups', []):
        # Find matching ground truth group
        group_name = group.get('group_name')
        if not group_name:
            continue
            
        truth_group = study_truth[study_truth['group_name'] == group_name]
        if len(truth_group) == 0:
            continue
        
        # Match on first matching group
        truth_group = truth_group.iloc[0]
        
        # Compare counts
        count_results = compare_counts(group, truth_group)
        for field, matches in count_results.items():
            if truth_group[field] is not None:
                field_totals[field] += 1
                if matches:
                    field_matches[field] += 1
                    
        # Compare age statistics
        age_results = compare_age_stats(group, truth_group) 
        for field, matches in age_results.items():
            if truth_group[field] is not None:
                field_totals[field] += 1
                if matches:
                    field_matches[field] += 1
                    
        # Group name was matched
        field_matches['group_names'] += 1
        field_totals['group_names'] += 1
        
    # Calculate accuracy for each field
    accuracies = {}
    for field in field_matches:
        if field_totals[field] > 0:
            accuracies[field] = field_matches[field] / field_totals[field]
        else:
            accuracies[field] = None
            
    return accuracies


def evaluate_extractions(
    extraction_dir: str,
    ground_truth_path: str,
    output_path: str = None
) -> Dict[str, float]:
    """Evaluate all extractions in directory against ground truth.
    
    Args:
        extraction_dir: Directory containing extraction JSON files
        ground_truth_path: Path to ground truth CSV
        output_path: Optional path to write results JSON
        
    Returns:
        Dictionary containing overall evaluation metrics
    """
    # Load ground truth
    ground_truth = load_ground_truth(ground_truth_path)
    
    # Track metrics across all studies
    total_matches = {
        'count': 0,
        'male_count': 0,
        'female_count': 0,
        'age_mean': 0,
        'age_minimum': 0,
        'age_maximum': 0,
        'age_median': 0,
        'group_names': 0
    }
    
    total_fields = {
        'count': 0,
        'male_count': 0,
        'female_count': 0, 
        'age_mean': 0,
        'age_minimum': 0,
        'age_maximum': 0,
        'age_median': 0,
        'group_names': 0
    }
    
    # Process each extraction file
    results = {'studies': {}}
    extraction_dir = Path(extraction_dir)
    for json_path in extraction_dir.glob('*.json'):
        with open(json_path) as f:
            extraction = json.load(f)
            
        # Evaluate this extraction
        metrics = evaluate_extraction(extraction, ground_truth)
        results['studies'][json_path.stem] = metrics
        
        # Add to totals
        for field in metrics:
            if metrics[field] is not None:
                total_matches[field] += metrics[field]
                total_fields[field] += 1
                
    # Calculate overall accuracies
    overall = {}
    for field in total_matches:
        if total_fields[field] > 0:
            overall[field] = total_matches[field] / total_fields[field]
        else:
            overall[field] = None
    
    results['overall'] = overall
    
    # Write results if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Evaluate participant demographics extraction against ground truth'
    )
    parser.add_argument(
        'extraction_dir',
        help='Directory containing extraction JSON files'
    )
    parser.add_argument(
        'ground_truth',
        help='Path to ground truth CSV file'
    )
    parser.add_argument(
        '--output',
        help='Optional path to write results JSON'
    )
    
    args = parser.parse_args()
    
    results = evaluate_extractions(
        args.extraction_dir,
        args.ground_truth,
        args.output
    )
    
    # Print overall results
    print("\nOverall Results:")
    for field, accuracy in results['overall'].items():
        if accuracy is not None:
            print(f"{field}: {accuracy:.3f}")
        else:
            print(f"{field}: N/A")

#!/usr/bin/env python3
"""
Main evaluation script that orchestrates the evaluation process for text extractors.
Combines ground truth analysis, metric calculations, and generates comprehensive reports.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from evaluation.scripts.analyze_ground_truth import (
    load_participant_demographics_ground_truth,
    load_nv_task_ground_truth
)
from evaluation.scripts.calculate_metrics import (
    calculate_metrics,
    PROJECT_ROOT,
    format_results,
    normalize_text,
    normalize_numeric,
    normalize_boolean,
    normalize_age_range
)

# Field configurations
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

def load_extracted_data(dataset_type: str) -> pd.DataFrame:
    """Load extracted data for the specified dataset type."""
    extracted_path = os.path.join(
        PROJECT_ROOT, 
        "evaluation", 
        "data", 
        f"{dataset_type}_extracted.csv"
    )
    return pd.read_csv(extracted_path)

def analyze_errors(ground_truth: pd.DataFrame, extracted: pd.DataFrame, 
                  field_configs: Dict) -> Dict[str, Dict]:
    """
    Analyze extraction errors to identify patterns and potential improvements.
    
    Args:
        ground_truth: Ground truth DataFrame
        extracted: Extracted data DataFrame
        field_configs: Field configuration dictionary
        
    Returns:
        Dictionary containing error analysis results
    """
    error_analysis = {}
    
    for field in field_configs.keys():
        if field not in ground_truth.columns or field not in extracted.columns:
            continue
            
        # Get normalization function if specified
        normalize = field_configs[field].get('normalize')
        
        # Compare values and collect errors
        gt_series = ground_truth[field].apply(normalize) if normalize else ground_truth[field]
        ex_series = extracted[field].apply(normalize) if normalize else extracted[field]
        
        # Find mismatches
        mismatches = []
        for gt, ex in zip(gt_series, ex_series):
            if pd.notna(gt) and (pd.isna(ex) or gt != ex):
                mismatches.append((gt, ex if pd.notna(ex) else "NOT_EXTRACTED"))
        
        # Analyze error patterns
        error_types = {}
        for gt, ex in mismatches:
            error_type = "Missing" if ex == "NOT_EXTRACTED" else "Incorrect"
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append((gt, ex))
        
        error_analysis[field] = {
            'total_errors': len(mismatches),
            'error_types': error_types,
            'error_rate': len(mismatches) / len(ground_truth) if len(ground_truth) > 0 else 0
        }
    
    return error_analysis

def generate_error_report(error_analysis: Dict[str, Dict]) -> str:
    """Generate a formatted error analysis report."""
    lines = ["Error Analysis Report", "===================", ""]
    
    for field, analysis in error_analysis.items():
        lines.append(f"Field: {field}")
        lines.append("-" * (len(field) + 7))
        lines.append(f"Total Errors: {analysis['total_errors']}")
        lines.append(f"Error Rate: {analysis['error_rate']:.2%}")
        
        lines.append("\nError Types:")
        for error_type, examples in analysis['error_types'].items():
            lines.append(f"\n{error_type} Values ({len(examples)} instances):")
            # Show up to 3 examples
            for gt, ex in examples[:3]:
                lines.append(f"  Ground Truth: {gt}")
                lines.append(f"  Extracted: {ex}")
                lines.append("")
        
        lines.append("-" * 50 + "\n")
    
    return "\n".join(lines)

def run_evaluation(dataset_type: str) -> Tuple[Dict, Dict]:
    """
    Run evaluation for a specific dataset type.
    
    Args:
        dataset_type: Type of dataset ('participant_demographics' or 'nv_task')
        
    Returns:
        Tuple of (metrics_results, error_analysis)
    """
    # Load ground truth data
    if dataset_type == 'participant_demographics':
        ground_truth_cols = load_participant_demographics_ground_truth()
        field_configs = pd_field_configs
    else:  # nv_task
        ground_truth_cols = load_nv_task_ground_truth()
        field_configs = nv_field_configs
    
    # Load data
    ground_truth_path = os.path.join(PROJECT_ROOT, "evaluation", "data", f"{dataset_type}_ground_truth.csv")
    ground_truth = pd.read_csv(ground_truth_path)
    extracted = load_extracted_data(dataset_type)
    
    # Calculate metrics
    metrics_results = calculate_metrics(ground_truth, extracted, field_configs)
    
    # Analyze errors
    error_analysis = analyze_errors(ground_truth, extracted, field_configs)
    
    return metrics_results, error_analysis

def identify_improvements(error_analysis: Dict[str, Dict]) -> List[str]:
    """Identify potential areas for improvement based on error analysis."""
    improvements = []
    
    # Sort fields by error rate
    sorted_fields = sorted(
        error_analysis.items(),
        key=lambda x: x[1]['error_rate'],
        reverse=True
    )
    
    for field, analysis in sorted_fields:
        if analysis['error_rate'] > 0.3:  # High error rate threshold
            improvements.append(f"High priority: Improve {field} extraction (Error rate: {analysis['error_rate']:.2%})")
            
            # Analyze error patterns
            missing = len(analysis['error_types'].get('Missing', []))
            incorrect = len(analysis['error_types'].get('Incorrect', []))
            
            if missing > incorrect:
                improvements.append(f"  - Focus on improving {field} detection/coverage")
            else:
                improvements.append(f"  - Focus on improving {field} accuracy/normalization")
    
    return improvements

def main():
    """Main execution function."""
    # Evaluate participant demographics
    print("Evaluating Participant Demographics Extraction...")
    pd_metrics, pd_errors = run_evaluation('participant_demographics')
    print("\nParticipant Demographics Results:")
    print("================================")
    print(format_results(pd_metrics))
    print("\nParticipant Demographics Error Analysis:")
    print(generate_error_report(pd_errors))
    
    # Evaluate NV task
    print("\nEvaluating NV Task Extraction...")
    nv_metrics, nv_errors = run_evaluation('nv_task')
    print("\nNV Task Results:")
    print("===============")
    print(format_results(nv_metrics))
    print("\nNV Task Error Analysis:")
    print(generate_error_report(nv_errors))
    
    # Generate improvement recommendations
    print("\nRecommended Improvements:")
    print("=======================")
    pd_improvements = identify_improvements(pd_errors)
    nv_improvements = identify_improvements(nv_errors)
    
    if pd_improvements:
        print("\nParticipant Demographics Improvements:")
        for imp in pd_improvements:
            print(imp)
            
    if nv_improvements:
        print("\nNV Task Improvements:")
        for imp in nv_improvements:
            print(imp)

if __name__ == "__main__":
    main()

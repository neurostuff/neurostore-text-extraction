#!/usr/bin/env python3
"""
Enhanced evaluation script for text extractors with improved validation and metrics.
Provides comprehensive evaluation of extraction accuracy and detailed error analysis.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional

import pandas as pd
from ns_extract.cli.run import run_pipelines, get_pipeline_map
from evaluation.scripts.calculate_metrics import (
    calculate_metrics,
    PROJECT_ROOT,
    format_results,
    normalize_text,
    normalize_numeric,
    normalize_boolean,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class FieldConfig:
    """Configuration for field validation and comparison."""

    normalize_func: Optional[Callable] = None
    required: bool = False
    tolerance: float = 0.0
    weight: float = 1.0


# Field configurations with validation rules
PARTICIPANT_DEMOGRAPHICS_CONFIG = {
    "age_mean": FieldConfig(normalize_numeric, tolerance=0.5, weight=1.0),
    "age_median": FieldConfig(normalize_numeric, tolerance=0.5, weight=1.0),
    "age_minimum": FieldConfig(normalize_numeric, tolerance=0.5, weight=1.0),
    "age_maximum": FieldConfig(normalize_numeric, tolerance=0.5, weight=1.0),
    "diagnosis": FieldConfig(normalize_text, required=True, weight=1.0),
    "group_name": FieldConfig(normalize_text, required=True, weight=1.0),
    "subgroup_name": FieldConfig(normalize_text, required=True, weight=1.0),
    "count": FieldConfig(normalize_numeric, required=True, weight=1.0),
    "female_count": FieldConfig(normalize_numeric, weight=1.0),
    "male_count": FieldConfig(normalize_numeric, weight=1.0),
}

TASK_CONFIG = {
    "HasRestingState": FieldConfig(normalize_boolean, required=True, weight=1.0),
    "Modality": FieldConfig(normalize_text, required=True, weight=1.0),
    "TaskName": FieldConfig(normalize_text, required=True, weight=1.0),
    "TaskDescription": FieldConfig(normalize_text, weight=1.0),
    "Condition": FieldConfig(normalize_text, required=True, weight=1.0),
}


def run_extraction_pipelines():
    """Run extraction pipelines on the test dataset."""
    pipeline_map = get_pipeline_map()

    # Configure pipelines to run
    pipeline_configs = [
        (
            "participant_demographics",
            {
                "extraction_model": "gpt-4o-mini-2024-07-18",
                "env_variable": "OPENAI_API_KEY",
            },
        ),
        (
            "task",
            {
                "extraction_model": "gpt-4o-mini-2024-07-18",
                "env_variable": "OPENAI_API_KEY",
            },
        ),
        (
            "contrasts",
            {
                "extraction_model": "gpt-4o-mini-2024-07-18",
                "env_variable": "OPENAI_API_KEY",
            },
        ),
    ]

    # Define paths
    dataset_path = Path(PROJECT_ROOT) / "evaluation" / "data" / "ns_pond_inputs"
    output_path = Path(PROJECT_ROOT) / "evaluation" / "results"

    # Run pipelines
    print("Running extraction pipelines...")
    run_pipelines(
        dataset_path=dataset_path,
        output_path=output_path,
        pipeline_configs=pipeline_configs,
        pipeline_map=pipeline_map,
        num_workers=2,
    )
    print("Extraction complete")
    return output_path


def load_extracted_results(output_path: Path, dataset_type: str) -> pd.DataFrame:
    """Load and format pipeline results with enhanced error handling.

    Args:
        output_path: Base path containing pipeline outputs
        dataset_type: Dataset type to load ('participant_demographics' or 'task')

    Args:
        output_path: Base path containing pipeline outputs
        dataset_type: Type of extractor results to load ('participant_demographics' or 'task')

    Returns:
        DataFrame containing combined results from all processed studies
    """
    # Find the version directory (most recent if multiple)
    extractor_dir = next(
        iter(sorted((output_path / dataset_type).glob("*/*/*"), reverse=True))
    )

    all_records = []

    # Process each study directory
    for study_dir in extractor_dir.iterdir():
        if not study_dir.is_dir():
            continue

        # Load study identifiers
        try:
            with open(study_dir / "info.json") as f:
                info = json.load(f)
                identifiers = info["identifiers"]
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            print(f"Error loading info.json for {study_dir}: {e}")
            continue

        # Load extracted results
        try:
            with open(study_dir / "results.json") as f:
                results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading results.json for {study_dir}: {e}")
            continue

        if dataset_type == "participant_demographics":
            # Flatten groups array into individual records
            for group in results.get("groups", []):
                # Convert field names to use underscores
                # Skip if no PMCID
                if not identifiers.get("pmcid"):
                    continue

                record = {
                    "pmcid": str(
                        identifiers.get("pmcid").lstrip("PMC")
                    ),  # Ensure pmcid is a string
                    "group_name": group.get("group_name", ""),
                    "count": group.get("count"),
                    "age_mean": group.get("age_mean"),
                    "age_median": group.get("age_median"),
                    "age_minimum": group.get("age_minimum"),
                    "age_maximum": group.get("age_maximum"),
                    "female_count": group.get("female_count"),
                    "male_count": group.get("male_count"),
                    "subgroup_name": group.get("subgroup_name", "_"),
                    "diagnosis": group.get("diagnosis", ""),
                }
                all_records.append(record)

        elif dataset_type == "contrasts":
                    # For task extractor, extract contrast information
                    if "MRIContrasts" in results:
                        # Combine all tasks into a single record per study
                        contrasts = results.get("MRIContrasts", [])

                        if not identifiers.get("pmcid"):
                            continue

                    record = {
                        "pmcid": str(
                            identifiers.get("pmcid").lstrip("PMC")
                        ),
                        "comparison": contrasts.get("comparison", ""),
                        "control_group": group.get("control_group", ""),
                        "group": group.get("group", ""),
                        "contrast_statistc": group.get("contrast_statistc", ""),
                        "atlas": group.get("atlas", ""),
                        "atlas_n_regions": group.get("atlas_n_regions"),
                        "roi": group.get("roi", ""),
                        "coord_system": group.get("coord_system", ""),
                        "x": group.get("x"),
                        "y": group.get("y"),
                        "z": group.get("z"),
                        "significance": group.get("significance"),
                        "significance_level": group.get("significance_level", ""),

                    }
                        
                    all_records.append(record)

        elif dataset_type == "task":
            # For task extractor, extract task information
            if "fMRITasks" in results:
                # Combine all tasks into a single record per study
                tasks = results.get("fMRITasks", [])
                if tasks:
                    # Safely collect task conditions
                    all_conditions = []
                    for task in tasks:
                        conditions = task.get("Conditions")
                        if isinstance(conditions, list):
                            all_conditions.extend(conditions)

                    record = {
                        "pmcid": identifiers.get("pmcid").lstrip("PMC"),
                        "HasRestingState": any(
                            task.get("RestingState", False) for task in tasks
                        ),
                        "Modality": ["fMRI-BOLD"],
                        "TaskName": [t["TaskName"] for t in tasks if t.get("TaskName")],
                        "TaskDescription": [
                            t["TaskDescription"]
                            for t in tasks
                            if t.get("TaskDescription")
                        ],
                        "Condition": all_conditions,
                    }
                    all_records.append(record)

    if not all_records:
        raise ValueError(f"No valid results found for {dataset_type}")

    return pd.DataFrame(all_records)


def analyze_errors(
    ground_truth: pd.DataFrame,
    extracted: pd.DataFrame,
    field_configs: Dict[str, FieldConfig],
) -> Dict[str, Dict]:
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

        # Get field configuration
        config = field_configs[field]

        # Compare values with normalization and tolerance
        gt_series = (
            ground_truth[field].apply(config.normalize_func)
            if config.normalize_func
            else ground_truth[field]
        )
        ex_series = (
            extracted[field].apply(config.normalize_func)
            if config.normalize_func
            else extracted[field]
        )

        # Find mismatches
        mismatches = []
        for gt, ex in zip(gt_series, ex_series):
            if pd.notna(gt):
                # Handle case where ex is a list
                if isinstance(ex, list):
                    if not ex:  # Empty list
                        mismatches.append((gt, "NOT_EXTRACTED"))
                elif pd.isna(ex):
                    mismatches.append((gt, "NOT_EXTRACTED"))
                else:
                    # Compare with tolerance for numeric values
                    if (
                        isinstance(gt, (int, float))
                        and isinstance(ex, (int, float))
                        and config.tolerance > 0
                    ):
                        if abs(gt - ex) > config.tolerance:
                            mismatches.append((gt, ex))
                    elif gt != ex:
                        mismatches.append((gt, ex))

        # Analyze error patterns
        error_types = {}
        for gt, ex in mismatches:
            error_type = "Missing" if ex == "NOT_EXTRACTED" else "Incorrect"
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append((gt, ex))

        error_analysis[field] = {
            "total_errors": len(mismatches),
            "error_types": error_types,
            "error_rate": (
                len(mismatches) / len(ground_truth) if len(ground_truth) > 0 else 0
            ),
            "weight": config.weight,
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
        for error_type, examples in analysis["error_types"].items():
            lines.append(f"\n{error_type} Values ({len(examples)} instances):")
            # Show up to 3 examples
            for gt, ex in examples[:3]:
                lines.append(f"  Ground Truth: {gt}")
                lines.append(f"  Extracted: {ex}")
                lines.append("")

        lines.append("-" * 50 + "\n")

    return "\n".join(lines)


def run_evaluation(
    dataset_type: str, extracted_data: pd.DataFrame
) -> Tuple[Dict, Dict]:
    """Run comprehensive evaluation with enhanced validation.

    Performs detailed comparison between extracted data and ground truth,
    calculating metrics and analyzing errors.

    Args:
        dataset_type: Type of dataset ('participant_demographics' or 'task')
        extracted_data: DataFrame containing the extracted results to evaluate

    Returns:
        Tuple of (metrics_results, error_analysis)

    Raises:
        ValueError: If required columns are missing or data validation fails
    """
    # Select appropriate field configs
    field_configs = (
        PARTICIPANT_DEMOGRAPHICS_CONFIG
        if dataset_type == "participant_demographics"
        else TASK_CONFIG
    )

    # Ensure extracted data has pmcid column
    if "pmcid" not in extracted_data.columns:
        raise ValueError("Extracted data must have 'pmcid' column")

    # Strip "PMC" prefix from extracted pmcids
    extracted_data["pmcid"] = extracted_data["pmcid"].str.lstrip("PMC")

    # Load and prepare ground truth data
    ground_truth_path = (
        Path(PROJECT_ROOT) / "evaluation" / "data" / f"{dataset_type}_ground_truth.csv"
    )
    ground_truth = pd.read_csv(ground_truth_path, dtype={"pmcid": str})

    if "pmcid" not in ground_truth.columns:
        raise ValueError("Ground truth must have 'pmcid' column")

    # Filter to include only matching pmcids and sort
    common_pmcids = set(ground_truth["pmcid"]).intersection(
        set(extracted_data["pmcid"])
    )
    ground_truth = ground_truth[ground_truth["pmcid"].isin(common_pmcids)]
    extracted_data = extracted_data[extracted_data["pmcid"].isin(common_pmcids)]

    # Sort both dataframes by pmcid
    ground_truth = ground_truth.sort_values("pmcid").reset_index(drop=True)
    extracted_data = extracted_data.sort_values("pmcid").reset_index(drop=True)

    print(
        f"Matched {len(common_pmcids)} PMCIDs between ground truth and extracted data"
    )

    # Normalize field names if needed
    if dataset_type == "participant_demographics":
        rename_map = {
            "subgroup name": "subgroup_name",
            "female count": "female_count",
            "male count": "male_count",
            "age mean": "age_mean",
            "age median": "age_median",
            "age minimum": "age_minimum",
            "age maximum": "age_maximum",
        }
        ground_truth = ground_truth.rename(columns=rename_map)

    # Validate required columns and field requirements
    required_cols = {"pmcid"} | {
        field for field, config in field_configs.items() if config.required
    }

    missing_gt = required_cols - set(ground_truth.columns)
    if missing_gt:
        raise ValueError(f"Missing required columns in ground truth data: {missing_gt}")

    missing_ext = required_cols - set(extracted_data.columns)
    if missing_ext:
        raise ValueError(f"Missing required columns in extracted data: {missing_ext}")

    # Group comparison for participant demographics
    if dataset_type == "participant_demographics":
        # Compare groups by PMCID
        matched_results = []
        for pmcid in extracted_data["pmcid"].unique():
            gt_rows = ground_truth[ground_truth["pmcid"] == pmcid]
            ext_rows = extracted_data[extracted_data["pmcid"] == pmcid]

            # For each ground truth row, find best matching extracted row
            for _, gt_row in gt_rows.iterrows():
                best_match = None
                best_score = -1

                for _, ext_row in ext_rows.iterrows():
                    match_score = 0.0
                    total_weight = 0.0

                    for field, config in field_configs.items():
                        gt_val = gt_row.get(field)
                        ext_val = ext_row.get(field)

                        if pd.notna(gt_val) and pd.notna(ext_val):
                            # Apply normalization if configured
                            if config.normalize_func:
                                gt_val = config.normalize_func(gt_val)
                                ext_val = config.normalize_func(ext_val)

                            # Compare with tolerance and weight
                            matched = False
                            if isinstance(gt_val, (int, float)) and isinstance(
                                ext_val, (int, float)
                            ):
                                if abs(gt_val - ext_val) <= config.tolerance:
                                    matched = True
                            elif gt_val == ext_val:
                                matched = True

                            if matched:
                                match_score += config.weight
                            total_weight += config.weight

                    weighted_score = (
                        match_score / total_weight if total_weight > 0 else 0
                    )
                    if weighted_score > best_score:
                        best_score = weighted_score
                        best_match = ext_row

                if best_match is not None:
                    matched_results.append(best_match)

        # Update extracted_data with matched results
        extracted_data = pd.DataFrame(matched_results)

    # Ensure both DataFrames have pmcid column
    if "pmcid" not in ground_truth.columns or "pmcid" not in extracted_data.columns:
        raise ValueError(
            "Both ground truth and extracted data must have 'pmcid' column"
        )

    # Sort both DataFrames by pmcid to ensure matching order
    ground_truth = ground_truth.sort_values("pmcid").reset_index(drop=True)
    extracted_data = extracted_data.sort_values("pmcid").reset_index(drop=True)

    # For participant demographics, consider it a match if any extracted group matches ground truth
    if dataset_type == "participant_demographics":
        matched_data = []
        for pmcid in ground_truth["pmcid"].unique():
            gt_row = ground_truth[ground_truth["pmcid"] == pmcid].iloc[0]
            extracted_groups = extracted_data[extracted_data["pmcid"] == pmcid]

            # Find best matching group
            best_match = None
            for _, group in extracted_groups.iterrows():
                score = 0.0
                total_weight = 0.0

                for field, config in field_configs.items():
                    if field not in gt_row or field not in group:
                        continue

                    gt_val = gt_row[field]
                    ext_val = group[field]

                    if pd.notna(gt_val) and pd.notna(ext_val):
                        # Apply normalization
                        if config.normalize_func:
                            gt_val = config.normalize_func(gt_val)
                            ext_val = config.normalize_func(ext_val)

                        # Compare with tolerance
                        matched = False
                        if isinstance(gt_val, (int, float)) and isinstance(
                            ext_val, (int, float)
                        ):
                            if abs(gt_val - ext_val) <= config.tolerance:
                                matched = True
                        elif gt_val == ext_val:
                            matched = True

                        if matched:
                            score += config.weight
                        total_weight += config.weight

                weighted_score = score / total_weight if total_weight > 0 else 0
                if best_match is None or weighted_score > best_match[1]:
                    best_match = (group, weighted_score)

            if best_match is not None:
                matched_data.append(best_match[0])

        # Create new DataFrame with best matches
        extracted_data = pd.DataFrame(matched_data)

    # Prepare data for metrics calculation
    metrics_data = extracted_data.copy()
    metrics_truth = ground_truth.copy()

    # Handle list fields in task data
    if dataset_type == "task":
        list_fields = [
            "Modality",
            "TaskName",
            "TaskDescription",
            "Condition",
            "ContrastDefinition",
        ]
        for field in list_fields:
            if field in metrics_data.columns:
                metrics_data[field] = metrics_data[field].apply(
                    lambda x: x[0] if isinstance(x, list) and x else ""
                )
            if field in metrics_truth.columns:
                metrics_truth[field] = metrics_truth[field].apply(
                    lambda x: x[0] if isinstance(x, list) and x else ""
                )

    # Convert FieldConfig to dict format expected by calculate_metrics
    metrics_field_configs = {
        field: {
            "normalize": (
                normalize_text
                if config.normalize_func == normalize_text
                else config.normalize_func
            ),
            "tolerance": config.tolerance,
            "weight": config.weight,
        }
        for field, config in field_configs.items()
    }

    # Calculate metrics
    metrics_results = calculate_metrics(
        metrics_truth, metrics_data, metrics_field_configs
    )

    # Analyze errors
    error_analysis = analyze_errors(ground_truth, extracted_data, field_configs)

    return metrics_results, error_analysis


def identify_improvements(error_analysis: Dict[str, Dict]) -> List[str]:
    """Identify potential areas for improvement based on error analysis.

    Analyzes error patterns and rates to generate prioritized improvement suggestions.

    Args:
        error_analysis: Dictionary containing error analysis results

    Returns:
        List of improvement suggestions ordered by priority
    """
    improvements = []

    # Sort fields by error rate
    sorted_fields = sorted(
        error_analysis.items(), key=lambda x: x[1]["error_rate"], reverse=True
    )

    for field, analysis in sorted_fields:
        if analysis["error_rate"] > 0.3:  # High error rate threshold
            improvements.append(
                f"High priority: Improve {field} extraction "
                f"(Error rate: {analysis['error_rate']:.2%})"
            )

            missing = len(analysis["error_types"].get("Missing", []))
            incorrect = len(analysis["error_types"].get("Incorrect", []))

            if missing > incorrect:
                improvements.append(
                    f"  - Focus on improving {field} detection/coverage"
                )
            else:
                improvements.append(
                    f"  - Focus on improving {field} accuracy/normalization"
                )

    return improvements


def main():
    """Main execution function."""
    # First run the extraction pipelines
    print("Step 1: Running Extraction Pipelines")
    print("===================================")
    output_path = run_extraction_pipelines()

    print("\nStep 2: Evaluating Results")
    print("=========================")

    # Evaluate participant demographics
    print("\nEvaluating Participant Demographics Extraction...")
    pd_extracted = load_extracted_results(output_path, "participant_demographics")
    pd_metrics, pd_errors = run_evaluation("participant_demographics", pd_extracted)
    print("\nParticipant Demographics Results:")
    print("================================")
    print(format_results(pd_metrics))
    print("\nParticipant Demographics Error Analysis:")
    print(generate_error_report(pd_errors))

    # Evaluate NV task (using 'task' directory name)
    print("\nEvaluating NV Task Extraction...")
    nv_extracted = load_extracted_results(output_path, "task")
    nv_metrics, nv_errors = run_evaluation("task", nv_extracted)
    print("\nNV Task Results:")
    print("===============")
    print(format_results(nv_metrics))
    print("\nNV Task Error Analysis:")
    print(generate_error_report(nv_errors))

    # Generate improvement recommendations
    print("\nStep 3: Improvement Recommendations")
    print("=================================")
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

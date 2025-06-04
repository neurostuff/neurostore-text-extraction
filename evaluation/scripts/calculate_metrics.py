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
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


# Normalization Functions
def normalize_text(text: Union[str, List[str], np.ndarray]) -> Union[str, List[str]]:
    """Normalize text fields by removing extra whitespace and converting to lowercase.
    Handles both single strings and lists of strings."""
    # Convert numpy array to list
    if isinstance(text, np.ndarray):
        text = text.tolist()

    # Handle lists
    if isinstance(text, list):
        if not text:  # Empty list
            return []
        # Normalize each non-null element
        return [
            re.sub(r"\s+", " ", str(item).strip().lower())
            for item in text
            if not (pd.isna(item) or item is None)
        ]

    # Handle single values
    if pd.isna(text) or text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def normalize_numeric(value: Union[str, float, int]) -> float:
    """Convert numeric values to float, handling various formats."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        # Remove non-numeric characters except decimal points
        cleaned = re.sub(r"[^\d.-]", "", value)
        return float(cleaned) if cleaned else np.nan
    return float(value)


def normalize_boolean(value: Union[str, bool]) -> bool:
    """Normalize boolean values, handling various formats."""
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "yes", "1", "t", "y")
    return bool(value)


def normalize_age_range(age: Union[str, float]) -> Tuple[float, float]:
    """Convert age values to standardized range format."""
    if pd.isna(age):
        return (np.nan, np.nan)

    if isinstance(age, (int, float)):
        return (float(age), float(age))

    # Handle ranges like "18-25" or "18 to 25"
    range_match = re.search(r"(\d+)(?:\s*[-to]+\s*)(\d+)", str(age))
    if range_match:
        return (float(range_match.group(1)), float(range_match.group(2)))

    # Handle single numbers in string format
    number_match = re.search(r"(\d+)", str(age))
    if number_match:
        value = float(number_match.group(1))
        return (value, value)

    return (np.nan, np.nan)


def calculate_field_metrics(
    ground_truth: pd.Series,
    extracted: pd.Series,
    normalization_func=None,
    tolerance: float = 0.0,
) -> Dict[str, float]:
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
        matches = sum(
            abs(g - e) <= tolerance
            for g, e in zip(ground_truth, extracted)
            if not (pd.isna(g) or pd.isna(e))
        )
    else:
        matches = 0
        for g, e in zip(ground_truth, extracted):
            if pd.isna(g) or pd.isna(e):
                continue

            # Handle list comparisons
            if isinstance(g, list) and isinstance(e, list):
                # Lists must contain same elements in any order
                g_normalized = set(normalize_text(item) for item in g)
                e_normalized = set(normalize_text(item) for item in e)
                if g_normalized == e_normalized:
                    matches += 1
            else:
                # Direct comparison for non-list values
                if g == e:
                    matches += 1

    precision = matches / total_extracted if total_extracted > 0 else 0.0
    recall = matches / total_ground_truth if total_ground_truth > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
        "total_ground_truth": total_ground_truth,
        "total_extracted": total_extracted,
    }


def calculate_metrics(
    ground_truth_df: pd.DataFrame,
    extracted_df: pd.DataFrame,
    field_configs: Dict[str, Dict],
) -> Dict[str, Dict]:
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
    aggregate_metrics = {"precision": [], "recall": [], "f1": []}

    for field, config in field_configs.items():
        if field not in ground_truth_df.columns or field not in extracted_df.columns:
            continue

        field_metrics = calculate_field_metrics(
            ground_truth_df[field],
            extracted_df[field],
            normalization_func=config.get("normalize"),
            tolerance=config.get("tolerance", 0.0),
        )

        results[field] = field_metrics

        # Add to aggregate calculations
        aggregate_metrics["precision"].append(field_metrics["precision"])
        aggregate_metrics["recall"].append(field_metrics["recall"])
        aggregate_metrics["f1"].append(field_metrics["f1"])

    # Calculate aggregate metrics
    results["aggregate"] = {
        "precision": np.mean(aggregate_metrics["precision"]),
        "recall": np.mean(aggregate_metrics["recall"]),
        "f1": np.mean(aggregate_metrics["f1"]),
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
        if field == "aggregate":
            continue
        line = f"{field:<20} {metrics['precision']:>10.2%} {metrics['precision']:>10.2%} {metrics['f1']:>10.2%}"
        output.append(line)

    # Aggregate metrics
    output.append("\nAggregate Metrics:")
    output.append("-----------------")
    agg = results["aggregate"]
    output.append(f"Overall Precision: {agg['precision']:.2%}")
    output.append(f"Overall Recall: {agg['recall']:.2%}")
    output.append(f"Overall F1 Score: {agg['f1']:.2%}")

    return "\n".join(output)

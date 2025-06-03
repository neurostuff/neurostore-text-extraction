#!/usr/bin/env python3
"""Script to run participant demographics evaluation on test set."""

from pathlib import Path
import logging
from typing import Optional

from evaluation.evaluate_participant_demographics import evaluate_extractions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_evaluation(
    test_set: Optional[str] = None,
    output_dir: Optional[str] = None
) -> None:
    """Run evaluation on participant demographics extraction.
    
    Args:
        test_set: Optional name of test set to evaluate
        output_dir: Optional directory to write results
    """
    # Set paths
    eval_dir = Path(__file__).parent.parent
    data_dir = eval_dir / 'data'
    
    # Ground truth path
    ground_truth_path = data_dir / 'participant_demographics_ground_truth.csv'
    if not ground_truth_path.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {ground_truth_path}"
        )
    
    # Extraction dir
    if test_set:
        extraction_dir = data_dir / 'extractions' / test_set
    else:
        extraction_dir = data_dir / 'extractions' / 'test'
        
    if not extraction_dir.exists():
        raise FileNotFoundError(
            f"Extraction directory not found: {extraction_dir}"
        )
    
    # Output path
    if output_dir:
        output_path = Path(output_dir) / 'participant_demographics_results.json'
    else:
        output_path = data_dir / 'results' / 'participant_demographics_results.json'
        
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Running evaluation with:")
    logger.info(f"Ground truth: {ground_truth_path}")
    logger.info(f"Extractions: {extraction_dir}")
    logger.info(f"Output: {output_path}")
    
    # Run evaluation
    results = evaluate_extractions(
        str(extraction_dir),
        str(ground_truth_path),
        str(output_path)
    )
    
    # Log results
    logger.info("\nResults:")
    for field, value in results['overall'].items():
        if value is not None:
            logger.info(f"{field}: {value:.3f}")
        else:
            logger.info(f"{field}: N/A")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run participant demographics evaluation'
    )
    parser.add_argument(
        '--test-set',
        help='Name of test set to evaluate'
    )
    parser.add_argument(
        '--output-dir',
        help='Directory to write results'
    )
    
    args = parser.parse_args()
    
    run_evaluation(
        test_set=args.test_set,
        output_dir=args.output_dir
    )

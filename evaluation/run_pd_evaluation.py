#!/usr/bin/env python3
"""Run and evaluate participant demographics extraction."""

import logging
from pathlib import Path

from ns_extract.dataset import Dataset
from ns_extract.pipelines.participant_demographics.model import ParticipantDemographicsExtractor
from evaluation.evaluate_participant_demographics import evaluate_extractions


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run extraction pipeline and evaluation."""
    # Set up paths - using absolute paths to avoid duplication
    base_dir = Path(__file__).resolve().parent
    input_dir = base_dir / "data/ns_pond_inputs"
    output_dir = base_dir / "data/pipeline_outputs"
    ground_truth_path = base_dir / "data/participant_demographics_ground_truth.csv"
    results_dir = base_dir / "data/results"
    env_file = (base_dir.parent / ".env").resolve()

    # Ensure output directories exist
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info(f"Loading dataset from {input_dir}")
    if not input_dir.exists():
        raise RuntimeError(f"Input directory not found: {input_dir}")
    dataset = Dataset(input_dir)

    # Run extraction pipeline
    logger.info("Running participant demographics extraction...")
    extractor = ParticipantDemographicsExtractor(
        extraction_model="gpt-4o-mini-2024-07-18",  # Specify the required extraction model
        env_file=env_file  # Pass API key for authentication
    )
    hash_dir = extractor.transform_dataset(dataset, output_dir)

    # Find results directory
    extract_dir = hash_dir
    if not extract_dir.exists():
        raise RuntimeError(f"Results directory not found: {extract_dir}")

    # Run evaluation
    logger.info("Running evaluation...")
    eval_results = evaluate_extractions(
        str(extract_dir),
        str(ground_truth_path),
        str(results_dir / "participant_demographics_results.json")
    )

    # Print results 
    logger.info("\nEvaluation Results:")
    for field, value in eval_results["overall"].items():
        if value is not None:
            logger.info(f"{field}: {value:.3f}")
        else:
            logger.info(f"{field}: N/A")


if __name__ == "__main__":
    main()

"""Evaluation framework for comparing extractor outputs with ground truth."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml

from pydantic import BaseModel


class MetricConfig(BaseModel):
    """Configuration for evaluation metrics."""
    accuracy: bool = True
    precision: bool = True
    recall: bool = True
    per_field_metrics: List[str] = []


class TestSetConfig(BaseModel):
    """Configuration for test sets."""
    num_studies: int = 20
    extractors: List[str] = []


class BaselineConfig(BaseModel):
    """Configuration for baseline metrics."""
    thresholds: Dict[str, float]


class EvalConfig(BaseModel):
    """Top-level evaluation configuration."""
    version: str
    metrics: MetricConfig
    test_sets: Dict[str, TestSetConfig]
    baseline: BaselineConfig


class EvaluationFramework:
    """Framework for running extractor evaluations against ground truth."""

    def __init__(
        self,
        config_path: Union[str, Path],
        test_data_dir: Union[str, Path],
        ground_truth_dir: Union[str, Path],
        results_dir: Optional[Union[str, Path]] = None
    ):
        """Initialize the evaluation framework.
        
        Args:
            config_path: Path to evaluation config YAML
            test_data_dir: Directory containing test input data
            ground_truth_dir: Directory containing ground truth files
            results_dir: Optional directory for storing results
        """
        self.config_path = Path(config_path)
        self.test_data_dir = Path(test_data_dir)
        self.ground_truth_dir = Path(ground_truth_dir)
        self.results_dir = Path(results_dir) if results_dir else Path("evaluation/results")
        self.config = self._load_config()

    def _load_config(self) -> EvalConfig:
        """Load and validate evaluation configuration."""
        with open(self.config_path) as f:
            config_dict = yaml.safe_load(f)
        return EvalConfig.model_validate(config_dict)

    def get_available_studies(self) -> List[str]:
        """Get list of study IDs available in both test data and ground truth."""
        test_studies = {p.name for p in self.test_data_dir.glob("*") if p.is_dir()}
        truth_studies = {
            p.stem for p in self.ground_truth_dir.glob("*.json")
        }
        return sorted(list(test_studies & truth_studies))

    def load_ground_truth(self, study_id: str) -> Dict:
        """Load ground truth data for a specific study.
        
        Args:
            study_id: ID of study to load ground truth for
            
        Returns:
            Ground truth data for the study
        """
        truth_path = self.ground_truth_dir / f"{study_id}.json"
        if not truth_path.exists():
            raise FileNotFoundError(f"No ground truth found for study {study_id}")
            
        with open(truth_path) as f:
            return json.load(f)

    def compute_metrics(
        self,
        extractor_output: Dict,
        ground_truth: Dict,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Compute evaluation metrics comparing extractor output to ground truth.
        
        Args:
            extractor_output: Output from the extractor
            ground_truth: Ground truth data to compare against
            metrics: Optional list of specific metrics to compute
            
        Returns:
            Dict mapping metric names to values
        """
        if metrics is None:
            metrics = []
            if self.config.metrics.accuracy:
                metrics.append("accuracy")
            if self.config.metrics.precision:
                metrics.append("precision") 
            if self.config.metrics.recall:
                metrics.append("recall")
            metrics.extend(self.config.metrics.per_field_metrics)

        results = {}
        for metric in metrics:
            if metric == "accuracy":
                results[metric] = self._compute_accuracy(extractor_output, ground_truth)
            elif metric == "precision":
                results[metric] = self._compute_precision(extractor_output, ground_truth)
            elif metric == "recall":
                results[metric] = self._compute_recall(extractor_output, ground_truth)
            elif metric in self.config.metrics.per_field_metrics:
                results[metric] = self._compute_field_metric(
                    extractor_output, ground_truth, metric
                )

        return results

    def _compute_accuracy(self, output: Dict, truth: Dict) -> float:
        """Compute overall accuracy between extractor output and ground truth."""
        # Count exact matches between corresponding fields
        matches = 0
        total = 0
        
        # Compare counts
        for group in truth["groups"]:
            output_group = next(
                (g for g in output["groups"] if g["group_name"] == group["group_name"]),
                None
            )
            if not output_group:
                continue
                
            for field in ["count", "male_count", "female_count"]:
                if group.get(field) is not None and output_group.get(field) is not None:
                    total += 1
                    if group[field] == output_group[field]:
                        matches += 1
                        
            # Compare age fields with tolerance
            for field in ["age_mean", "age_minimum", "age_maximum"]:
                if group.get(field) is not None and output_group.get(field) is not None:
                    total += 1
                    # Allow 1-year tolerance for age fields
                    if abs(group[field] - output_group[field]) <= 1:
                        matches += 1
        
        return matches / total if total > 0 else 0.0

    def _compute_precision(self, output: Dict, truth: Dict) -> float:
        """Compute precision between extractor output and ground truth."""
        # For demographics, precision is the fraction of extracted values that are correct
        true_positives = 0
        false_positives = 0
        
        for output_group in output["groups"]:
            truth_group = next(
                (g for g in truth["groups"] if g["group_name"] == output_group["group_name"]),
                None
            )
            if not truth_group:
                # Extracted group not in ground truth
                false_positives += 1
                continue
                
            # Check each field
            for field in ["count", "male_count", "female_count"]:
                if output_group.get(field) is not None:
                    if truth_group.get(field) == output_group[field]:
                        true_positives += 1
                    else:
                        false_positives += 1
                        
            # Check age fields with tolerance
            for field in ["age_mean", "age_minimum", "age_maximum"]:
                if output_group.get(field) is not None:
                    if (
                        truth_group.get(field) is not None 
                        and abs(truth_group[field] - output_group[field]) <= 1
                    ):
                        true_positives += 1
                    else:
                        false_positives += 1
        
        return (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )

    def _compute_recall(self, output: Dict, truth: Dict) -> float:
        """Compute recall between extractor output and ground truth."""
        # For demographics, recall is the fraction of ground truth values that were extracted
        true_positives = 0
        false_negatives = 0
        
        for truth_group in truth["groups"]:
            output_group = next(
                (g for g in output["groups"] if g["group_name"] == truth_group["group_name"]),
                None
            )
            if not output_group:
                # Ground truth group not found in output
                false_negatives += 1
                continue
                
            # Check each field
            for field in ["count", "male_count", "female_count"]:
                if truth_group.get(field) is not None:
                    if output_group.get(field) == truth_group[field]:
                        true_positives += 1
                    else:
                        false_negatives += 1
                        
            # Check age fields with tolerance
            for field in ["age_mean", "age_minimum", "age_maximum"]:
                if truth_group.get(field) is not None:
                    if (
                        output_group.get(field) is not None
                        and abs(truth_group[field] - output_group[field]) <= 1
                    ):
                        true_positives += 1
                    else:
                        false_negatives += 1
        
        return (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )

    def _compute_field_metric(self, output: Dict, truth: Dict, field: str) -> float:
        """Compute accuracy for a specific demographic field."""
        matches = 0
        total = 0
        
        for truth_group in truth["groups"]:
            output_group = next(
                (g for g in output["groups"] if g["group_name"] == truth_group["group_name"]),
                None
            )
            if not output_group:
                continue
                
            if truth_group.get(field) is not None and output_group.get(field) is not None:
                total += 1
                if field in ["age_mean", "age_minimum", "age_maximum"]:
                    # Allow 1-year tolerance for age fields
                    if abs(truth_group[field] - output_group[field]) <= 1:
                        matches += 1
                else:
                    if truth_group[field] == output_group[field]:
                        matches += 1
                    
        return matches / total if total > 0 else 0.0

    def save_results(self, results: Dict, name: str):
        """Save evaluation results.
        
        Args:
            results: Results dict to save
            name: Name for the results file
        """
        results_path = self.results_dir / f"{name}.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    def compare_to_baseline(self, results: Dict) -> bool:
        """Compare results to baseline thresholds.
        
        Args:
            results: Results dict to compare
            
        Returns:
            True if results meet or exceed baseline thresholds
        """
        for metric, threshold in self.config.baseline.thresholds.items():
            if results.get(metric, 0) < threshold:
                return False
        return True

    def run_evaluation(self, test_set: str = "default") -> Dict:
        """Run complete evaluation using specified test set.
        
        Args:
            test_set: Name of test set configuration to use
            
        Returns:
            Evaluation results
        """
        if test_set not in self.config.test_sets:
            raise ValueError(f"Unknown test set: {test_set}")

        test_config = self.config.test_sets[test_set]
        
        # Get available studies
        studies = self.get_available_studies()
        if len(studies) < test_config.num_studies:
            raise ValueError(
                f"Not enough studies available. Need {test_config.num_studies}, "
                f"found {len(studies)}"
            )

        results = {
            "config": {
                "test_set": test_set,
                "num_studies": test_config.num_studies,
                "metrics": self.config.metrics.dict(),
            },
            "extractors": {}
        }

        # Run evaluation for each extractor
        for extractor_name in test_config.extractors:
            extractor_results = {
                "metrics": {},
                "studies": {}
            }
            
            # Get extractor instance
            try:
                # Import extractor class dynamically
                module_path = f"ns_extract.pipelines.{extractor_name}.model"
                class_name = "".join(x.capitalize() for x in extractor_name.split("_"))
                
                import importlib
                module = importlib.import_module(module_path)
                extractor_class = getattr(module, f"{class_name}Extractor")
                extractor = extractor_class()
                
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Failed to load extractor {extractor_name}: {e}")
            
            # Evaluate each study
            for study_id in studies[:test_config.num_studies]:
                # Load ground truth
                ground_truth = self.load_ground_truth(study_id)
                
                # Run extractor
                study_dir = self.test_data_dir / study_id
                extractor_output = extractor.extract(study_dir)
                
                # Compute metrics
                study_metrics = self.compute_metrics(extractor_output, ground_truth)
                extractor_results["studies"][study_id] = {
                    "metrics": study_metrics,
                    "output": extractor_output
                }
            
            # Compute overall metrics
            overall_metrics = {}
            for metric in study_metrics:
                values = [
                    s["metrics"][metric] 
                    for s in extractor_results["studies"].values()
                ]
                overall_metrics[metric] = sum(values) / len(values)
            
            extractor_results["metrics"] = overall_metrics
            results["extractors"][extractor_name] = extractor_results

        return results

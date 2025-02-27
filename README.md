# ns-extract

NeuroStore extraction pipelines for analyzing scientific papers.

## Installation

```bash
pip install -e .
```

## Usage

You can run pipelines in two ways:

1. Using command line arguments:

```bash 
python scripts /path/to/dataset /path/to/output --pipelines word_count task participant_demographics
```

2. Using a YAML configuration file:

```bash
ns-extract /path/to/dataset /path/to/output --config pipeline_config.yaml
```

Example configuration file:

```yaml
pipelines:
  # Simple pipeline without arguments
  - word_count

  # Pipeline with configuration arguments
  - name: participant_demographics
    args:
      model_name: "gpt-4"
      temperature: 0.7

  # Another pipeline with different arguments
  - name: task
    args:
      extraction_method: "hybrid"
      confidence_threshold: 0.8
```

Available pipelines:
- word_count
- word_deviance
- task
- participant_demographics
# ns-extract

NeuroStore extraction pipelines for extracting information from scientific articles. 

## Installation

```bash
pip install -e .
```

## Usage

Pipelines are applied to the neurostore-data-pond (on-disk represenation of NeuroStore's database index of fMRI articles)

1. Using command line arguments:

```bash 
ns-extract /path/to/dataset /path/to/output --pipelines word_count task participant_demographics
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

## Documentation

- [Schema Metadata System](docs/schema_metadata.md) - Documentation for the schema metadata system that controls text processing operations (normalization, abbreviation expansion) while maintaining clean schema definitions.

# Schema Metadata System Documentation

## Overview

The schema metadata system is a powerful feature that separates processing instructions from schema definitions in the neurostore-text-extraction project. It allows developers to annotate schema fields with metadata tags that control text processing operations while maintaining clean schema output for LLMs.

### Key Benefits

1. **Separation of Concerns**: Keeps schema definitions focused on data structure while handling processing logic separately
2. **Flexible Processing**: Supports multiple processing operations through metadata tags
3. **Clean Output**: Ensures processed data meets schema requirements while preserving original definitions
4. **Maintainable Code**: Centralizes text processing logic and makes it reusable across schemas

## Usage Guide

### Adding Metadata to Schema Fields

Schema metadata is added using the `json_schema_extra` parameter in Pydantic field definitions:

```python
from pydantic import BaseModel, Field

class ExampleSchema(BaseModel):
    value: str = Field(
        description="Example field with processing metadata",
        json_schema_extra={
            "normalize_text": True,
            "expand_abbreviations": True
        }
    )
```

### Available Metadata Tags

1. **normalize_text**
   - Purpose: Standardizes text formatting
   - Effects: 
     - Strips whitespace
     - Converts to title case
     - Handles null values ("None", "N/A", etc.)
   - Example: "some text  " → "Some Text"

2. **expand_abbreviations**
   - Purpose: Expands abbreviated terms to their full form
   - Effects:
     - Uses scispacy to detect abbreviations in source text
     - Replaces abbreviations with their expanded forms
   - Example: "MRI scan" → "Magnetic Resonance Imaging scan"

### Example Schema Definition

```python
class ParticipantGroup(BaseModel):
    name: str = Field(
        description="Name of the participant group",
        json_schema_extra={"normalize_text": True}
    )
    
    diagnosis: str = Field(
        description="Clinical diagnosis of the group",
        json_schema_extra={
            "normalize_text": True,
            "expand_abbreviations": True
        }
    )
```

## Implementation Details

### Processing Pipeline

1. **Field Collection**
   - During extractor initialization, the system scans schema definitions
   - Fields with metadata tags are collected and stored for processing
   - Nested fields are handled using dot notation (e.g., "groups[].diagnosis")

2. **Text Processing**
   - Post-processing occurs after initial transformation
   - Source text is analyzed for abbreviations (if needed)
   - Fields are processed according to their metadata tags
   - Processing order: abbreviation expansion → text normalization

### Nested Field Handling

The system supports processing fields at any level of nesting:

- Simple fields: `field_name`
- Nested objects: `parent.field_name`
- List items: `list_field[].field_name`
- Dictionary values: `dict_field[].field_name`

```python
class NestedSchema(BaseModel):
    groups: List[ParticipantGroup]  # Will process each group's fields
    metadata: Dict[str, str]        # Will process dictionary values
```

## Best Practices

### When to Use Metadata Tags

1. **normalize_text**
   - Use for fields that need consistent formatting
   - Appropriate for categorical data, names, labels
   - Helpful for downstream analysis and comparison

2. **expand_abbreviations**
   - Use for fields containing domain-specific terminology
   - Important for medical terms, technical abbreviations
   - Enhances readability and standardization

### Testing Metadata-Enhanced Schemas

1. Test field processing:
   ```python
   def test_field_normalization():
       schema = ExampleSchema(value="  test value  ")
       assert schema.value == "Test Value"
   ```

2. Test abbreviation expansion:
   ```python
   def test_abbreviation_expansion():
       text = "MRI (Magnetic Resonance Imaging) scan"
       schema = ExampleSchema(value="The MRI scan")
       assert "Magnetic Resonance Imaging" in schema.value
   ```

### Common Pitfalls to Avoid

1. **Over-processing**
   - Don't add metadata tags to fields that don't need processing
   - Consider the impact on performance and data integrity

2. **Inconsistent Application**
   - Apply metadata tags consistently across similar fields
   - Document any exceptions or special cases

3. **Missing Source Text**
   - Ensure source text is available when using expand_abbreviations
   - Handle cases where abbreviation context is missing

4. **Circular References**
   - Avoid processing the same field multiple times
   - Be careful with recursive schema definitions

## Recommended Workflow

1. Define schema structure and field types
2. Identify fields needing processing
3. Add appropriate metadata tags
4. Test processing outcomes
5. Monitor and adjust as needed

By following these guidelines, you can effectively use the schema metadata system to maintain clean, well-structured data while applying necessary processing operations.

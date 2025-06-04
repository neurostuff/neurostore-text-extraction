# TaskExtractor Version Comparison (1.0.0 vs 1.1.0)

This document compares the results between TaskExtractor versions 1.0.0 and 1.1.0 for 14 different studies.

## Key Improvements in Version 1.1.0

1. Enhanced Detail and Specificity
   - Task descriptions are more comprehensive and precise
   - Task metrics are more specifically defined with measurement units/methods
   - Study objectives contain more detailed methodological information

2. Behavioral Tasks Extraction
   - Version 1.0.0 often omitted behavioral tasks
   - Version 1.1.0 consistently includes behavioral tasks with:
     - Detailed task descriptions
     - Specific design details
     - Clear task metrics
     - Associated cognitive domains

3. Resting State Metadata
   - Version 1.0.0: Minimal or null RestingStateMetadata
   - Version 1.1.0: Structured metadata including:
     - Total duration information
     - Participant instructions
     - Eyes open/closed requirements
     - Physiological measurements

## Example Comparisons

### Study: 455y7Qh5jgni
- Task metrics improved from general "Response Time, Accuracy" to specific "Binary sequences indicating correct versus incorrect responses, Response times measured within 3 seconds"
- TaskDesign changed from "EventRelated" to more accurate "Mixed"
- More nuanced concept categorization

### Study: 834PWSjKgSGe
- Added detailed RestingStateMetadata including participant instructions
- Expanded TaskMetrics to include specific measurements (ALFF, FC to PCC)
- Added comprehensive behavioral tasks (MMSE, MoCA, Digit Symbol Test, etc.)

### Study: 3XC7fFxU7Sn6
- More descriptive TaskName: "Dorsiflexion Task" → "Ankle Dorsiflexion Task"
- Expanded TaskMetrics from single "fMRI BOLD signal" to multiple specific metrics:
  - Brain activation patterns
  - Functional connectivity
  - Signal intensity changes
- Enhanced Concepts list with additional detail (added "Voluntary movement")
- Modified TaskDesign classification from "EventRelated" to "Mixed"

## Common Patterns Across Studies

1. Task Classification Refinement
   - Version 1.1.0 shows more careful consideration of task design classification
   - Several tasks reclassified from "EventRelated" to "Mixed" when appropriate

2. Metric Specification
   - Version 1.0.0: Generally uses broad, single metrics
   - Version 1.1.0: Breaks down into multiple specific measurements
   - Includes measurement methods and units where applicable

3. Conceptual Framework
   - Version 1.1.0 provides more nuanced and complete concept lists
   - Better captures the multi-dimensional nature of cognitive tasks

### Study: 4H46zmwJ7MNZ
- Added separate Behavioral Tasks section for probe session
- Enhanced task metrics specification:
  - Version 1.0.0: "fMRI BOLD signal, Response accuracy"
  - Version 1.1.0: Specific "Accuracy of responses, Reaction time" for both fMRI and behavioral tasks
- More detailed timing information in DesignDetails (1500ms response window, 400ms feedback)
- Domain categorization refined to include "Executive cognitive control" instead of just "Attention"

## Overall Improvements in 1.1.0

1. Response Timing Details
   - Added specific timing windows for responses
   - Included feedback durations
   - Better documentation of trial structure

2. Task Organization
   - Clear separation between fMRI and behavioral components
   - Better documentation of task relationships
   - More systematic organization of task components

3. Domain Specificity
   - More precise cognitive domain categorization
   - Better alignment with standardized neuropsychological terminology
   - Clearer distinction between primary and secondary domains

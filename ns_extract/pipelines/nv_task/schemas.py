from typing import List
from pydantic import BaseModel, Field
from typing_extensions import Literal, Optional, Dict


class TaskMetadataModel(BaseModel):
    TaskName: str = Field(
        description=(
            "Exact name of the task as stated in text (e.g., 'Stroop Task', 'Go/No-Go Task'). "
            "If no explicit name is provided, "
            "create brief descriptive name based on core task features. "
            "Use verbatim terminology from source for any technical/scientific terms."
        )
    )
    TaskDescription: str = Field(
        description=(
            "1-2 sentence summary capturing: "
            "(1) What participants were instructed to do "
            "(2) Type of stimuli/materials used "
            "(3) Primary measures/outcomes "
            "(4) Overall task objective "
            "Use direct quotes where possible. Maintain original terminology."
        )
    )
    DesignDetails: str = Field(
        description=(
            "Detailed task design description including ALL of: "
            "- Design type (block/event-related/mixed) "
            "- Number and duration of runs/blocks/trials "
            "- Trial structure and timing "
            "- Inter-trial/block intervals "
            "- Stimulus presentation parameters "
            "- Response collection methods "
            "Quote directly from text. Flag any missing key details."
        )
    )
    Conditions: Optional[List[str]] = Field(
        description=(
            "Complete list of distinct experimental conditions and control conditions. "
            "Include ALL conditions mentioned in design or analysis. "
            "Use exact names/labels from text. Note any hierarchical/nested structure."
        )
    )
    TaskMetrics: Optional[List[str]] = Field(
        description=(
            "ALL outcomes measured during task execution including: "
            "- Behavioral measures (e.g., accuracy, reaction time) "
            "- Neural measures (e.g., BOLD response) "
            "- Subjective measures (e.g., ratings) "
            "Use precise terminology from source text."
        )
    )
    Concepts: Optional[List[str]] = Field(
        description=(
            "List of specific mental processes and cognitive concepts that the task "
            "engages or measures, including: "
            "- Core cognitive processes (e.g., 'working memory', 'attention') "
            "- Specific mechanisms (e.g., 'response inhibition', 'conflict monitoring') "
            "- Perceptual processes (e.g., 'visual perception', 'auditory processing') "
            "- Target mental constructs (e.g., 'emotion regulation', 'reward learning') "
            "Extract ONLY terms explicitly mentioned in text. Use exact terminology."
        )
    )
    Domain: Optional[
        List[
            Literal[
                "Perception",
                "Attention",
                "Reasoning and decision making",
                "Executive cognitive control",
                "Learning and memory",
                "Language",
                "Action",
                "Emotion",
                "Social function",
                "Motivation",
            ]
        ]
    ] = Field(
        description=(
            "Primary cognitive domain(s) engaged by the task. "
            "Select ALL that apply based on explicit task description and measures. "
            "Do not infer domains not clearly indicated in text."
        )
    )


class fMRITaskMetadataModel(TaskMetadataModel):
    RestingState: bool = Field(
        description=(
            "Indicate if this was a resting state acquisition. "
            "Set true ONLY if explicitly described as resting state, "
            "rest period, or baseline state with no active task demands."
        )
    )
    RestingStateMetadata: Optional[Dict[str, str]] = Field(
        description=(
            "For resting state tasks ONLY, include following details if available: "
            "- Total duration of rest periods "
            "- Specific instructions given to participants "
            "- Eyes open/closed requirements "
            "- Any concurrent physiological measurements "
            "Return null for non-resting state tasks. Use exact descriptions from text."
        ),
        default=None,
    )
    TaskDesign: List[Literal["Blocked", "EventRelated", "Mixed", "Other"]] = Field(
        description="Design(s) of the task"
    )
    TaskDuration: Optional[str] = Field(
        description="Total duration of the task, e.g., '10 minutes' or '600 seconds'."
    )


class StudyMetadataModel(BaseModel):
    """Model for capturing fMRI study metadata including tasks and imaging details"""

    Modality: List[
        Literal[
            "fMRI-BOLD",
            "StructuralMRI",
            "DiffusionMRI",
            "PET FDG",
            "PET [15O]-water",
            "fMRI-CBF",
            "fMRI-CBV",
            "MEG",
            "EEG",
            "Other",
        ]
    ] = Field(
        description="Modality of the neuroimaging data",
    )
    StudyObjective: Optional[str] = Field(
        description="A brief summary of the primary research question or objective of the study."
    )
    Exclude: Optional[Literal["MetaAnalysis", "Review"]] = Field(
        description=(
            "Only studies that conduct primary data collection are to be included. Thus, if "
            "a study is primarily either a meta-analysis or a review, note here."
        )
    )
    fMRITasks: List[fMRITaskMetadataModel] = Field(
        description=(
            "List of fMRI tasks performed by the subjects inside the scanner and their metadata. "
            "If the study did not include fMRI tasks, leave this field empty."
        )
    )
    BehavioralTasks: Optional[List[TaskMetadataModel]] = Field(
        description=(
            "List of behavioral tasks performed by the subjects outside "
            "the scanner and their metadata. If the study did not include "
            "behavioral tasks, leave this field empty."
        )
    )


# No wrapper schema needed - validation is handled by pipeline using individual schemas

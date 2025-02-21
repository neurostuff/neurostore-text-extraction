from typing import List
from pydantic import BaseModel, Field
from typing_extensions import Literal, Optional, Dict

class TaskMetadataModel(BaseModel):
    TaskName: str = Field(
        description="Name of the task, e.g., 'Stroop Task' or 'Go/No-Go Task'. Provide the name as it appears in the paper or a descriptive name if unspecified."
    )
    TaskDescription: str = Field(
        description="In 1-2 sentences, describe the key features of the task, such as its purpose or what it measures."
    )
    DesignDetails: str = Field(
        description="""Provide a detailed description of the task design in up to 1 paragraph. Include 
        information on the number of conditions, the number of trials per condition, the length of trials, 
        and the length of inter-trial intervals. Quote directly from the paper where possible."""
    )
    Conditions: Optional[List[str]] = Field(
        description="Conditions of task performed by the subjects."
        )
    TaskMetrics: Optional[List[str]] = Field(
        description="Key metrics or outcomes measured during the task, e.g., 'response time', 'accuracy', 'fMRI BOLD signal'."
    )


class fMRITaskMetadataModel(TaskMetadataModel):
    RestingState: bool = Field(
        description="Was this task a resting state task?"
        )
    RestingStateMetadata: Optional[Dict[str, str]] = Field(
        description="Additional details about the resting-state task, such as duration and instructions provided to participants, if applicable."
    )
    TaskDesign: List[Literal["Blocked", "EventRelated", "Mixed", "Other"]] = Field(
        description="Design(s) of the task"
        )
    TaskDuration: Optional[str] = Field(
        description="Total duration of the task, e.g., '10 minutes' or '600 seconds'."
    )


class StudyMetadataModel(BaseModel):
    Modality: List[Literal["fMRI-BOLD", "StructuralMRI", "DiffusionMRI", "PET FDG", "PET [15O]-water", "fMRI-CBF", "fMRI-CBV", "MEG", "EEG", "Other"]] = Field(
        description="Modality of the neuroimaging data",
    )
    StudyObjective: Optional[str] = Field(
        description="A brief summary of the primary research question or objective of the study."
    )
    Exclude: Optional[Literal['MetaAnalysis', 'Review']] = Field(
        description="Only studies that conduct primary data collection are to be be included. Thus, if a study is primarily either a meta-analysis or a review, note here.",
        )
    fMRITasks: List[fMRITaskMetadataModel] = Field(
        description="List of fMRI tasks performed by the subjects inside the scanner and their metadata. If the study did not include fMRI tasks, leave this field empty."
        )
    BehavioralTasks: List[TaskMetadataModel] = Field(
        description="List of behavioral tasks performed by the subjects outside the scanner and their metadata. If the study did not include behavioral tasks, leave this field empty."
        )

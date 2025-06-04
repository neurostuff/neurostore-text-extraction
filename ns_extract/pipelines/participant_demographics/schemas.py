from typing import List, Optional
from pydantic import BaseModel, Field

from ..data_structures import NORMALIZE_TEXT, EXPAND_ABBREVIATIONS


class GroupBase(BaseModel):
    count: int = Field(
        description="Total number of participants finally included in this group. "
        "Must be explicitly stated in the text. Do not include excluded participants."
    )
    diagnosis: Optional[str] = Field(
        description="Clinical/Medical diagnosis using EXACT terminology from the text. "
        "Include subtypes and comorbidities if mentioned. Preserve technical terms precisely.",
        json_schema_extra={NORMALIZE_TEXT: True, EXPAND_ABBREVIATIONS: True},
    )
    group_name: str = Field(
        description="Primary group classification: 'healthy' for control/comparison groups, "
        "'patients' for those with clinical conditions",
        enum=["healthy", "patients"],
    )
    subgroup_name: Optional[str] = Field(
        description="The verbatim name of the group, if available",
        examples=[
            "Professional Collision Sport Athletes",
            "Young Hispanic Females",
            "Depressed Patients Without Psychotic Symptoms",
        ],
        json_schema_extra={NORMALIZE_TEXT: True, EXPAND_ABBREVIATIONS: True},
    )
    male_count: Optional[int] = Field(
        description="Number of male participants EXPLICITLY reported for this group."
    )
    female_count: Optional[int] = Field(
        description="Number of female participants EXPLICITLY reported for this group."
    )
    age_mean: Optional[float] = Field(
        default=None,
        description="Arithmetic mean age as EXPLICITLY stated in the text.",
    )
    age_range: Optional[str] = Field(
        default=None,
        description="Age range exactly as reported in the text, separated by a dash. "
        "Use null if only minimum/maximum are separately reported.",
    )
    age_minimum: Optional[int] = Field(
        default=None,
        description="Lowest age reported for this group, either as explicit minimum "
        "or lower bound of range. Must be stated in text.",
    )
    age_maximum: Optional[int] = Field(
        default=None,
        description="Highest age reported for this group, either as explicit maximum "
        "or upper bound of range. Must be stated in text.",
    )
    age_median: Optional[int] = Field(
        default=None,
        description="Median age if EXPLICITLY stated. Return null if not directly "
        "reported. Do not calculate from other values.",
    )


class GroupImaging(GroupBase):
    imaging_sample: str = Field(
        description="Indicates if this specific group underwent fMRI, MRI, or any "
        "neuroimaging procedure. Must be explicitly mentioned in text.",
        enum=["yes", "no"],
    )


class BaseDemographicsSchema(BaseModel):
    groups: List[GroupImaging]

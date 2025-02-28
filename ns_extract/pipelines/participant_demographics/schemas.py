from typing import List, Optional
from pydantic import BaseModel, Field


class GroupBase(BaseModel):
    count: int = Field(description="Number of participants in this group")
    diagnosis: Optional[str] = Field(description="Diagnosis of the group, if any")
    group_name: str = Field(description="Group name, healthy or patients",
                            enum=["healthy", "patients"])
    subgroup_name: Optional[str] = Field(description="Subgroup name")
    male_count: Optional[int] = Field(description="Number of male participants in this group")
    female_count: Optional[int] = Field(description="Number of female participants in this group")
    age_mean: Optional[float] = Field(default=None, description="Mean age of participants in this group")
    age_range: Optional[str] = Field(default=None, description="Age range of participants in this group, separated by a dash")
    age_minimum: Optional[int] = Field(default=None, description="Minimum age of participants in this group")
    age_maximum: Optional[int] = Field(default=None, description="Maximum age of participants in this group")
    age_median: Optional[int] = Field(default=None, description="Median age of participants in this group")


class GroupImaging(GroupBase):
    imaging_sample: str = Field(
        description="Did this subgroup undergo fMRI, MRI or neuroimaging, yes or no", enum=["yes", "no"])


class BaseDemographicsSchema(BaseModel):
    groups: List[GroupImaging]
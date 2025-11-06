from typing import List, Optional
from pydantic import BaseModel, Field

from ..data_structures import NORMALIZE_TEXT, EXPAND_ABBREVIATIONS


class ContrastBase(BaseModel):
    comparison: str = Field(
        description="Comparison format or direction, "
        "such as 'A vs B', 'A > B', etc.",
        examples=[
            "Childer vs Adults",
            "Placebo > Treatment",
            "Group 1 < Group 2B",
            "Healthy - Patients",
        ],
        json_schema_extra={NORMALIZE_TEXT: True, EXPAND_ABBREVIATIONS: True},
    )
    control_group: str = Field(
        description="Name of the control group, if applicable. "
        "Does not necessarily include the word 'control'.",
        examples=[
            "Healthy Controls",
            "Controls",
            "Neurotypical Controls",
            "Placebo",
            "No treatment",
        ],
        json_schema_extra={NORMALIZE_TEXT: True, EXPAND_ABBREVIATIONS: True},
    )
    group: str = Field(
        description="Any group compared against the control group. ",
        examples=[
            "Patients with Depression",
            "Treated Subjects",
            "After Treatment",
        ],
        json_schema_extra={NORMALIZE_TEXT: True, EXPAND_ABBREVIATIONS: True},
    )
    contrast_statistc: Optional[str] = Field(
        description="Statistic used for this contrast, such as 't-statistic', 'z-score', etc.",
        examples=["t-statistic", "z-score", "F-statistic", "correlation"],
        json_schema_extra={NORMALIZE_TEXT: True},
    )
    atlas: Optional[str] = Field(
        description="Atlas used for this contrast, if mentioned.",
        examples=["Harvard-Oxford", "AAL", "Schaefer"],
        json_schema_extra={NORMALIZE_TEXT: True, EXPAND_ABBREVIATIONS: True},
    )
    atlas_n_regions: Optional[int] = Field(
        description="Number of regions in the atlas, if mentioned.",
        examples=[64, 100, 200, 400],
        json_schema_extra={NORMALIZE_TEXT: True},

    )
    roi: Optional[str] = Field(
        description="Region of interest if mentioned.",
        examples=[
            "Left DLPFC",
            "Temporal gyrus",
            "Angular gyrus",
            "Calcarine fissure",
            "IPL",
            "SomMot_9",
        ],
        json_schema_extra={NORMALIZE_TEXT: True, EXPAND_ABBREVIATIONS: True},
    )

    coord_system: Optional[str] = Field(
        description="Coordinate system.",
        examples=[
            "Talairach",
            "MNI",
            "Native",
        ],
        json_schema_extra={NORMALIZE_TEXT: True},
    )
    x: Optional[int] = Field(
        default=None,
        description="Brain coordinate on the x-axis mentioned in a table or text.",
        examples=[
            "x=34",
            "x=-22",
        ],
    )
    y: Optional[int] = Field(
        default=None,
        description="Brain coordinate on the y-axis mentioned in a table or text.",
        examples=[
            "y=14",
            "y=-52",
        ],
    )
    z: Optional[int] = Field(
        default=None,
        description="Brain coordinate on the z-axis mentioned in a table or text.",
        examples=[
            "z=34",
            "z=-22",
        ],
    )
    significance: Optional[bool] = Field(
        description="Is the contrast significant? The response is binary (True or False). "
        "Usually denoted with asterisks, bold font, or p-value.",
        examples=["p < 0.05", "p < 0.001", "p = 0.001", "*", "**", "***"],
        json_schema_extra={NORMALIZE_TEXT: True},
    )

    significance_level: Optional[bool] = Field(
        description="The p-value or alpha significance threshold. E.g., 'p < 0.05', 'alpha = 0.025', etc.",
        examples=["p < 0.05", "p < 0.001", "p = 0.001", "*", "**", "***"],
        json_schema_extra={NORMALIZE_TEXT: True},
    )

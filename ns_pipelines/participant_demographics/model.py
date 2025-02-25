""" Extract participant demographics from articles. """
from .prompts import base_message
from .schemas import BaseDemographicsSchema
import pandas as pd
import numpy as np

from ns_pipelines.pipeline import BasePromptPipeline


class ParticipantDemographicsExtractor(BasePromptPipeline):
    """Participant demographics extraction pipeline."""

    _version = "1.0.0"
    _prompt = base_message
    _output_schema = BaseDemographicsSchema

    def post_process(self, result):
        # Clean known issues with GPT demographics result

        meta_keys = ["pmid", "rank", "start_char", "end_char", "id"]
        meta_keys = [k for k in meta_keys if k in result]

        # Convert JSON to DataFrame
        df = pd.json_normalize(
            result, record_path=["groups"],
            meta=meta_keys
            )
        
        df.columns = df.columns.str.replace(' ', '_')

        df = df.fillna(value=np.nan)
        df["group_name"] = df["group_name"].fillna("healthy")

        # Drop rows where count is NA
        df = df[~pd.isna(df["count"])]

        # Set group_name to healthy if no diagnosis
        df.loc[
            (df["group_name"] != "healthy") & (pd.isna(df["diagnosis"])),
            "group_name",
        ] = "healthy"

        # If no male count, substract count from female count columns
        ix_male_miss = (pd.isna(df["male_count"])) & ~(
            pd.isna(df["female_count"])
        )
        df.loc[ix_male_miss, "male_count"] = (
            df.loc[ix_male_miss, "count"]
            - df.loc[ix_male_miss, "female_count"]
        )

        # Same for female count
        ix_female_miss = (pd.isna(df["female_count"])) & ~(
            pd.isna(df["male_count"])
        )
        df.loc[ix_female_miss, "female_count"] = (
            df.loc[ix_female_miss, "count"]
            - df.loc[ix_female_miss, "male_count"]
        )

        return {"groups": df.to_dict(orient="records")}
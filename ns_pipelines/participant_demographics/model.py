""" Extract participant demographics from articles. """
from . import prompts
import pandas as pd
import numpy as np

from ns_pipelines.pipeline import BasePromptPipeline


class ParticipantDemographicsExtractor(BasePromptPipeline):
    """Participant demographics extraction pipeline."""

    _version = "1.0.0"
    _prompts = prompts

    def post_process(self, prediction):
        # Clean known issues with GPT demographics prediction

        meta_keys = ["pmid", "rank", "start_char", "end_char", "id"]
        meta_keys = [k for k in meta_keys if k in prediction]

        # Convert JSON to DataFrame
        prediction = pd.json_normalize(
            prediction, record_path=["groups"],
            meta=meta_keys
            )
        
        prediction.columns = prediction.columns.str.replace(' ', '_')

        prediction = prediction.fillna(value=np.nan)
        prediction["group_name"] = prediction["group_name"].fillna("healthy")

        # Drop rows where count is NA
        prediction = prediction[~pd.isna(prediction["count"])]

        # Set group_name to healthy if no diagnosis
        prediction.loc[
            (prediction["group_name"] != "healthy") & (pd.isna(prediction["diagnosis"])),
            "group_name",
        ] = "healthy"

        # If no male count, substract count from female count columns
        ix_male_miss = (pd.isna(prediction["male_count"])) & ~(
            pd.isna(prediction["female_count"])
        )
        prediction.loc[ix_male_miss, "male_count"] = (
            prediction.loc[ix_male_miss, "count"]
            - prediction.loc[ix_male_miss, "female_count"]
        )

        # Same for female count
        ix_female_miss = (pd.isna(prediction["female_count"])) & ~(
            pd.isna(prediction["male_count"])
        )
        prediction.loc[ix_female_miss, "female_count"] = (
            prediction.loc[ix_female_miss, "count"]
            - prediction.loc[ix_female_miss, "male_count"]
        )

        return {"groups": prediction.to_dict(orient="records")}
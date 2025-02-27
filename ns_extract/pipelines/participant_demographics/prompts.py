base_message = """
You will be provided with a text sample from a scientific journal.
The sample is delimited with triple backticks.

Your task is to identify groups of participants that participated in the study, and underwent MRI.
If there is no mention of any participant groups, return a null array.

For each group identify:
    - the number of participants in each group, and the diagnosis.
    - the number of male participants, and their mean age, median age, minimum and maximum age
    - the number of female participants, and their mean age, median age, minimum and maximum age.
    - if this group of participants underwent MRI, fMRI or neuroimaging procedure.

Be as accurate as possible, and report information (especially diagnosis) using the technical terms (and abbreviations) used in the article.
If any of the information is missing, return `null` for that field.

Text sample: ${text}
"""

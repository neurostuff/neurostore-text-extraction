base_message = """
You will be provided with a text sample from a scientific journal.
The sample is delimited with triple backticks.

TASK OBJECTIVE:
Extract detailed participant demographic information with particular attention to
groups that underwent MRI procedures.
If there is no mention of any participant groups, return a null array.

EXTRACTION GUIDELINES:

1. GROUP IDENTIFICATION:
   - Identify ALL distinct participant groups (both patient and control groups)
   - Record group names/labels EXACTLY as used in the article
   - Note which groups underwent MRI/fMRI/neuroimaging procedures
   - Consider all subgroups (e.g., age-based, condition-based divisions)

2. DEMOGRAPHIC DETAILS (for each group):
   Count:
   - Report exact participant numbers as stated
   - Account for any excluded participants
   - Note if numbers are approximated/ranges

   Clinical Status:
   - Use EXACT diagnostic terms from the text
   - Include any disorder subtypes mentioned
   - Note any comorbid conditions
   - For control groups, note any specific health criteria

   Gender Distribution:
   - Record male/female counts as explicitly stated
   - Do not calculate/infer counts if not directly reported
   - Note if only percentages or ratios are given

   Age Information:
   - Record ALL age metrics provided (mean, median, range)
   - Preserve exact decimal places for reported values
   - Include age units if specified (years, months)
   - Note any age-specific subgroups

3. DATA QUALITY:
   - Return `null` for ANY unclear or missing information
   - Do not make assumptions about unreported demographics
   - Flag any inconsistent participant counts
   - Preserve original terminology and specificity

IMPORTANT REMINDERS:
- Extract information EXACTLY as stated in the text
- Use technical/medical terms verbatim from the source
- Do not infer or calculate missing values
- Return `null` for any information not explicitly provided

Text sample: ${text}

Return the extracted information in a structured format matching the specified schema,
ensuring each field contains only explicitly stated information from the text.
"""

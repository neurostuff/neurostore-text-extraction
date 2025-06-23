base_message = """
You will be provided with a text sample from a scientific journal.
The sample is delimited with triple backticks.

TASK OBJECTIVE:
Extract detailed pinformation about MRI contrasts performed in the study.
The extracted information should be structured according to the provided schema.
If any information specified in the schema is not mentioned in the text, 
return `null` for that field.
x
EXTRACTION GUIDELINES:

1. STUDY DOI:
   - SEARCH FOR THE STUDY DOI IN THE TEXT
   - Note that the contrast can be denoted in various ways, such as "contrast", "comparison" or "significance test", etc.
   - If no contrast is mentioned in the text, return `null` for the contrast field

1. CONTRAST IDENTIFICATION:
   - Look for the section or table that includees MRI contrasts
   - Note that the contrast can be denoted in various ways, such as "contrast", "comparison" or "significance test", etc.
   - If no contrast is mentioned in the text, return `null` for the contrast field

1. CONTRAST IDENTIFICATION:
   - Look for the section or table that includees MRI contrasts
   - Note that the contrast can be denoted in various ways, such as "contrast", "comparison" or "significance test", etc.
   - If no contrast is mentioned in the text, return `null` for the contrast field

2. TEST STATISTIC:
   - The test statistic can be denoted in various ways, such as "t-statistic", "z-score", "F-statistic", etc.
   - Report the test statistic exactly as stated, without inferring or calculating


3. SIGNIFICANCE LEVEL:
    - Significance level can be reported as "significant, "*", "**" or as an exact p-value (e.g., "p < 0.05")

    P-values:
   - If the exact p-value is mentioned, report it in the appropriate field (not as binary significance)
   - Report the significance level exactly as stated
   - If no significance level is mentioned, return `null` for the significance field

   Significance:
    - If the contrast is significant, report it as True in the significance field
    - If the contrast is not significant, report it as False in the significance field

4. ATLAS / PARCELLATION:
   - Report the atlas or parcellation used in the study
   - Report the number of regions in the atlas
   - Report all atlas-related information exactly as stated, without inferring anything

5. REGION OF INTEREST (ROI):
    - If a specific region of interest is mentioned, report it exactly as stated
    - Note that the region of interest can be denoted in various ways, such as "ROI", "region", "area", etc.
    - ROI can be a specific atlas label or a more general term
    - ROI can be denoted as an abbreviation, but not necessarily

6. COORDINATE SYSTEM:
    - Report the coordinate system used (e.g., Talairach, MNI, Native)
    - If no coordinate system is mentioned, return `null` for the coord_system field

7. BRAIN COORDINATES:
    - Extract brain coordinates in the format "x=34, y=-22, z=56".
    - Report each coordinated in its designated field (x, y or z)
    - Note that the coordinates can be mentioned in the main text or in a table
    - If no coordinates are mentioned, return `null` for the x, y, and z fields

IMPORTANT REMINDERS:
- Extract information EXACTLY as stated in the text
- Use technical/medical terms verbatim from the source
- Do not infer or calculate missing values
- Return `null` for any information not explicitly provided

Text sample: ${text}

Return the extracted information in a structured format matching the specified schema,
ensuring each field contains only explicitly stated information from the text.
"""

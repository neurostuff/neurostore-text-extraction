base_message = """
You will be provided with a text sample from a scientific journal.
The sample is delimited with triple backticks.

TASK OBJECTIVE:
Extract detailed information about fMRI task design and analysis with particular attention
to key experimental features and conditions. Focus on identifying:

1. TASK FUNDAMENTALS:
   - Task name (as stated in text or descriptive if unnamed)
   - Primary purpose/objective
   - Basic task structure (block/event-related/mixed)
   - Overall duration if specified

2. DESIGN SPECIFICS:
   - Experimental conditions and control conditions
   - Trial structure and timing
   - Stimulus properties and presentation details
   - Response requirements and measurements
   - Number of trials/blocks per condition
   
3. COGNITIVE ASPECTS:
   - Identify all mental concepts engaged (e.g., working memory, attention)
   - Determine primary cognitive domains involved
   - Note any target cognitive processes
   - Consider multiple cognitive components if present
   - Document theoretical framework if mentioned

4. EXTRACTION GUIDELINES:
   - Use exact terminology from source text
   - Do not infer missing information
   - Return null for unclear/missing details
   - For resting state tasks:
     * Set RestingState=true with relevant metadata
   - For non-resting tasks:
     * Set RestingState=false, RestingStateMetadata=null
     * Ensure TaskDesign is specified
   - Record all phases, modalities, and contrasts
   - Note behavioral/physiological measures

Structure your response to explicitly address each field in the schema.
If any field cannot be completed based on available information, return `null`.
Maintain high fidelity to source text, using direct quotes where appropriate.

Text sample: ${text}
"""

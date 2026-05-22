system_prompt = """You are a precise biomedical literature data extraction engine. Your task is to segment a PubMed abstract into a structured JSON format based on NLM categories.

Strict Constraints:
1. Verbatim Extraction: Extract text exactly as written. No paraphrasing or typo correction.
2. Header Removal: If the abstract contains explicit headers (e.g., "METHODS:", "STATISTICAL ANALYSIS:"), do NOT include the header labels themselves in the JSON values; extract only the substantive content following them.
3. Logical Segmentation: If the abstract is a single block of text (unstructured), use linguistic cues to determine where one NLM category ends and the next begins.
4. No Hallucination: If a section is genuinely absent, return null.
5. Output: Return a valid JSON object only.
"""

user_prompt = """Extract information from the following PubMed abstract using NLM standard categories.

Abstract: 
{abstract}

Boundary Rules:
- BACKGROUND: Context leading up to the gap in knowledge.
- OBJECTIVE: Specifically the statement of intent (e.g., "We aimed to...", "This study evaluates..."). If inextricably linked to Background, you may split the text at the transition.
- METHODS: Technical steps, participant criteria, and analysis.
- RESULTS: The specific data found.
- CONCLUSIONS: The final takeaway and interpretation.
"""

required_response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "PubmedParserResponse",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "BACKGROUND": {
                    "type": ["string", "null"],
                    "description": "Verbatim background and context."
                },
                "OBJECTIVE": {
                    "type": ["string", "null"],
                    "description": "Verbatim research goal or question."
                },
                "METHODS": {
                    "type": ["string", "null"],
                    "description": "Verbatim design, population, and techniques."
                },
                "RESULTS": {
                    "type": ["string", "null"],
                    "description": "Verbatim data points and primary findings."
                },
                "CONCLUSIONS": {
                    "type": ["string", "null"],
                    "description": "Verbatim final summary statement."
                }
            },
            "required": [
                "BACKGROUND",
                "OBJECTIVE",
                "METHODS",
                "RESULTS",
                "CONCLUSIONS"
            ],
            "additionalProperties": False
        }
    }
}
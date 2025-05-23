from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional

class TextIn(BaseModel):
    """Input text model for location extraction."""
    text: str = Field(
        ...,
        min_length=1,
        json_schema_extra={"example": "I will travel from London to Tokyo, passing through Paris and then to New York."}
    )

class LocationOut(BaseModel):
    """Output model for extracted locations."""
    input_text: str = Field(..., description="The original input text")
    extracted_locations: List[str] = Field(..., description="List of extracted location names")
    model_used: str = Field(..., description="Name of the model used for extraction")
    error_message: Optional[str] = Field(None, description="Error message if any")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "input_text": "We visited Berlin and Rome last summer.",
                    "extracted_locations": ["Berlin", "Rome"],
                    "model_used": "spaCy",
                    "error_message": None
                },
                {
                    "input_text": "Unknown model query.",
                    "extracted_locations": [],
                    "model_used": "N/A",
                    "error_message": "Model not found."
                }
            ]
        }
    )
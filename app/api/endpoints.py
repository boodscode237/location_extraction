from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from app.api.models import TextIn, LocationOut
from app.services.spacy_service import extract_locations_with_spacy
from app.services.bilstm_service import extract_locations_with_bilstm
from app.frontend.html import HTML_CONTENT 
from app.core.config import logger 

router = APIRouter()

@router.post("/extract-with-spacy/",
             response_model=LocationOut,
             tags=["Location Extraction"],
             summary="Extract locations using spaCy",
             description="Processes the input text with a spaCy NER model to identify and return geographical locations.")
async def extract_spacy_endpoint(data: TextIn):
    """
    Endpoint to extract locations using the **spaCy** model.
    - Processes input text.
    - Identifies entities like GPE (Geopolitical Entity), LOC (Location).
    - Returns a list of unique location names found.
    """
    user_sentence = data.text
    logger.info(f"Received request for spaCy extraction: '{user_sentence[:70]}...'")
    result = await extract_locations_with_spacy(user_sentence)

    if "error" in result:
        raise HTTPException(status_code=result.get("status_code", 500), detail=result["error"])

    return LocationOut(
        input_text=user_sentence,
        extracted_locations=result.get("locations", []),
        model_used=result.get("model_used", "spaCy")
    )

@router.post("/extract-with-bilstm/",
             response_model=LocationOut,
             tags=["Location Extraction"],
             summary="Extract locations using BiLSTM-CRF",
             description="Processes the input text with a BiLSTM-CRF model to identify and return geographical locations based on B-LOC and I-LOC tags.")
async def extract_bilstm_endpoint(data: TextIn):
    """
    Endpoint to extract locations using the **BiLSTM-CRF** model.
    - Tokenizes input text (using spaCy's tokenizer).
    - Converts tokens to IDs and feeds them to the BiLSTM-CRF model.
    - Decodes predicted tags to identify location spans (B-LOC, I-LOC).
    - Returns a list of unique location names found.
    """
    user_sentence = data.text
    logger.info(f"Received request for BiLSTM-CRF extraction: '{user_sentence[:70]}...'")
    result = await extract_locations_with_bilstm(user_sentence)

    if "error" in result:
        raise HTTPException(status_code=result.get("status_code", 500), detail=result["error"])

    return LocationOut(
        input_text=user_sentence,
        extracted_locations=result.get("locations", []),
        model_used=result.get("model_used", "BiLSTM-CRF")
    )

@router.get("/",
            response_class=HTMLResponse,
            tags=["Frontend"],
            summary="Serves the HTML frontend",
            description="Provides a simple web interface to interact with the location extraction API.")
async def get_frontend_form():
    """
    Serves the main HTML page for interacting with the API.
    """
    logger.info("Serving HTML frontend.")
    return HTMLResponse(content=HTML_CONTENT, status_code=200)

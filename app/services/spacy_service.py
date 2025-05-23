# app/services/spacy_service.py 
from typing import List, Dict, Any
from app.core.config import logger
from app.models.loaders import get_spacy_nlp

async def extract_locations_with_spacy(text: str) -> Dict[str, Any]:
    """
    Extracts locations using the loaded spaCy model.
    """
    spacy_nlp_instance = get_spacy_nlp()

    if spacy_nlp_instance is None:
        logger.warning("SpaCy model requested for extraction but not loaded.")
        return {"error": "SpaCy model is not available or not loaded.", "status_code": 503} 

    try:
        doc = spacy_nlp_instance(text)
        spacy_found_locs = [ent.text for ent in doc.ents if ent.label_.upper() in ["LOCATION", "LOC", "GPE"]]

    
        unique_locs = sorted(list(set(spacy_found_locs)), key=lambda loc: text.find(loc))

        logger.info(f"SpaCy extracted: {unique_locs} from text: '{text[:70]}...'")
        return {
            "locations": unique_locs,
            "model_used": "spaCy"
        }
    except Exception as e:
        logger.error(f"Error during spaCy model processing: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while processing with spaCy: {str(e)}", "status_code": 500}
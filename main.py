import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import API_TITLE, API_DESCRIPTION, API_VERSION, ALLOWED_ORIGINS, logger
from app.models.loaders import load_all_models
from app.api.endpoints import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    logger.info("--- FastAPI application starting up ---")
    load_all_models()
    logger.info("--- FastAPI application startup sequence finished ---")
    from app.models.loaders import get_spacy_nlp, get_bilstm_model
    if not get_spacy_nlp() and not get_bilstm_model():
        logger.critical("CRITICAL: NO MODELS WERE LOADED. API WILL NOT FUNCTION CORRECTLY.")
    elif not get_spacy_nlp():
        logger.warning("WARNING: SpaCy model failed to load. '/extract-with-spacy/' will not work.")
    elif not get_bilstm_model():
        logger.warning("WARNING: BiLSTM-CRF model failed to load. '/extract-with-bilstm/' will not work.")
    else:
        logger.info("All models loaded. API is ready.")
    
    yield  # Application runs here
    
    # Shutdown actions
    logger.info("--- FastAPI application shutting down ---")
    logger.info("--- FastAPI application shutdown sequence finished ---")

# Create FastAPI app instance with lifespan
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan
)

# Add CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="")

# Health Check Endpoint
@app.get("/health", tags=["Health Check"], summary="Check API health")
async def health_check():
    logger.info("Health check endpoint called.")
    return {"status": "ok", "message": "API is healthy"}

if __name__ == "__main__":
    logger.info("Starting Uvicorn server directly from main.py...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
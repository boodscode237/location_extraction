# Location Extractor API

This API extracts geographical locations (cities, countries, etc.) from a given text using two different models:
1.  A pre-trained spaCy Named Entity Recognition (NER) model.
2.  A custom-trained BiLSTM-CRF model using PyTorch.

The API provides endpoints for each model and a simple HTML frontend to interact with them.

## Project Structure
```markdown
location_extractor_api/
├── README.md
├── requirements.txt
├── .gitignore
├── main.py
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── endpoints.py
│   │   └── models.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bilstm.py
│   │   └── loaders.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── spacy_service.py
│   │   └── bilstm_service.py
│   └── frontend/
│       ├── __init__.py
│       └── html.py
├── data/
│   ├── BILSTM/
│   │   ├── ner_word2idx.pkl
│   │   ├── ner_tag2idx.pkl
│   │   └── best_bilstm_crf_location_ner_model_pytorch.pth
│   └── ner_model_spacy/
├── logs/
```
                      
## Setup

1.  **Create Directories**:
    * Manually create the `data/BILSTM/` and `data/ner_model_spacy/` directories.
    * Place your BiLSTM model files (`ner_word2idx.pkl`, `ner_tag2idx.pkl`, `best_bilstm_crf_location_ner_model_pytorch.pth`) into `data/BILSTM/`.
    * Place your trained spaCy model files into `data/ner_model_spacy/`.
    * Manually create the `logs/` directory if you intend to configure file-based logging.

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    You might also need to download the base spaCy model if your custom model depends on it (e.g., `python -m spacy download en_core_web_sm`).

3.  **Environment Variables (Optional but Recommended)**:
    Consider using a `.env` file for managing paths and other configurations, loading them with a library like `python-dotenv`.

## Running the Application

Navigate to the `location_extractor_api` directory in your terminal.

Run the FastAPI application using Uvicorn:
```bash
uvicorn main:app --reload
The --reload flag enables auto-reloading when code changes, which is useful for development.Accessing the ApplicationUI: Open your browser to http://127.0.0.1:8000/API Docs (Swagger UI): http://120.0.1:8000/docsAlternative API Docs (ReDoc): http://127.0.0.1:8000/redocModelsspaCyThe spaCy model is loaded from the ./data/ner_model_spacy/ directory. It identifies entities tagged as LOCATION, LOC, or GPE.BiLSTM-CRFThe BiLSTM-CRF model is a PyTorch-based sequence tagger.Model Architecture: Defined in app/models/bilstm.py.Weights and Mappings: Loaded from ./data/BILSTM/.ner_word2idx.pkl: Word to index mapping.ner_tag2idx.pkl: Tag to index mapping.best_bilstm_crf_location_ner_model_pytorch.pth: Model weights.Tokenization: Uses spaCy's tokenizer for consistency before feeding tokens to the BiLSTM model.LoggingThe application uses Python's built-in logging module. Logs are printed to the console by default.Key Files and Responsibilitiesmain.py: Initializes the FastAPI application, sets up CORS, includes routers, and defines the startup event for model loading.app/core/config.py: Stores paths, model hyperparameters, and other constants.app/models/loaders.py: Contains functions to load the spaCy and BiLSTM-CRF models and their associated files (mappings, etc.). It also holds the global model objects.app/models/bilstm.py: Defines the BiLSTM_CRF PyTorch nn.Module class.app/services/spacy_service.py: Houses the logic for extracting locations using the spaCy model.app/services/bilstm_service.py: Houses the logic for tokenizing input, processing with the BiLSTM-CRF model, and extracting location tags.app/api/endpoints.py: Defines the API routes (/extract-with-spacy/, /extract-with-bilstm/) and the root HTML frontend route.app/api/models.py: Contains Pydantic models for request (TextIn) and

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

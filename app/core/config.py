import os
import logging
import torch

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Base Directory ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data") 

# --- Model Configuration ---
SPACY_MODEL_PATH = os.path.join(DATA_DIR, "ner_model_spacy")
BILSTM_MODEL_DIR = os.path.join(DATA_DIR, "BILSTM")
WORD2IDX_PATH = os.path.join(BILSTM_MODEL_DIR, "ner_word2idx.pkl")
TAG2IDX_PATH = os.path.join(BILSTM_MODEL_DIR, "ner_tag2idx.pkl")
MODEL_WEIGHTS_PATH = os.path.join(BILSTM_MODEL_DIR, "best_bilstm_crf_location_ner_model_pytorch.pth")

BILSTM_EMBED_DIM = 150
BILSTM_LSTM_UNITS = 128
BILSTM_LAYERS = 2
BILSTM_DROPOUT = 0.35

BILSTM_MAX_SEQ_LEN = 100
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0 
UNK_IDX = 1 

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    logger.info("CUDA is available. Using GPU.")
else:
    DEVICE = torch.device("cpu")
    logger.info("CUDA not available. Using CPU.")

# --- API Information ---
API_TITLE = "Location Extractor API (spaCy & BiLSTM-CRF)"
API_DESCRIPTION = "Extracts locations using spaCy or a PyTorch BiLSTM-CRF model."
API_VERSION = "1.0.0" 

ALLOWED_ORIGINS = ["*"] 

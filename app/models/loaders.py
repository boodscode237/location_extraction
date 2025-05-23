import os
import pickle
import spacy
import torch

from app.core.config import (
    logger, DEVICE, SPACY_MODEL_PATH,
    BILSTM_MODEL_DIR, WORD2IDX_PATH, TAG2IDX_PATH, MODEL_WEIGHTS_PATH,
    BILSTM_EMBED_DIM, BILSTM_LSTM_UNITS, BILSTM_DROPOUT, BILSTM_LAYERS,
    PAD_TOKEN, UNK_TOKEN, PAD_IDX, UNK_IDX
)
from app.models.bilstm import BiLSTM_CRF

spacy_nlp = None
bilstm_crf_model = None
bilstm_word2idx = None
bilstm_tag2idx = None
bilstm_idx2tag = None

def load_all_models():
    """
    Loads the spaCy and BiLSTM-CRF models and their associated mappings.
    This function is intended to be called at application startup.
    """
    global spacy_nlp, bilstm_crf_model, bilstm_word2idx, bilstm_tag2idx, bilstm_idx2tag

    logger.info("--- Attempting to load spaCy model ---")
    try:
        if os.path.exists(SPACY_MODEL_PATH) and os.path.isdir(SPACY_MODEL_PATH):
            logger.info(f"Loading spaCy model from '{SPACY_MODEL_PATH}'...")
            spacy_nlp = spacy.load(SPACY_MODEL_PATH)
            logger.info("SpaCy model loaded successfully.")
        else:
            logger.warning(f"SpaCy model directory not found at '{SPACY_MODEL_PATH}'. SpaCy functionality will be disabled.")
            spacy_nlp = None
    except Exception as e_spacy_load:
        logger.error(f"CRITICAL ERROR loading spaCy model: {e_spacy_load}", exc_info=True)
        spacy_nlp = None

    logger.info("--- Attempting to load BiLSTM-CRF model ---")
    try:
        required_files = [WORD2IDX_PATH, TAG2IDX_PATH, MODEL_WEIGHTS_PATH]
        if not all(os.path.exists(f) for f in required_files):
            missing = [f for f in required_files if not os.path.exists(f)]
            raise FileNotFoundError(f"Missing BiLSTM model files: {missing}. Searched in '{BILSTM_MODEL_DIR}'")

        logger.info(f"Loading word2idx from {WORD2IDX_PATH}")
        with open(WORD2IDX_PATH, 'rb') as f:
            bilstm_word2idx = pickle.load(f)
        logger.info(f"Loading tag2idx from {TAG2IDX_PATH}")
        with open(TAG2IDX_PATH, 'rb') as f:
            bilstm_tag2idx = pickle.load(f)

        if PAD_TOKEN not in bilstm_word2idx or bilstm_word2idx[PAD_TOKEN] != PAD_IDX:
            logger.warning(
                f"'{PAD_TOKEN}' not found at index {PAD_IDX} in word2idx or index mismatch. "
                f"Current: {bilstm_word2idx.get(PAD_TOKEN)}. Forcing {PAD_TOKEN} to {PAD_IDX}."
            )
            bilstm_word2idx[PAD_TOKEN] = PAD_IDX
        
        if UNK_TOKEN not in bilstm_word2idx or bilstm_word2idx[UNK_TOKEN] != UNK_IDX:
            logger.warning(
                f"'{UNK_TOKEN}' not found at index {UNK_IDX} in word2idx or index mismatch. "
                f"Current: {bilstm_word2idx.get(UNK_TOKEN)}. Forcing {UNK_TOKEN} to {UNK_IDX}."
            )
            current_token_at_unk_idx = next((token for token, idx in bilstm_word2idx.items() if idx == UNK_IDX and token != UNK_TOKEN), None)
            if current_token_at_unk_idx:
                 logger.error(f"Index {UNK_IDX} intended for '{UNK_TOKEN}' is already taken by '{current_token_at_unk_idx}'. This is a critical issue.")
            bilstm_word2idx[UNK_TOKEN] = UNK_IDX

        bilstm_idx2tag = {idx: tag for tag, idx in bilstm_tag2idx.items()}
        logger.info(f"Created idx2tag mapping. Found {len(bilstm_idx2tag)} tags.")

        vocab_size = len(bilstm_word2idx)
        num_tags = len(bilstm_tag2idx)
        logger.info(f"BiLSTM Params: vocab_size={vocab_size}, num_tags={num_tags}")

        bilstm_crf_model = BiLSTM_CRF(
            vocab_size=vocab_size,
            embed_dim=BILSTM_EMBED_DIM,
            lstm_units=BILSTM_LSTM_UNITS,
            num_tags=num_tags,
            dropout_rate=BILSTM_DROPOUT,
            num_bilstm_layers=BILSTM_LAYERS,
            padding_idx=bilstm_word2idx.get(PAD_TOKEN, PAD_IDX) 
        )

        logger.info(f"Loading model weights from {MODEL_WEIGHTS_PATH}")
        bilstm_crf_model.to(DEVICE)
        bilstm_crf_model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
        bilstm_crf_model.eval()

        logger.info("BiLSTM-CRF model loaded successfully and moved to device: %s.", DEVICE)

    except FileNotFoundError as e_bilstm_file:
        logger.error(f"ERROR loading BiLSTM model (file not found): {e_bilstm_file}")
        bilstm_crf_model = None
    except pickle.UnpicklingError as e_pickle:
        logger.error(f"ERROR loading BiLSTM mappings (pickle error): {e_pickle}", exc_info=True)
        bilstm_crf_model = None
    except RuntimeError as e_runtime:
        logger.error(f"CRITICAL RUNTIME ERROR loading BiLSTM-CRF model state_dict: {e_runtime}", exc_info=True)
        logger.error(
            "This often indicates a mismatch between the saved model's architecture "
            "(hyperparameters like embed_dim, lstm_units, num_layers, num_tags) "
            "and the current model definition. Please check app/core/config.py "
            "BILSTM_* parameters."
        )
        bilstm_crf_model = None
    except Exception as e_bilstm_load:
        logger.error(f"CRITICAL ERROR loading BiLSTM-CRF model: {e_bilstm_load}", exc_info=True)
        bilstm_crf_model = None

    if not spacy_nlp and not bilstm_crf_model:
        logger.warning("WARNING: NO MODELS WERE LOADED SUCCESSFULLY.")
    elif not spacy_nlp:
        logger.warning("WARNING: SpaCy model failed to load.")
    elif not bilstm_crf_model:
        logger.warning("WARNING: BiLSTM-CRF model failed to load.")
    else:
        logger.info("Model loading sequence finished. Both spaCy and BiLSTM-CRF models appear ready.")

# Functions to safely access loaded models/mappings
def get_spacy_nlp():
    """Returns the loaded spaCy NLP object."""
    return spacy_nlp

def get_bilstm_model():
    """Returns the loaded BiLSTM-CRF model object."""
    return bilstm_crf_model

def get_bilstm_word2idx():
    """Returns the word_to_index mapping for the BiLSTM model."""
    return bilstm_word2idx

def get_bilstm_idx2tag():
    """Returns the index_to_tag mapping for the BiLSTM model."""
    return bilstm_idx2tag

def get_bilstm_tag2idx():
    """Returns the tag_to_index mapping for the BiLSTM model."""
    return bilstm_tag2idx

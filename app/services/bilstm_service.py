import torch
from typing import List, Dict, Any

from app.core.config import logger, DEVICE, BILSTM_MAX_SEQ_LEN, PAD_TOKEN, UNK_TOKEN, PAD_IDX, UNK_IDX
from app.models.loaders import get_bilstm_model, get_bilstm_word2idx, get_bilstm_idx2tag, get_spacy_nlp

async def extract_locations_with_bilstm(text: str) -> Dict[str, Any]:
    """
    Extracts locations using the loaded BiLSTM-CRF model.
    Uses spaCy for initial tokenization.
    """
    bilstm_model = get_bilstm_model()
    word2idx = get_bilstm_word2idx()
    idx2tag = get_bilstm_idx2tag()
    spacy_tokenizer = get_spacy_nlp() 

    if bilstm_model is None or word2idx is None or idx2tag is None:
        logger.warning("BiLSTM-CRF model, word2idx, or idx2tag requested but not loaded.")
        return {"error": "BiLSTM-CRF model or its mappings are not available.", "status_code": 503}

    if spacy_tokenizer is None:
        logger.warning("SpaCy tokenizer (needed for BiLSTM preprocessing) is not loaded.")
        return {"error": "SpaCy tokenizer is not available, which is required for BiLSTM.", "status_code": 503}

    try:
        # 1. Tokenize using spaCy's tokenizer for consistency
        # We only need the tokenizer part of spaCy here, not the full pipeline if it's heavy.
        # If spacy_nlp is a full pipeline, spacy_nlp.tokenizer(text) is more direct.
        # If it's loaded via spacy.load(), it includes the tokenizer.
        doc = spacy_tokenizer.tokenizer(text) # Use the tokenizer component
        tokens = [token.text for token in doc if token.text.strip()] # Ensure no empty string tokens

        if not tokens:
            logger.info("BiLSTM: Input text resulted in no tokens after spaCy tokenization.")
            return {"locations": [], "model_used": "BiLSTM-CRF"}

        # 2. Convert tokens to IDs, handling unknown words
        # Ensure UNK_TOKEN exists and has a valid index (UNK_IDX)
        unk_idx_to_use = word2idx.get(UNK_TOKEN, UNK_IDX) # Fallback to configured UNK_IDX if UNK_TOKEN not in map
        if UNK_TOKEN not in word2idx:
             logger.warning(f"'{UNK_TOKEN}' not in word2idx. Using default UNK_IDX: {unk_idx_to_use}")


        word_ids = [word2idx.get(token, unk_idx_to_use) for token in tokens]

        # 3. Pad sequence and create mask
        seq_len = len(word_ids)
        pad_idx_to_use = word2idx.get(PAD_TOKEN, PAD_IDX) # Fallback to configured PAD_IDX

        if seq_len < BILSTM_MAX_SEQ_LEN:
            padded_word_ids = word_ids + [pad_idx_to_use] * (BILSTM_MAX_SEQ_LEN - seq_len)
            mask = [1] * seq_len + [0] * (BILSTM_MAX_SEQ_LEN - seq_len)
        else:
            padded_word_ids = word_ids[:BILSTM_MAX_SEQ_LEN]
            mask = [1] * BILSTM_MAX_SEQ_LEN
            logger.warning(f"Input text truncated to {BILSTM_MAX_SEQ_LEN} tokens for BiLSTM: '{' '.join(tokens[:BILSTM_MAX_SEQ_LEN])}'")
            tokens = tokens[:BILSTM_MAX_SEQ_LEN] # Also truncate original tokens list to match

        # 4. Convert to Tensors and move to device
        input_tensor = torch.tensor([padded_word_ids], dtype=torch.long).to(DEVICE)
        mask_tensor = torch.tensor([mask], dtype=torch.bool).to(DEVICE) # CRF expects bool mask

        # 5. Perform Inference
        with torch.no_grad():
            predicted_tag_ids_batch = bilstm_model.decode(input_tensor, mask_tensor)

        if not predicted_tag_ids_batch: # Should not happen if decode is successful
            logger.error("BiLSTM model decode returned an empty list.")
            return {"error": "BiLSTM model decoding failed.", "status_code": 500}

        predicted_tag_ids = predicted_tag_ids_batch[0] # Get first (and only) item for batch size 1

        # Ensure predicted_tag_ids matches the length of original (potentially truncated) tokens
        # The CRF decode should return a list of tags for each item in the batch,
        # and each list of tags should correspond to the unmasked part of the sequence.
        # However, some implementations might return tags for the full MAX_SEQ_LEN.
        # We should only consider tags up to the original (or truncated) sequence length.
        
        # 6. Convert tag IDs back to tag names
        # Only convert tags for the actual tokens, not padding
        actual_predicted_tags = [idx2tag.get(tag_id, 'O') for tag_id in predicted_tag_ids[:len(tokens)]]

        # 7. Extract location spans (B-LOC, I-LOC scheme)
        locations = []
        current_location_tokens = []
        for i, token_text in enumerate(tokens): # Iterate over original (or truncated) tokens
            if i >= len(actual_predicted_tags): # Safety break if tags are shorter than tokens
                break
            tag = actual_predicted_tags[i]

            if tag == 'B-LOC':
                if current_location_tokens: # Finalize previous location
                    locations.append(" ".join(current_location_tokens))
                current_location_tokens = [token_text]
            elif tag == 'I-LOC':
                if current_location_tokens: # Continue current location
                    current_location_tokens.append(token_text)
                else: 
                    current_location_tokens = [token_text]
            else: # 'O' tag or other tags
                if current_location_tokens: # Finalize current location
                    locations.append(" ".join(current_location_tokens))
                    current_location_tokens = []

        if current_location_tokens: # Add any trailing location
            locations.append(" ".join(current_location_tokens))

        # Remove duplicates and sort by appearance
        unique_locs = sorted(list(set(locations)), key=lambda loc: text.find(loc))

        logger.info(f"BiLSTM extracted: {unique_locs} from text: '{text[:70]}...'")
        return {
            "locations": unique_locs,
            "model_used": "BiLSTM-CRF"
        }

    except Exception as e:
        logger.error(f"Error during BiLSTM-CRF model processing: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred while processing with BiLSTM-CRF: {str(e)}", "status_code": 500}

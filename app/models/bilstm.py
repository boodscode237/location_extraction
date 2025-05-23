import torch
import torch.nn as nn
from torchcrf import CRF
from app.core.config import logger 

class BiLSTM_CRF(nn.Module):
    """
    BiLSTM-CRF model class for sequence tagging.
    """
    def __init__(self, vocab_size, embed_dim,
                 lstm_units, num_tags,
                 dropout_rate=0.3, num_bilstm_layers=1, padding_idx=0):
        super().__init__()
        logger.info(
            f"Initializing BiLSTM_CRF: vocab_size={vocab_size}, embed_dim={embed_dim}, "
            f"lstm_units={lstm_units}, num_tags={num_tags}, dropout={dropout_rate}, "
            f"layers={num_bilstm_layers}, pad_idx={padding_idx}"
        )

        # Embedding layer - maps word indices to embeddings
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(embed_dim,
                           lstm_units,
                           num_layers=num_bilstm_layers,
                           bidirectional=True,
                           batch_first=True, # Model expects batch_first=True
                           dropout=dropout_rate if num_bilstm_layers > 1 else 0)

        # Linear layer to map BiLSTM output to tag space
        self.hidden2tag = nn.Linear(lstm_units * 2, num_tags) # lstm_units * 2 for bidirectional

        # CRF layer - expects batch_first=False by default for torchcrf
        # We will handle the permutation in forward/decode/compute_loss
        self.crf = CRF(num_tags, batch_first=False)

    def _to_seq_first(self, x: torch.Tensor, is_mask: bool = False) -> torch.Tensor:
        """
        Helper to convert from batch-first (B, L, C) or (B, L) to sequence-first (L, B, C) or (L, B).
        CRF layer expects sequence-first.
        """
        if x.dim() == 3:
            return x.permute(1, 0, 2)
        elif x.dim() == 2: 
            return x.permute(1, 0)
        else:
            logger.warning(f"Unexpected tensor dimension in _to_seq_first: {x.dim()}")
            return x


    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get emissions (logits before CRF).
        Input word_ids shape: (batch_size, seq_len)
        Output emissions shape: (batch_size, seq_len, num_tags)
        """
        embed = self.embedding(word_ids)  
        lstm_out, _ = self.lstm(embed)    
        emissions = self.hidden2tag(lstm_out) 
        return emissions

    def compute_loss(self, word_ids: torch.Tensor, tag_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute loss using CRF negative log likelihood.
        word_ids shape: (batch_size, seq_len)
        tag_ids shape: (batch_size, seq_len)
        mask shape: (batch_size, seq_len), boolean or byte tensor
        """
        emissions = self.forward(word_ids) 

        emissions_seq_first = self._to_seq_first(emissions)
        tag_ids_seq_first = self._to_seq_first(tag_ids)
        mask_seq_first = self._to_seq_first(mask, is_mask=True).bool() 

        # CRF computes negative log likelihood, so we return it directly
    
        loss = -self.crf(
            emissions_seq_first,
            tag_ids_seq_first,
            mask=mask_seq_first,
            reduction='mean'
        )
        return loss

    def decode(self, word_ids: torch.Tensor, mask: torch.Tensor) -> list:
        """
        Viterbi decoding to find the best tag sequence.
        word_ids shape: (batch_size, seq_len)
        mask shape: (batch_size, seq_len), boolean or byte tensor
        Returns a list of lists, where each inner list contains the predicted tag IDs for a sequence.
        """
        emissions = self.forward(word_ids) # (B, L, num_tags)

        # Permute for CRF layer
        emissions_seq_first = self._to_seq_first(emissions)
        mask_seq_first = self._to_seq_first(mask, is_mask=True).bool() # Ensure mask is boolean

        # CRF decode returns a list of lists of tag indices
        return self.crf.decode(
            emissions_seq_first,
            mask=mask_seq_first
        )

# src/models/text_encoder.py
import torch
import torch.nn as nn
from transformers import AutoModel
from .layout_embedding import AbsLayoutEmbedding

class TextEncoder(nn.Module):
    """
    This wrapper around a HuggingFace encoder (e.g., DeBERTa-v3-base).

    Responsibilities:
    - Load the HF model
    - Handle freezing/unfreezing
    - provide clean forward() API that only returns hidden states.
    - Expose model dimensions (d_model) so fusuon blocks can adapt automatically (given differing HF model types)
    - Add positional 2D encodings (DeBerta handles 1D positional)
     """
    def __init__(
            self, 
            model_name: str = "microsoft/deberta-v3-base",
            freeze: bool = True,
            return_hidden_states: bool = False,
    ):
        super().__init__()
        # Load HF encoder
        self.model = AutoModel.from_pretrained(
            model_name, 
            output_hidden_states=return_hidden_states, 
        )


        # Hidden dim
        self.d_model = self.model.config.hidden_size

        # Intitialize layout embedding module
        self.layout_encoder = AbsLayoutEmbedding(d_model=self.d_model)

        self.layout_gate = nn.Parameter(torch.tensor(0.1))

        self.norm = nn.LayerNorm(self.d_model)
        # Whether to output all layers or just the last layer
        self.return_hidden_states = return_hidden_states

        if freeze:
            self.freeze_all()

    # Freezing Utilities
    def freeze_all(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def unfreeze_last_k_layers(self, k: int):
        """
        Unfreeze the last k transformer layers. 
        Useful during fine-tuning.
        """
        # DeBERTa stores layers in self.model.encoder.layer
        for layer in self.model.encoder.layer[-k:]:
            for p in layer.parameters():
                p.requires_grad = True
    
    # Forward

    def forward(self, input_ids, attention_mask, bboxes):
        """
        input_ids:      (B, T)
        attention_mask: (B, T)
        bboxes: (B, T, 4) float in [0,1] (normalize in dataset)

        Returns:
            last_hidden_state: (B, T, d_model)
            (optionally) a tuple of all hidden states
        """
        outputs = self.model(input_ids, attention_mask = attention_mask)

        H_texts = outputs.last_hidden_state

        # Should be handled by DataLoader
        bboxes = bboxes.to(H_texts.device)

        # layout_emb: (B, T, d_model)
        layout_emb = self.layout_encoder(bboxes)

        # broadcast mask 
        layout_emb = layout_emb * attention_mask.unsqueeze(-1).float()

        # Combine (addition)
        H_text_layout = H_texts + self.layout_gate * layout_emb
        H_text_layout = self.norm(H_text_layout)

        # HF returns model output
        if self.return_hidden_states:
            return H_text_layout, outputs.hidden_states
        else:   
            return H_text_layout
        
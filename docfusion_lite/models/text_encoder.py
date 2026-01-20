# src/models/text_encoder.py
import torch
import torch.nn as nn
from transformers import AutoModel

class TextEncoder(nn.Module):
    """
    This wrapper around a HuggingFace encoder (e.g., LayoutLMv1).

    Responsibilities:
    - Load the HF model
    - Handle freezing/unfreezing
    - provide clean forward() API that only returns hidden states.
    - Expose model dimensions (d_model) so fusuon blocks can adapt automatically (given differing HF model types)
    - Add positional 2D encodings (LayoutLMv1 handles 2D bbox positional internally)
     """
    def __init__(
            self,
            model_name: str = "SCUT-DLVCLab/lilt-roberta-en-base",
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

        # Optional normalization (not part of vanilla LayoutLMv1). Keep/remove consistently.
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
        for layer in self.model.encoder.layer[-k:]:
            for p in layer.parameters():
                p.requires_grad = True

    # Forward

    def forward(self, input_ids, attention_mask, bboxes, token_type_ids=None):
        """
        input_ids:      (B, T)
        attention_mask: (B, T)
        bboxes: (B, T, 4) int in [0,1000] (normalize in dataset)

        Returns:
            last_hidden_state: (B, T, d_model)
            (optionally) a tuple of all hidden states
        """
        # Should be handled by DataLoader
        bboxes = bboxes.to(input_ids.device)

        # LayoutLMv1 expects bbox as torch.long
        if bboxes.dtype != torch.long:
            bboxes = bboxes.to(torch.long)

        bboxes = bboxes.clamp(0, 1000)

        model_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bboxes,
        )
        if token_type_ids is not None:
            model_kwargs["token_type_ids"] = token_type_ids

        outputs = self.model(**model_kwargs)

        H_texts = outputs.last_hidden_state

        H_text_layout = H_texts
        #H_text_layout = self.norm(H_text_layout)

        # HF returns model output
        if self.return_hidden_states:
            return H_text_layout, outputs.hidden_states
        else:
            return H_text_layout

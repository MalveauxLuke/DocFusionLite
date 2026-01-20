# src/models/docfusion_model.py

import torch
import torch.nn as nn
from typing import Optional

from .text_encoder import TextEncoder
from .vision_encoder import VisionEncoder
from .region_pooling import RegionPooler
from .fusion import FusionStem, FusionLayer
from .gating import DocGate

class DocFusionModel(nn.Module):
    """
    High-level document model.

    Modes:
      - "3.1": one FusionStem, then text-only transformer (no FusionLayer).
      - "3.3": FusionStem (optional) + multiple FusionLayer (deep multimodal),
               then optional text-only transformer.

    This class just wires modules; it doesn't implement new math.
    """

    def __init__(
        self,
        mode: str = "3.1",               # "3.1" or "3.3"
        d_model: int = 768,
        d_region: int = 256,
        d_patch: int = 768,
        n_heads: int = 8,
        num_fusion_layers: int = 0,      # only used in 3.3
        num_encoder_layers: int = 2,
        num_labels: int = 16,
        dropout: float = 0.1,
        use_gate_stem: bool = False,   
        use_gate_fusion_layer: bool = False,  # only used in 3.3
        use_region_ffn: bool = True,    # only used in 3.3
        use_doc_gate: bool = True,
        text_model_name: str = "SCUT-DLVCLab/lilt-roberta-en-base",
        vision_model_name: str = "microsoft/dit-base",
    ):
        super().__init__()
        assert mode in ("3.1", "3.3")
        self.mode = mode
        self.use_doc_gate = use_doc_gate

        # 1. Encoders
        self.text_encoder = TextEncoder(
            model_name=text_model_name, 
            freeze=True,)
        
        self.vision_encoder = VisionEncoder(
            model_name=vision_model_name,
            freeze=True,
        )
        # Define model dimensions 
        d_patch = self.vision_encoder.d_patch # vision encoder embedding dim
        d_model = self.text_encoder.d_model # text encoder embedding dim
        # 2. RegionPooler 
        self.region_pooler = RegionPooler(
            d_patch=d_patch, 
            d_region=d_region,)
        self.region_proj = nn.Linear(d_patch, d_model)
        # 3. FusionStem
        self.fusion_stem = FusionStem(
            d_model=d_model,
            dropout=dropout,
            use_gate=use_gate_stem,
        )

        self.doc_gate = DocGate(d_model=d_model, dropout=dropout) if use_doc_gate else None

        # 4. Deep fusion layers

        if self.mode == "3.3":
            self.fusion_layers = nn.ModuleList([
                FusionLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    use_region_ffn=use_region_ffn,
                    use_gate=use_gate_fusion_layer,
                )
                for _ in range(num_fusion_layers)
            ])
        else:
            self.fusion_layers = nn.ModuleList()
        
        # 5. Post-fusion-text-only transformer encoder (not 3.3)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.post_fusion_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
        )

        # 6. Task head (token classification example)
        self.token_head = nn.Linear(d_model, num_labels)

    def forward(self, batch) -> torch.Tensor:
        """
        batch should provide at least:
          - batch.input_ids:      (B,T)
          - batch.attention_mask: (B,T)
          - batch.images:         (B,3,H,W)
          - batch.token_boxes:    (B,T,4)
        """
        # 1. Text encoder
        h_text = self.text_encoder(
            input_ids = batch.input_ids,
            attention_mask = batch.attention_mask,
            bboxes = batch.token_boxes,
        ) # (B, T, d_model)

        g_doc: Optional[torch.Tensor] = None
        if self.doc_gate is not None:
            g_doc = self.doc_gate(h_text, batch.attention_mask)

        # 2. Vision encoder
        patch_feats, patch_boxes = self.vision_encoder(
            pixel_values = batch.images,
        ) # patch_feats: (B, P, d_patch), pach_boxes: (B, P, 4)
        # 3. Region Pooling
        region_feats, region_mask = self.region_pooler(
            patch_feats=patch_feats,
            token_boxes=batch.token_boxes,
            patch_boxes=patch_boxes,
            token_mask=batch.attention_mask,
        ) # region_feats: (B, T, d_patch) region_mask: (B, T)
        region_proj  = self.region_proj(region_feats) # (B, T, d_patch)->(B,T,d_model)

        #patch_feats = None
        #region_proj = torch.zeros_like(region_proj)

        # g_doc = None
        # 4. Initital fusion (fusion stem)
        h = self.fusion_stem(
            h_text=h_text,
            region_proj=region_proj,
            g_doc=g_doc,
        )# h: (B,T,d_model)
        # 5. deep Fusion (3.3)
        if self.mode == "3.3" and len(self.fusion_layers) > 0:
            # If you need patch masks, compute them here
            patch_mask: Optional[torch.Tensor] = None

            for layer in self.fusion_layers:
                h = layer(
                    h=h,
                    region_proj=region_proj,
                    patch_feats=patch_feats,
                    g_doc=g_doc,
                    text_mask=batch.attention_mask,
                    patch_mask=patch_mask,
                )
        # 6. Post fusion text-only encoder
        # nn.TransformerEncoder expects src_key_padding_mask with True for PAD
        src_key_padding_mask = (batch.attention_mask == 0) if batch.attention_mask is not None else None
        h = self.post_fusion_encoder(
            h,
            src_key_padding_mask=src_key_padding_mask,
        )  # (B,T,d_model)

        # 7. Task head
        logits = self.token_head(h)  # (B,T,num_labels)
        return logits
    



    
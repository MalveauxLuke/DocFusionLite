import torch
import torch.nn as nn
import torch.nn.functional as F


class AbsLayoutEmbedding(nn.Module):
    """
    Absolute 2D layout embeddings from normalized bboxes.

    How it works:
    1. We take each coordinate and bucket it into a specified number of buckets. 
    2. We learn embeddings for each bucket
    3. We use our coordinate bucket embeddings and project using a small MLP so we fit the dimensions of the output of our text backbone

    Inputs:
        bboxes: (B, T, 4) float in [0, 1], columns = (x0, y0, x1, y1)

    Outputs:
        layout_emb: (B, T, d_model) to be ADDED to text hidden states.
    """
    def __init__(
        self,
        num_buckets: int = 128,
        coord_emb_dim: int = 32,   # per-coordinate dim
        d_layout: int = 128,       # internal layout dim
        d_model: int = 768,        # same as text encoder hidden size
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_buckets = num_buckets

        # Embedding table for all coords (simplest)
        self.coord_embed = nn.Embedding(num_buckets, coord_emb_dim)

        in_dim = 4 * coord_emb_dim  # x0,y0,x1,y1 concatenated

        # Small MLP: 4*coord_dim -> d_layout -> d_model
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_layout),
            nn.GELU(),
            nn.Linear(d_layout, d_model),
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # learned global rate on layout input to text backbone
        # Prevents random layout noise from corrupint good pretrained model
        # Allows use to see how much the model "cares" about layout.
        # self.layout_scale = nn.Parameter(torch.tensor(0.0))


    def _bucketize(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords: (B, T, 4) float in [0, 1]
        returns: (B, T, 4) long in [0, num_buckets-1]
        """

        # Convert [0,1] -> [0, num_buckets-1]
        buckets = (coords * (self.num_buckets - 1)).round()
        buckets = buckets.clamp(0, self.num_buckets - 1).long()
        return buckets
    
    def forward(self, bboxes: torch.Tensor) -> torch.Tensor:
        """
        bboxes: (B, T, 4) float in [0,1]
        """
        assert bboxes.dim() == 3 and bboxes.size(-1) == 4, \
            f"Expected (B,T,4) bboxes, got {tuple(bboxes.shape)}"

        B, T, _ = bboxes.shape

        # (B, T, 4) -> (B, T, 4) long
        bucket_ids = self._bucketize(bboxes)

        # Embed each coord separately: (B, T, 4, coord_emb_dim)
        coord_embs = self.coord_embed(bucket_ids)

        # Concatenate coords: (B, T, 4*coord_emb_dim)
        coord_embs = coord_embs.view(B, T, -1)

        # MLP -> (B, T, d_model)
        layout_emb = self.mlp(coord_embs)

        # norm to ensure we dont give our transformer an unexpectadly large positional embedding.
        layout_emb = self.norm(layout_emb)
        layout_emb = self.dropout(layout_emb)

        return layout_emb
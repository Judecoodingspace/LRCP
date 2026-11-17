import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    """
    Local compatibility implementation of MMDetection 2.x
    LearnedPositionalEncoding.

    Parameter names and tensor shapes are kept compatible with the
    original implementation:
      row_embed.weight
      col_embed.weight
    """

    def __init__(
        self,
        num_feats,
        row_num_embed=50,
        col_num_embed=50,
        **kwargs
    ):
        super().__init__()

        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)

        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        h, w = mask.shape[-2:]

        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)

        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)

        pos = torch.cat(
            (
                x_embed.unsqueeze(0).repeat(h, 1, 1),
                y_embed.unsqueeze(1).repeat(1, w, 1),
            ),
            dim=-1,
        )

        pos = (
            pos.permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(mask.shape[0], 1, 1, 1)
        )

        return pos

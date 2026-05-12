import torch
import torch.nn as nn

from ..builder import MODELS


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, source, query, attn_masks=None, query_pe=None):
        outputs = []
        for i in range(len(source)):
            q_pos = query_pe[i] if query_pe is not None else None
            q = self.with_pos_embed(query[i], q_pos)
            attn_mask = attn_masks[i] if attn_masks is not None else None
            output, _ = self.attn(q, source[i], source[i], attn_mask=attn_mask)
            output = self.dropout(output) + query[i]
            output = self.norm(output)
            outputs.append(output)
        return outputs


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, pe=None):
        outputs = []
        for i in range(len(x)):
            q = k = self.with_pos_embed(x[i], pe[i] if pe is not None else None)
            output, _ = self.attn(q, k, x[i])
            output = self.dropout(output) + x[i]
            output = self.norm(output)
            outputs.append(output)
        return outputs


class FFN(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.0, activation_fn="relu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU() if activation_fn == "relu" else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        outputs = []
        for i in range(len(x)):
            output = self.net(x[i])
            output = output + x[i]
            output = self.norm(output)
            outputs.append(output)
        return outputs


@MODELS.register_module("SPFormerDecoder")
class SPFormerDecoder(nn.Module):
    def __init__(
        self,
        num_layer=6,
        num_query=100,
        num_class=18,
        in_channel=32,
        d_model=256,
        nhead=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn="relu",
        iter_pred=True,
        attn_mask=False,
        use_query_pos=False,
        use_score=False,
        use_param_query=True,
    ):
        super().__init__()
        self.use_score = use_score
        self.num_layer = num_layer
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask

        self.input_proj = nn.Sequential(
            nn.Linear(in_channel, d_model), nn.LayerNorm(d_model), nn.ReLU()
        )
        self.query = nn.Embedding(num_query, d_model)
        if use_query_pos:
            self.pe = nn.Embedding(num_query, d_model)

        self.cross_attn_layers = nn.ModuleList(
            [CrossAttentionLayer(d_model, nhead, dropout) for _ in range(num_layer)]
        )
        self.self_attn_layers = nn.ModuleList(
            [SelfAttentionLayer(d_model, nhead, dropout) for _ in range(num_layer)]
        )
        self.ffn_layers = nn.ModuleList(
            [FFN(d_model, hidden_dim, dropout, activation_fn) for _ in range(num_layer)]
        )

        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, num_class + 1)
        )
        if self.use_score:
            self.out_score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
            )
        self.x_mask = nn.Sequential(
            nn.Linear(in_channel, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )

    def _forward_head(self, query, mask_feats):
        pred_labels = []
        pred_masks = []
        attn_masks = []
        pred_scores = [] if self.use_score else None
        for i in range(len(query)):
            norm_query = self.out_norm(query[i])
            pred_labels.append(self.out_cls(norm_query))
            if self.use_score:
                pred_scores.append(self.out_score(norm_query))
            pred_mask = torch.einsum("nd,md->nm", norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_masks.append(attn_mask.detach())
            pred_masks.append(pred_mask)
        return pred_labels, pred_scores, pred_masks, attn_masks if self.attn_mask else None

    def _get_query(self, batch_size):
        pe = (
            self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)
            if getattr(self, "pe", None) is not None
            else None
        )
        query = self.query.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        return query, pe

    def forward(self, input_dict):
        sp_feat = input_dict["sp_feat"]
        batch_size = len(sp_feat)

        inst_feats = [self.input_proj(x) for x in sp_feat]
        mask_feats = [self.x_mask(x) for x in sp_feat]
        query, pe = self._get_query(batch_size)

        pred_labels, pred_masks, pred_scores = [], [], []
        pred_label, pred_score, pred_mask, attn_mask = self._forward_head(
            query, mask_feats
        )
        pred_labels.append(pred_label)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)

        for i in range(self.num_layer):
            query = self.cross_attn_layers[i](
                inst_feats, query, attn_mask, query_pe=pe
            )
            query = self.self_attn_layers[i](query, pe)
            query = self.ffn_layers[i](query)
            pred_label, pred_score, pred_mask, attn_mask = self._forward_head(
                query, mask_feats
            )
            pred_labels.append(pred_label)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        if not self.iter_pred:
            return {"labels": pred_label, "masks": pred_mask, "scores": pred_score}

        return {
            "labels": pred_label,
            "masks": pred_mask,
            "scores": pred_score,
            "aux_outputs": [
                {"labels": a, "masks": b, "scores": c}
                for a, b, c in zip(pred_labels[:-1], pred_masks[:-1], pred_scores[:-1])
            ],
        }

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from ..builder import MODELS, build_model
from ..utils.structure import Point
from .misc import process_instance, process_label, split_offset
from .nms import mask_matrix_nms


@MODELS.register_module("SPFormer-v1m1")
class SPFormer(nn.Module):
    def __init__(
        self,
        backbone,
        decoder,
        criterion=None,
        semantic_num_classes=18,
        semantic_ignore_index=-1,
        segment_ignore_index=(-1, 0, 2),
        instance_ignore_index=-1,
        topk_insts=200,
        score_thr=0.0,
        npoint_thr=100,
        sp_score_thr=0.55,
        nms=True,
    ):
        super().__init__()

        self.semantic_num_classes = semantic_num_classes
        self.semantic_ignore_index = semantic_ignore_index
        self.segment_ignore_index = tuple(segment_ignore_index)
        self.instance_ignore_index = instance_ignore_index

        ignored = sorted([i for i in self.segment_ignore_index if i >= 0])
        total = self.semantic_num_classes + len(ignored)
        kept = [i for i in range(total) if i not in ignored]
        self.register_buffer(
            "class_map", torch.tensor(kept, dtype=torch.long), persistent=False
        )

        self.backbone = build_model(backbone)
        self.decoder = build_model(decoder)
        self.criterion = build_model(criterion) if criterion is not None else None

        self.topk_insts = topk_insts
        self.score_thr = score_thr
        self.npoint_thr = npoint_thr
        self.sp_score_thr = sp_score_thr
        self.nms = nms

    def forward(self, input_dict):
        vx_offset = input_dict["offset"].int()
        pt_offset = input_dict["origin_offset"].int()

        feats = self.backbone(Point(input_dict))
        input_dict["feats"] = feats

        inv = split_offset(input_dict["inverse"], pt_offset)
        sp_raw = split_offset(input_dict["superpoint"], pt_offset)
        sp = [torch.unique(_sp, return_inverse=True)[1] for _sp in sp_raw]

        vx_feat = split_offset(feats, vx_offset)
        vx_coord = split_offset(input_dict["coord"], vx_offset)
        vx_grid_coord = split_offset(input_dict["grid_coord"], vx_offset)

        input_dict["sp_feat"] = [
            scatter_mean(_feat[_inv], _sp, dim=0)
            for _feat, _inv, _sp in zip(vx_feat, inv, sp)
        ]
        input_dict["sp_coord"] = [
            scatter_mean(torch.floor(_coord[_inv] * 50), _sp, dim=0)
            for _coord, _inv, _sp in zip(vx_coord, inv, sp)
        ]
        input_dict["sp_grid_coord"] = [
            scatter_mean(_coord[_inv].float(), _sp, dim=0)
            for _coord, _inv, _sp in zip(vx_grid_coord, inv, sp)
        ]
        input_dict["sp"] = sp

        out = self.decoder(input_dict)

        return_dict = {}
        if self.criterion is not None and "origin_segment" in input_dict:
            return_dict.update(self.criterion(out, self.prepare_target(input_dict)))
        else:
            return_dict["loss"] = torch.tensor(
                0.0, device=feats.device, requires_grad=self.training
            )

        if not self.training:
            return_dict = self.prediction(out, return_dict, sp)
        return return_dict

    @torch.no_grad()
    def prepare_target(self, input_dict):
        pt_offset = input_dict["origin_offset"].int()
        pt_ins = split_offset(input_dict["origin_instance"], pt_offset)
        pt_sem = split_offset(input_dict["origin_segment"], pt_offset)
        sp = input_dict["sp"]

        target = {"inst_gt": []}
        for p_ins, p_cls, p_sp in zip(pt_ins, pt_sem, sp):
            p_ins = process_instance(
                p_ins.clone(),
                p_cls.clone(),
                self.segment_ignore_index,
                self.instance_ignore_index,
            )
            p_cls = process_label(
                p_cls.clone(),
                self.segment_ignore_index,
                self.semantic_ignore_index,
            )

            p_ins_mask = p_ins.clone()
            if torch.sum(p_ins_mask == self.instance_ignore_index) != 0:
                p_ins_mask[p_ins_mask == self.instance_ignore_index] = (
                    torch.max(p_ins_mask) + 1
                )
                p_ins_mask = torch.nn.functional.one_hot(p_ins_mask)[:, :-1]
            else:
                p_ins_mask = torch.nn.functional.one_hot(p_ins_mask)

            if p_ins_mask.shape[1] != 0:
                p_ins_mask = p_ins_mask.T
                sp_ins_mask = scatter_mean(p_ins_mask.float(), p_sp, dim=-1) > 0.5
            else:
                sp_ins_mask = p_ins_mask.new_zeros(
                    (0, p_sp.max() + 1), dtype=torch.bool
                )

            unique_insts = p_ins.unique()
            insts = unique_insts[unique_insts != self.instance_ignore_index]
            gt_labels = insts.new_zeros(len(insts))
            for i, inst_id in enumerate(insts):
                gt_labels[i] = p_cls[p_ins == inst_id][0]

            target["inst_gt"].append(dict(labels=gt_labels, masks=sp_ins_mask))
        return target

    def prediction(self, out, return_dict, sp):
        scores, masks, classes = self.predict_by_feat(out, sp)
        masks = masks.cpu().detach().numpy()
        classes = classes.cpu().detach().numpy()

        sort_scores = scores.sort(descending=True)
        sort_scores_index = sort_scores.indices.cpu().detach().numpy()
        return_dict["pred_scores"] = sort_scores.values.cpu().detach().numpy()
        return_dict["pred_masks"] = masks[sort_scores_index]
        return_dict["pred_classes"] = classes[sort_scores_index]
        return return_dict

    def predict_by_feat(self, out, sp):
        cls_preds = out["labels"][0]
        pred_masks = out["masks"][0]

        scores = F.softmax(cls_preds, dim=-1)[:, :-1]
        if out.get("scores", None) is not None and out["scores"] is not None:
            scores = scores * out["scores"][0].sigmoid()

        labels = (
            torch.arange(self.semantic_num_classes, device=scores.device)
            .unsqueeze(0)
            .repeat(len(cls_preds), 1)
            .flatten(0, 1)
        )
        scores, topk_idx = scores.flatten(0, 1).topk(self.topk_insts, sorted=False)
        labels = labels[topk_idx]
        topk_idx = torch.div(topk_idx, self.semantic_num_classes, rounding_mode="floor")

        mask_pred = pred_masks[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        mask_scores = (mask_pred_sigmoid * (mask_pred > 0)).sum(1) / (
            (mask_pred > 0).sum(1) + 1e-6
        )
        scores = scores * mask_scores

        if self.nms:
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel="linear"
            )

        mask_pred = mask_pred_sigmoid[:, sp[0]] > self.sp_score_thr

        score_mask = scores > self.score_thr
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        npoint_mask = mask_pred.sum(1) > self.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return scores, mask_pred, self.class_map[labels]

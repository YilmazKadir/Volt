import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.amp import autocast

from ..builder import MODELS, build_model

_CE_LABEL_SMOOTHING = 0.05
_LARGE_CONSTANT = 1e6


def _cross_entropy(inputs, targets, class_weight):
    return F.cross_entropy(
        inputs, targets, weight=class_weight, label_smoothing=_CE_LABEL_SMOOTHING
    )


@torch.jit.script
def batch_sigmoid_bce_loss(inputs, targets):
    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    pos_loss = torch.einsum("nc,mc->nm", pos, targets)
    neg_loss = torch.einsum("nc,mc->nm", neg, (1 - targets))
    return (pos_loss + neg_loss) / inputs.shape[1]


@torch.jit.script
def batch_dice_loss(inputs, targets):
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    return 1 - (numerator + 1) / (denominator + 1)


def get_iou(inputs, targets):
    inputs = inputs.sigmoid()
    binarized_inputs = (inputs >= 0.5).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    return intersection / (union + 1e-6)


def dice_loss(inputs, targets):
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    return (1 - (numerator + 1) / (denominator + 1)).mean()


@MODELS.register_module("SPFormerQueryClassificationCost")
class QueryClassificationCost:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, pred_instances, gt_instances, **kwargs):
        return (
            -pred_instances["scores"].softmax(-1)[:, gt_instances["labels"]]
            * self.weight
        )


@MODELS.register_module("SPFormerMaskBCECost")
class MaskBCECost:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, pred_instances, gt_instances, **kwargs):
        with autocast("cuda", enabled=False):
            cost = batch_sigmoid_bce_loss(
                pred_instances["masks"].float(), gt_instances["masks"].float()
            )
        return cost * self.weight


@MODELS.register_module("SPFormerMaskDiceCost")
class MaskDiceCost:
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, pred_instances, gt_instances, **kwargs):
        with autocast("cuda", enabled=False):
            cost = batch_dice_loss(
                pred_instances["masks"].float(), gt_instances["masks"].float()
            )
        return cost * self.weight


@MODELS.register_module("SPFormerHungarianMatcher")
class HungarianMatcher:
    def __init__(self, costs):
        self.costs = [build_model(cost) for cost in costs]

    @torch.no_grad()
    def __call__(self, pred_instances, gt_instances, **kwargs):
        labels = gt_instances["labels"]
        if len(labels) == 0:
            empty = labels.new_empty((0,), dtype=torch.int64)
            return empty, empty

        cost_value = torch.stack(
            [cost(pred_instances, gt_instances) for cost in self.costs]
        ).sum(dim=0)
        if torch.isnan(cost_value).any() or torch.isinf(cost_value).any():
            cost_value = torch.where(
                torch.isnan(cost_value) | torch.isinf(cost_value),
                torch.full_like(cost_value, _LARGE_CONSTANT),
                cost_value,
            )
        query_ids, object_ids = linear_sum_assignment(cost_value.cpu().numpy())
        return labels.new_tensor(query_ids, dtype=torch.int64), labels.new_tensor(
            object_ids, dtype=torch.int64
        )


@MODELS.register_module("SPFormerCriterion")
class SPFormerCriterion:
    def __init__(
        self,
        matcher,
        loss_weight,
        non_object_weight,
        num_classes,
        fix_dice_loss_weight,
        iter_matcher,
        fix_mean_loss=False,
    ):
        self.matcher = build_model(matcher)
        self.class_weight = [1] * num_classes + [non_object_weight]
        self.loss_weight = loss_weight
        self.fix_dice_loss_weight = fix_dice_loss_weight
        self.iter_matcher = iter_matcher
        self.fix_mean_loss = fix_mean_loss

    def get_layer_loss(self, aux_outputs, insts, indices=None):
        cls_preds = aux_outputs["labels"]
        pred_masks = aux_outputs["masks"]
        pred_insts = [
            dict(scores=cls_preds[i], masks=pred_masks[i])
            for i in range(len(cls_preds))
        ]
        if indices is None:
            indices = [self.matcher(pred_insts[i], insts[i]) for i in range(len(insts))]

        cls_losses = []
        for cls_pred, inst, (idx_q, idx_gt) in zip(cls_preds, insts, indices):
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long
            )
            cls_target[idx_q] = inst["labels"][idx_gt]
            cls_losses.append(
                _cross_entropy(
                    cls_pred, cls_target, cls_pred.new_tensor(self.class_weight)
                )
            )
        cls_loss = torch.mean(torch.stack(cls_losses))

        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for batch_i, (mask, inst, (idx_q, idx_gt)) in enumerate(
            zip(pred_masks, insts, indices)
        ):
            if inst["masks"].shape[0] == 0:
                continue
            pred_mask = mask[idx_q]
            tgt_mask = inst["masks"][idx_gt]
            mask_bce_losses.append(
                F.binary_cross_entropy_with_logits(pred_mask, tgt_mask.float())
            )
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask.float()))

            if aux_outputs.get("scores", None) is None:
                continue
            pred_score = aux_outputs["scores"][batch_i][idx_q]
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)
            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                score_losses.append(
                    F.mse_loss(pred_score[filter_id], tgt_score[filter_id])
                )

        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = torch.tensor(
                0.0, requires_grad=True, device=pred_masks[0].device
            )

        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum() / len(pred_masks)
            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4
            if self.fix_mean_loss:
                mask_bce_loss = mask_bce_loss * len(pred_masks) / len(mask_bce_losses)
                mask_dice_loss = (
                    mask_dice_loss * len(pred_masks) / len(mask_dice_losses)
                )
        else:
            device = pred_masks[0].device
            mask_bce_loss = torch.tensor(0.0, requires_grad=True, device=device)
            mask_dice_loss = torch.tensor(0.0, requires_grad=True, device=device)

        return (
            self.loss_weight[0] * cls_loss
            + self.loss_weight[1] * mask_bce_loss
            + self.loss_weight[2] * mask_dice_loss
            + self.loss_weight[3] * score_loss
        )

    def __call__(self, pred, target):
        insts = target["inst_gt"]
        cls_preds = pred["labels"]
        pred_masks = pred["masks"]
        pred_insts = [
            dict(scores=cls_preds[i], masks=pred_masks[i])
            for i in range(len(cls_preds))
        ]
        indices = [self.matcher(pred_insts[i], insts[i]) for i in range(len(insts))]

        cls_losses = []
        for cls_pred, inst, (idx_q, idx_gt) in zip(cls_preds, insts, indices):
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long
            )
            cls_target[idx_q] = inst["labels"][idx_gt]
            cls_losses.append(
                _cross_entropy(
                    cls_pred, cls_target, cls_pred.new_tensor(self.class_weight)
                )
            )
        cls_loss = torch.mean(torch.stack(cls_losses))

        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for i, (mask, inst, (idx_q, idx_gt)) in enumerate(
            zip(pred_masks, insts, indices)
        ):
            if inst["masks"].shape[0] == 0:
                continue
            pred_mask = mask[idx_q]
            tgt_mask = inst["masks"][idx_gt]
            mask_bce_losses.append(
                F.binary_cross_entropy_with_logits(pred_mask, tgt_mask.float())
            )
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask.float()))
            if pred.get("scores", None) is None:
                continue
            pred_score = pred["scores"][i][idx_q]
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)
            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                score_losses.append(
                    F.mse_loss(pred_score[filter_id], tgt_score[filter_id])
                )

        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = torch.tensor(
                0.0, requires_grad=True, device=pred_masks[0].device
            )

        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum() / len(pred_masks)
            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4
            if self.fix_mean_loss:
                mask_bce_loss = mask_bce_loss * len(pred_masks) / len(mask_bce_losses)
                mask_dice_loss = (
                    mask_dice_loss * len(pred_masks) / len(mask_dice_losses)
                )
        else:
            device = pred_masks[0].device
            mask_bce_loss = torch.tensor(0.0, requires_grad=True, device=device)
            mask_dice_loss = torch.tensor(0.0, requires_grad=True, device=device)

        loss = (
            self.loss_weight[0] * cls_loss
            + self.loss_weight[1] * mask_bce_loss
            + self.loss_weight[2] * mask_dice_loss
            + self.loss_weight[3] * score_loss
        )

        losses = {
            "loss_cls": cls_loss,
            "loss_mask": mask_bce_loss,
            "loss_dice": mask_dice_loss,
            "loss_score": score_loss,
        }
        if "aux_outputs" in pred:
            indices_for_aux = None if self.iter_matcher else indices
            for aux_outputs in pred["aux_outputs"]:
                loss = loss + self.get_layer_loss(aux_outputs, insts, indices_for_aux)
        losses["loss"] = loss
        return losses

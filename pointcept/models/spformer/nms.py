import torch


def mask_matrix_nms(
    masks: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    filter_thr: float = -1,
    nms_pre: int = -1,
    max_num: int = -1,
    kernel: str = "gaussian",
    sigma: float = 2.0,
    mask_area: torch.Tensor | None = None,
):
    if len(labels) == 0:
        return (
            scores.new_zeros(0),
            labels.new_zeros(0),
            masks.new_zeros(0, *masks.shape[-1:]),
            labels.new_zeros(0),
        )
    if mask_area is None:
        mask_area = masks.sum(1).float()

    scores, sort_inds = torch.sort(scores, descending=True)
    keep_inds = sort_inds
    if nms_pre > 0 and len(sort_inds) > nms_pre:
        sort_inds = sort_inds[:nms_pre]
        keep_inds = keep_inds[:nms_pre]
        scores = scores[:nms_pre]
    masks = masks[sort_inds]
    mask_area = mask_area[sort_inds]
    labels = labels[sort_inds]

    num_masks = len(labels)
    flatten_masks = masks.reshape(num_masks, -1).float()
    inter_matrix = torch.mm(flatten_masks, flatten_masks.transpose(1, 0))
    expanded_mask_area = mask_area.expand(num_masks, num_masks)
    iou_matrix = (
        inter_matrix
        / (expanded_mask_area + expanded_mask_area.transpose(1, 0) - inter_matrix)
    ).triu(diagonal=1)
    expanded_labels = labels.expand(num_masks, num_masks)
    label_matrix = (expanded_labels == expanded_labels.transpose(1, 0)).triu(diagonal=1)

    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(num_masks, num_masks).transpose(1, 0)
    decay_iou = iou_matrix * label_matrix

    if kernel == "linear":
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        decay_matrix = torch.exp(-1 * sigma * (decay_iou**2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou**2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)

    scores = scores * decay_coefficient

    if filter_thr > 0:
        keep = scores >= filter_thr
        keep_inds = keep_inds[keep]
        if not keep.any():
            return (
                scores.new_zeros(0),
                labels.new_zeros(0),
                masks.new_zeros(0, *masks.shape[-1:]),
                labels.new_zeros(0),
            )
        masks = masks[keep]
        scores = scores[keep]
        labels = labels[keep]

    scores, sort_inds = torch.sort(scores, descending=True)
    keep_inds = keep_inds[sort_inds]
    if max_num > 0 and len(sort_inds) > max_num:
        sort_inds = sort_inds[:max_num]
        keep_inds = keep_inds[:max_num]
        scores = scores[:max_num]
    masks = masks[sort_inds]
    labels = labels[sort_inds]

    return scores, labels, masks, keep_inds

import torch


def split_offset(value: torch.Tensor, offset: torch.Tensor) -> list[torch.Tensor]:
    ret = []
    start = 0
    for end in offset:
        end_int = int(end)
        ret.append(value[start:end_int])
        start = end_int
    return ret


def process_label(
    labels: torch.Tensor,
    segment_ignore_index: tuple[int, ...] = (-1, 0, 2),
    semantic_ignore_index: int = -1,
) -> torch.Tensor:
    ignored = [
        label for label in segment_ignore_index if label != semantic_ignore_index
    ]
    for label in ignored:
        labels[labels == label] = semantic_ignore_index
    for label in sorted([label for label in ignored if label >= 0], reverse=True):
        labels[labels > label] -= 1
    return labels


def process_instance(
    instance: torch.Tensor,
    segment: torch.Tensor,
    segment_ignore_index: tuple[int, ...] = (-1, 0, 2),
    instance_ignore_index: int = -1,
) -> torch.Tensor:
    mask = torch.ones_like(instance, dtype=torch.bool)
    for label in segment_ignore_index:
        mask[segment == label] = False
    instance[~mask] = instance_ignore_index
    _, inverse = torch.unique(instance[mask], return_inverse=True)
    instance[mask] = inverse
    return instance

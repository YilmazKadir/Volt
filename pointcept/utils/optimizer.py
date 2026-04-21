"""
Optimizer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import copy
import torch
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry

OPTIMIZERS = Registry("optimizers")


OPTIMIZERS.register_module(module=torch.optim.SGD, name="SGD")
OPTIMIZERS.register_module(module=torch.optim.Adam, name="Adam")
OPTIMIZERS.register_module(module=torch.optim.AdamW, name="AdamW")


def split_weight_decay_groups(param_groups):
    new_params = []
    for group in param_groups:
        decay_group = {k: v for k, v in group.items() if k not in ["names", "params"]}
        no_decay_group = {
            k: v for k, v in group.items() if k not in ["names", "params"]
        }
        no_decay_group["weight_decay"] = 0.0

        decay_group["names"], decay_group["params"] = [], []
        no_decay_group["names"], no_decay_group["params"] = [], []

        for n, p in zip(group["names"], group["params"]):
            if p.ndim == 1 or n.endswith(".bias"):
                no_decay_group["names"].append(n)
                no_decay_group["params"].append(p)
            else:
                decay_group["names"].append(n)
                decay_group["params"].append(p)

        if decay_group["params"]:
            new_params.append(decay_group)
        if no_decay_group["params"]:
            new_params.append(no_decay_group)

    return new_params


def build_optimizer(cfg, model, param_dicts=None):
    cfg = copy.deepcopy(cfg)
    if param_dicts is None:
        cfg.params = [dict(names=[], params=[])]
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            cfg.params[0]["names"].append(n)
            cfg.params[0]["params"].append(p)
    else:
        cfg.params = [dict(names=[], params=[], lr=cfg.lr)]
        for i in range(len(param_dicts)):
            param_group = dict(names=[], params=[])
            if "lr" in param_dicts[i].keys():
                param_group["lr"] = param_dicts[i].lr
            if "momentum" in param_dicts[i].keys():
                param_group["momentum"] = param_dicts[i].momentum
            if "weight_decay" in param_dicts[i].keys():
                param_group["weight_decay"] = param_dicts[i].weight_decay
            cfg.params.append(param_group)

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            flag = False
            for i in range(len(param_dicts)):
                if param_dicts[i].keyword in n:
                    cfg.params[i + 1]["names"].append(n)
                    cfg.params[i + 1]["params"].append(p)
                    flag = True
                    break
            if not flag:
                cfg.params[0]["names"].append(n)
                cfg.params[0]["params"].append(p)

    cfg.params = split_weight_decay_groups(cfg.params)

    logger = get_root_logger()
    for i in range(len(cfg.params)):
        param_names = cfg.params[i].pop("names")
        message = ""
        for key in cfg.params[i].keys():
            if key != "params":
                message += f" {key}: {cfg.params[i][key]};"
        logger.info(f"Params Group {i+1} -{message} Params: {param_names}.")
    return OPTIMIZERS.build(cfg=cfg)

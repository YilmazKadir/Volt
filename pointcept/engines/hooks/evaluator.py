"""
Evaluate Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import numpy as np
import torch
import torch.distributed as dist
import pointops
from uuid import uuid4

import pointcept.utils.comm as comm
from pointcept.engines.defaults import AMP_DTYPE
from pointcept.utils.misc import intersection_and_union_gpu

from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class ClsEvaluator(HookBase):
    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                with torch.amp.autocast(
                    "cuda",
                    enabled=self.trainer.cfg.enable_amp,
                    dtype=AMP_DTYPE[self.trainer.cfg.amp_dtype],
                ):
                    if self.trainer.cfg.use_ema:
                        output_dict = self.trainer.ema(input_dict)
                    else:
                        output_dict = self.trainer.model(input_dict)
            output = output_dict["cls_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            label = input_dict["category"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                label,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                dist.all_reduce(intersection)
                dist.all_reduce(union)
                dist.all_reduce(target)
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] "
                "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = all_acc  # save for saver
        self.trainer.comm_info["current_metric_name"] = "allAcc"  # save for saver

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("allAcc", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class SemSegEvaluator(HookBase):
    def __init__(self, write_cls_iou=False):
        self.write_cls_iou = write_cls_iou

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        for i, input_dict in enumerate(self.trainer.val_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                with torch.amp.autocast(
                    "cuda",
                    enabled=self.trainer.cfg.enable_amp,
                    dtype=AMP_DTYPE[self.trainer.cfg.amp_dtype],
                ):
                    if self.trainer.cfg.use_ema:
                        output_dict = self.trainer.ema(input_dict)
                    else:
                        output_dict = self.trainer.model(input_dict)
            output = output_dict["seg_logits"]
            loss = output_dict["loss"]
            pred = output.max(1)[1]
            segment = input_dict["segment"]
            if "inverse" in input_dict.keys():
                assert "origin_segment" in input_dict.keys()
                pred = pred[input_dict["inverse"]]
                segment = input_dict["origin_segment"]
            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                self.trainer.cfg.data.num_classes,
                self.trainer.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                dist.all_reduce(intersection)
                dist.all_reduce(union)
                dist.all_reduce(target)
            intersection, union, target = (
                intersection.cpu().numpy(),
                union.cpu().numpy(),
                target.cpu().numpy(),
            )
            # Here there is no need to sync since sync happened in dist.all_reduce
            self.trainer.storage.put_scalar("val_intersection", intersection)
            self.trainer.storage.put_scalar("val_union", union)
            self.trainer.storage.put_scalar("val_target", target)
            self.trainer.storage.put_scalar("val_loss", loss.item())
            info = "Test: [{iter}/{max_iter}] ".format(
                iter=i + 1, max_iter=len(self.trainer.val_loader)
            )
            if "origin_coord" in input_dict.keys():
                info = "Interp. " + info
            self.trainer.logger.info(
                info
                + "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )
        loss_avg = self.trainer.storage.history("val_loss").avg
        intersection = self.trainer.storage.history("val_intersection").total
        union = self.trainer.storage.history("val_union").total
        target = self.trainer.storage.history("val_target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.trainer.logger.info(
            "Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc
            )
        )
        for i in range(self.trainer.cfg.data.num_classes):
            self.trainer.logger.info(
                "Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}".format(
                    idx=i,
                    name=self.trainer.cfg.data.names[i],
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                )
            )
        current_epoch = self.trainer.epoch + 1
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar("val/mIoU", m_iou, current_epoch)
            self.trainer.writer.add_scalar("val/mAcc", m_acc, current_epoch)
            self.trainer.writer.add_scalar("val/allAcc", all_acc, current_epoch)
            if self.write_cls_iou:
                for i in range(self.trainer.cfg.data.num_classes):
                    self.trainer.writer.add_scalar(
                        f"val/cls_{i}-{self.trainer.cfg.data.names[i]} IoU",
                        iou_class[i],
                        current_epoch,
                    )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = m_iou  # save for saver
        self.trainer.comm_info["current_metric_name"] = "mIoU"  # save for saver

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("mIoU", self.trainer.best_metric_value)
        )


@HOOKS.register_module()
class InsSegEvaluator(HookBase):
    def __init__(
        self,
        segment_ignore_index=(-1, 0, 2),
        instance_ignore_index=-1,
        min_region_size=100,
    ):
        self.segment_ignore_index = tuple(segment_ignore_index)
        self.instance_ignore_index = int(instance_ignore_index)

        self.valid_class_names = None  # update in before train
        self.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_sizes = int(min_region_size)
        self.distance_threshes = float("inf")
        self.distance_confs = -float("inf")

    def before_train(self):
        self.valid_class_names = [
            self.trainer.cfg.data.names[i]
            for i in range(self.trainer.cfg.data.num_classes)
            if i not in self.segment_ignore_index
        ]
        self.class_name_map = {
            i: self.trainer.cfg.data.names[i]
            for i in range(self.trainer.cfg.data.num_classes)
            if i not in self.segment_ignore_index
        }

    def after_epoch(self):
        if self.trainer.cfg.evaluate:
            self.eval()

    def associate_instances(self, pred, segment, instance):
        segment = segment.cpu().numpy()
        instance = instance.cpu().numpy()
        pred_classes = np.asarray(pred["pred_classes"])
        void_mask = np.isin(segment, self.segment_ignore_index)
        gt_instances = {name: [] for name in self.valid_class_names}
        pred_instances = {name: [] for name in self.valid_class_names}

        assert (
            pred_classes.shape[0]
            == pred["pred_scores"].shape[0]
            == pred["pred_masks"].shape[0]
        )
        assert pred["pred_masks"].shape[1] == segment.shape[0] == instance.shape[0]
        instance_ids, idx, counts = np.unique(
            instance, return_index=True, return_counts=True
        )
        segment_ids = segment[idx]
        for i in range(len(instance_ids)):
            if instance_ids[i] == self.instance_ignore_index:
                continue
            if segment_ids[i] in self.segment_ignore_index:
                continue
            gt_inst = dict()
            gt_inst["instance_id"] = instance_ids[i]
            gt_inst["segment_id"] = segment_ids[i]
            gt_inst["dist_conf"] = 0.0
            gt_inst["med_dist"] = -1.0
            gt_inst["vert_count"] = counts[i]
            gt_inst["matched_pred"] = []
            gt_instances[self.class_name_map[segment_ids[i]]].append(gt_inst)

        instance_id = 0
        for i in range(len(pred_classes)):
            if pred_classes[i] in self.segment_ignore_index:
                continue
            pred_inst = dict()
            pred_inst["uuid"] = uuid4()
            pred_inst["instance_id"] = instance_id
            pred_inst["segment_id"] = pred_classes[i]
            pred_inst["confidence"] = pred["pred_scores"][i]
            pred_inst["mask"] = np.not_equal(pred["pred_masks"][i], 0)
            pred_inst["vert_count"] = np.count_nonzero(pred_inst["mask"])
            pred_inst["void_intersection"] = np.count_nonzero(
                np.logical_and(void_mask, pred_inst["mask"])
            )
            if pred_inst["vert_count"] < self.min_region_sizes:
                continue
            segment_name = self.class_name_map[pred_inst["segment_id"]]
            matched_gt = []
            for gt_inst in gt_instances[segment_name]:
                intersection = np.count_nonzero(
                    np.logical_and(
                        instance == gt_inst["instance_id"], pred_inst["mask"]
                    )
                )
                if intersection > 0:
                    gt_inst_ = gt_inst.copy()
                    pred_inst_ = pred_inst.copy()
                    gt_inst_["intersection"] = intersection
                    pred_inst_["intersection"] = intersection
                    matched_gt.append(gt_inst_)
                    gt_inst["matched_pred"].append(pred_inst_)
            pred_inst["matched_gt"] = matched_gt
            pred_instances[segment_name].append(pred_inst)
            instance_id += 1
        return gt_instances, pred_instances

    def evaluate_matches(self, scenes):
        overlaps = self.overlaps
        min_region_sizes = [self.min_region_sizes]
        dist_threshes = [self.distance_threshes]
        dist_confs = [self.distance_confs]
        ap_table = np.zeros(
            (len(dist_threshes), len(self.valid_class_names), len(overlaps)), float
        )
        pr_rc = np.zeros((2, len(self.valid_class_names), len(overlaps)), float)

        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)
        ):
            for oi, overlap_th in enumerate(overlaps):
                pred_visited = {}
                for scene in scenes:
                    for _ in scene["pred"]:
                        for label_name in self.valid_class_names:
                            for p in scene["pred"][label_name]:
                                if "uuid" in p:
                                    pred_visited[p["uuid"]] = False

                for li, label_name in enumerate(self.valid_class_names):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False

                    for scene in scenes:
                        pred_instances = scene["pred"][label_name]
                        gt_instances = scene["gt"][label_name]

                        gt_instances = [
                            gt
                            for gt in gt_instances
                            if gt["vert_count"] >= min_region_size
                            and gt["med_dist"] <= distance_thresh
                            and gt["dist_conf"] >= distance_conf
                        ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=bool)

                        for gti, gt in enumerate(gt_instances):
                            found_match = False
                            for pred in gt["matched_pred"]:
                                if pred_visited[pred["uuid"]]:
                                    continue
                                overlap = float(pred["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - pred["intersection"]
                                )
                                if overlap > overlap_th:
                                    confidence = pred["confidence"]
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred["uuid"]] = True
                            if not found_match:
                                hard_false_negatives += 1

                        cur_true = cur_true[cur_match]
                        cur_score = cur_score[cur_match]

                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred["matched_gt"]:
                                overlap = float(gt["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - gt["intersection"]
                                )
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred["void_intersection"]
                                for gt in pred["matched_gt"]:
                                    if gt["segment_id"] in self.segment_ignore_index:
                                        num_ignore += gt["intersection"]
                                    if (
                                        gt["vert_count"] < min_region_size
                                        or gt["med_dist"] > distance_thresh
                                        or gt["dist_conf"] < distance_conf
                                    ):
                                        num_ignore += gt["intersection"]
                                proportion_ignore = (
                                    float(num_ignore) / pred["vert_count"]
                                )
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, pred["confidence"])

                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    if has_gt and has_pred:
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        _, unique_indices = np.unique(y_score_sorted, return_index=True)
                        num_prec_recall = len(unique_indices) + 1
                        num_examples = len(y_score_sorted)
                        num_true_examples = (
                            y_true_sorted_cumsum[-1]
                            if len(y_true_sorted_cumsum) > 0
                            else 0
                        )

                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp) if (tp + fp) > 0 else 0.0
                            r = float(tp) / (tp + fn) if (tp + fn) > 0 else 0.0
                            precision[idx_res] = p
                            recall[idx_res] = r

                        precision[-1] = 1.0
                        recall[-1] = 0.0

                        f1_score = 2 * precision * recall / (precision + recall + 1e-4)
                        f1_argmax = f1_score.argmax()
                        best_pr = precision[f1_argmax]
                        best_rc = recall[f1_argmax]

                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.0)
                        step_widths = np.convolve(
                            recall_for_conv, [-0.5, 0, 0.5], "valid"
                        )
                        ap_current = np.dot(precision, step_widths)
                    elif has_gt:
                        ap_current = 0.0
                        best_pr = 0.0
                        best_rc = 0.0
                    else:
                        ap_current = float("nan")
                        best_pr = float("nan")
                        best_rc = float("nan")

                    ap_table[di, li, oi] = ap_current
                    pr_rc[0, li, oi] = best_pr
                    pr_rc[1, li, oi] = best_rc

        d_inf = 0
        o50 = np.where(np.isclose(self.overlaps, 0.5))
        o25 = np.where(np.isclose(self.overlaps, 0.25))
        o_all_but_25 = np.where(np.logical_not(np.isclose(self.overlaps, 0.25)))

        ap_scores = dict()
        ap_scores["all_ap"] = np.nanmean(ap_table[d_inf, :, o_all_but_25])
        ap_scores["all_ap_50%"] = np.nanmean(ap_table[d_inf, :, o50])
        ap_scores["all_ap_25%"] = np.nanmean(ap_table[d_inf, :, o25])
        ap_scores["all_prec_50%"] = np.nanmean(pr_rc[0, :, o50])
        ap_scores["all_rec_50%"] = np.nanmean(pr_rc[1, :, o50])
        ap_scores["classes"] = {}
        for li, label_name in enumerate(self.valid_class_names):
            ap_scores["classes"][label_name] = {
                "ap": np.average(ap_table[d_inf, li, o_all_but_25]),
                "ap50%": np.average(ap_table[d_inf, li, o50]),
                "ap25%": np.average(ap_table[d_inf, li, o25]),
                "prec50%": np.average(pr_rc[0, li, o50]),
                "rec50%": np.average(pr_rc[1, li, o50]),
            }
        return ap_scores

    def print_results(self, ap_scores):
        all_ap = ap_scores["all_ap"]
        all_ap_50 = ap_scores["all_ap_50%"]
        all_ap_25 = ap_scores["all_ap_25%"]
        all_prec_50 = ap_scores["all_prec_50%"]
        all_rec_50 = ap_scores["all_rec_50%"]

        sep = ""
        col1 = ":"
        line_len = 66
        self.trainer.logger.info("#" * line_len)
        line = ""
        line += "{:<15}".format("what") + sep + col1
        line += "{:>10}".format("AP") + sep
        line += "{:>10}".format("AP_50%") + sep
        line += "{:>10}".format("AP_25%") + sep
        line += "{:>10}".format("Prec_50%") + sep
        line += "{:>10}".format("Rec_50%") + sep
        self.trainer.logger.info(line)
        self.trainer.logger.info("#" * line_len)

        for label_name in self.valid_class_names:
            ap = ap_scores["classes"][label_name]["ap"]
            ap_50 = ap_scores["classes"][label_name]["ap50%"]
            ap_25 = ap_scores["classes"][label_name]["ap25%"]
            prec_50 = ap_scores["classes"][label_name]["prec50%"]
            rec_50 = ap_scores["classes"][label_name]["rec50%"]
            line = "{:<15}".format(label_name) + sep + col1
            line += sep + "{:>10.3f}".format(ap) + sep
            line += sep + "{:>10.3f}".format(ap_50) + sep
            line += sep + "{:>10.3f}".format(ap_25) + sep
            line += sep + "{:>10.3f}".format(prec_50) + sep
            line += sep + "{:>10.3f}".format(rec_50) + sep
            self.trainer.logger.info(line)

        self.trainer.logger.info("-" * line_len)
        line = "{:<15}".format("average") + sep + col1
        line += "{:>10.3f}".format(all_ap) + sep
        line += "{:>10.3f}".format(all_ap_50) + sep
        line += "{:>10.3f}".format(all_ap_25) + sep
        line += "{:>10.3f}".format(all_prec_50) + sep
        line += "{:>10.3f}".format(all_rec_50) + sep
        self.trainer.logger.info(line)
        self.trainer.logger.info("#" * line_len)

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        scenes = []
        val_loader = self.trainer.val_loader
        if isinstance(val_loader, (list, tuple)):
            val_loader = val_loader[0]

        val_sum = 0.0
        val_n = 0
        val_loss_sums = {}

        for i, input_dict in enumerate(val_loader):
            assert len(input_dict["offset"]) == 1
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                if self.trainer.cfg.use_ema:
                    output_dict = self.trainer.ema(input_dict)
                else:
                    output_dict = self.trainer.model(input_dict)

            loss = output_dict["loss"]
            losses = {
                k: v for k, v in output_dict.items() if ("loss" in k and k != "loss")
            }

            segment = input_dict["segment"]
            instance = input_dict["instance"]
            if "origin_coord" in input_dict.keys():
                segment = input_dict["origin_segment"]
                instance = input_dict["origin_instance"]
                if output_dict["pred_masks"].shape[1] != segment.shape[0]:
                    reverse, _ = pointops.knn_query(
                        1,
                        input_dict["coord"].float(),
                        input_dict["offset"].int(),
                        input_dict["origin_coord"].float(),
                        input_dict["origin_offset"].int(),
                    )
                    reverse = reverse.cpu().flatten().long()
                    output_dict["pred_masks"] = output_dict["pred_masks"][:, reverse]

            gt_instances, pred_instance = self.associate_instances(
                output_dict, segment, instance
            )
            scenes.append(dict(gt=gt_instances, pred=pred_instance))

            val_sum += float(loss.item())
            val_n += 1
            for key, value in losses.items():
                val_loss_sums[key] = val_loss_sums.get(key, 0.0) + float(value.item())

            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(val_loader), loss=loss.item()
                )
            )
            torch.cuda.empty_cache()

        loss_avg = {}
        if val_n > 0:
            loss_avg["val_loss"] = val_sum / val_n
            for key in val_loss_sums.keys():
                loss_avg[key] = val_loss_sums[key] / val_n
        else:
            loss_avg["val_loss"] = float("nan")
            for key in val_loss_sums.keys():
                loss_avg[key] = float("nan")

        comm.synchronize()
        scenes_sync = comm.gather(scenes, dst=0)

        if comm.is_main_process():
            scenes = [scene for scenes_ in scenes_sync for scene in scenes_]
            ap_scores = self.evaluate_matches(scenes)
            self.print_results(ap_scores)
            current_epoch = self.trainer.epoch + 1
            if self.trainer.writer is not None:
                self.trainer.writer.add_scalar(
                    "val/loss", loss_avg["val_loss"], current_epoch
                )
                for key in loss_avg.keys():
                    if key == "val_loss":
                        continue
                    self.trainer.writer.add_scalar(
                        "val/" + key, loss_avg[key], current_epoch
                    )
                self.trainer.writer.add_scalar(
                    "val/mAP", ap_scores["all_ap"], current_epoch
                )
                self.trainer.writer.add_scalar(
                    "val/AP50", ap_scores["all_ap_50%"], current_epoch
                )
                self.trainer.writer.add_scalar(
                    "val/AP25", ap_scores["all_ap_25%"], current_epoch
                )
                self.trainer.writer.add_scalar(
                    "val/Prec50", ap_scores["all_prec_50%"], current_epoch
                )
                self.trainer.writer.add_scalar(
                    "val/Rec50", ap_scores["all_rec_50%"], current_epoch
                )
            self.trainer.logger.info(
                "<<<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<"
            )
            self.trainer.comm_info["current_metric_value"] = ap_scores["all_ap_50%"]
            self.trainer.comm_info["current_metric_name"] = "AP50"

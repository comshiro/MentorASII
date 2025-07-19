# utils/loss.py - Functions for calculating losses in object detection and segmentation tasks

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

EPS = 1e-7 # 10^(-7) - pentru stabilitate numerica


def bbox_decode(tbox, x_idx, y_idx, stride, H, W, device):
    cx = (x_idx + tbox[:, 0]) * stride
    cy = (y_idx + tbox[:, 1]) * stride
    w = tbox[:, 2] * stride
    h = tbox[:, 3] * stride
    return torch.stack([cx, cy, w, h], dim=1)


def box_cxcywh_to_xyxy(box):
    cx, cy, w, h = box.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def bbox_iou_xyxy(box1, box2, eps=1e-7):

    inter_x1 = torch.max(box1[:, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, 3], box2[:, 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)
    union = area1 + area2 - inter + eps
    return inter / union


def bbox_ciou(box1, box2, eps=1e-7): # iou - Complete Intersection over Union

    iou = bbox_iou_xyxy(box1, box2, eps)
    # center distance
    b1_cx = (box1[:, 0] + box1[:, 2]) / 2
    b1_cy = (box1[:, 1] + box1[:, 3]) / 2
    b2_cx = (box2[:, 0] + box2[:, 2]) / 2
    b2_cy = (box2[:, 1] + box2[:, 3]) / 2
    center_dist = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2

    # enclosing box
    enc_x1 = torch.min(box1[:, 0], box2[:, 0])
    enc_y1 = torch.min(box1[:, 1], box2[:, 1])
    enc_x2 = torch.max(box1[:, 2], box2[:, 2])
    enc_y2 = torch.max(box1[:, 3], box2[:, 3])
    enc_w = (enc_x2 - enc_x1).clamp(min=0)
    enc_h = (enc_y2 - enc_y1).clamp(min=0)
    c = enc_w ** 2 + enc_h ** 2 + eps

    # aspect ratio term
    w1 = (box1[:, 2] - box1[:, 0]).clamp(min=0)
    h1 = (box1[:, 3] - box1[:, 1]).clamp(min=0)
    w2 = (box2[:, 2] - box2[:, 0]).clamp(min=0)
    h2 = (box2[:, 3] - box2[:, 1]).clamp(min=0)
    v = (4 / (3.14159265 ** 2)) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (center_dist / c + alpha * v)
    return ciou


class FocalBCE(nn.Module): # BCE - Binary Cross Entropy, Focal Loss - pentru clasificare binara - functie de loss

    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (K, C), targets: (K, C) - logits - predictiile inainte de activare, targets - etichetele
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1 - prob) * (1 - targets) # pt - probabilitatea de apartenenta la clasa
        focal = (self.alpha * targets + (1 - self.alpha) * (1 - targets)) * (1 - pt) ** self.gamma * bce
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


class DetectionSegLoss(nn.Module):

    def __init__(self, num_classes, mask_weight=1.0, cls_weight=1.0, box_weight=5.0):
        super().__init__()
        self.num_classes = num_classes
        self.mask_weight = mask_weight
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.focal = FocalBCE(gamma=2.0, alpha=0.25)

    def forward(self, outputs: Dict, targets: Dict, proto, mask_targets=None):

        device = proto.device
        total_box_loss = torch.tensor(0., device=device)
        total_cls_loss = torch.tensor(0., device=device)
        total_mask_loss = torch.tensor(0., device=device)
        count = 0

        for lvl, tdict in targets.items():
            stride_map = {
                "P3": 8,
                "P4": 16,
                "P5": 32,
                "P6": 64
            }
            stride = stride_map[lvl]
            pred = outputs[lvl]
            box_pred = pred["box"]    # (B,4,H,W)
            cls_pred = pred["cls"]    # (B,C,H,W)
            coef_pred = pred["coef"]  # (B,P,H,W)

            b_idx = tdict["batch_idx"]
            if b_idx.numel() == 0:
                continue
            x_idx = tdict["x_idx"]
            y_idx = tdict["y_idx"]
            tbox = tdict["tbox"]      # (K,4)
            tcls = tdict["cls"]       # (K,)

            pred_box_cells = box_pred[b_idx, :, y_idx, x_idx].permute(1, 0)  # (4,K) -> vrem (K,4)
            pred_box_cells = pred_box_cells.transpose(0, 1)  # (K,4)

            pred_cls_cells = cls_pred[b_idx, :, y_idx, x_idx].permute(1, 0)  # (C,K)->(K,C)
            pred_cls_cells = pred_cls_cells.transpose(0, 1)

            decoded_pred = bbox_decode(pred_box_cells, x_idx, y_idx, stride,
                                       box_pred.shape[-2], box_pred.shape[-1], device)
            decoded_target = bbox_decode(tbox, x_idx, y_idx, stride,
                                         box_pred.shape[-2], box_pred.shape[-1], device)

            pred_xyxy = box_cxcywh_to_xyxy(decoded_pred)
            target_xyxy = box_cxcywh_to_xyxy(decoded_target)

            ciou = bbox_ciou(pred_xyxy, target_xyxy)  # (K,)
            box_loss = (1 - ciou).mean()

            cls_onehot = torch.zeros((tcls.shape[0], self.num_classes),
                                     device=device)
            cls_onehot[torch.arange(tcls.shape[0]), tcls] = 1.0
            cls_loss = self.focal(pred_cls_cells, cls_onehot)

            # Mask loss (placeholder simplu: 0 la început dacă nu ai GT)
            mask_loss = torch.tensor(0., device=device)
            # TODO: după ce implementezi pipeline-ul coef->proto->crop->mask_gt, calculezi BCE aici.

            total_box_loss += box_loss
            total_cls_loss += cls_loss
            total_mask_loss += mask_loss
            count += 1

        if count == 0:
            total = total_box_loss + total_cls_loss + total_mask_loss
        else:
            total = (self.box_weight * total_box_loss +
                     self.cls_weight * total_cls_loss +
                     self.mask_weight * total_mask_loss)

        return total, {
            "loss_box": total_box_loss.detach(),
            "loss_cls": total_cls_loss.detach(),
            "loss_mask": total_mask_loss.detach(),
            "loss_total": total.detach()
        }

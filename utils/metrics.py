import torch
from typing import List, Dict

def box_iou_matrix(boxes1, boxes2):
    # boxes: (N,4) xyxy
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]))
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-7)


def ap50(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, iou_thres=0.5): # average precision at IoU threshold, gt_boxes - ground truth boxes

    device = pred_boxes.device
    classes = torch.unique(torch.cat([pred_classes, gt_classes], dim=0))
    ap_list = []
    for c in classes:
        pb = pred_boxes[pred_classes == c]
        ps = pred_scores[pred_classes == c]
        gb = gt_boxes[gt_classes == c]
        if gb.numel() == 0 and pb.numel() == 0:
            continue
        if gb.numel() == 0:
            ap_list.append(torch.tensor(0., device=device))
            continue
        if pb.numel() == 0:
            ap_list.append(torch.tensor(0., device=device))
            continue
        idx = torch.argsort(ps, descending=True)
        pb = pb[idx]
        ps = ps[idx]
        matched = torch.zeros(gb.shape[0], dtype=torch.bool, device=device)
        tp = []
        fp = []
        for i in range(pb.shape[0]):
            ious = box_iou_matrix(pb[i:i+1], gb).squeeze(0)  # (G,)
            max_iou, max_j = ious.max(0)
            if max_iou >= iou_thres and not matched[max_j]:
                matched[max_j] = True
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)
        tp = torch.tensor(tp, device=device).cumsum(0)
        fp = torch.tensor(fp, device=device).cumsum(0)
        recalls = tp / (gb.shape[0] + 1e-7) # gb - ground truth boxes
        precisions = tp / (tp + fp + 1e-7)
        # AP (trapezoidal) – simplu
        ap = torch.trapz(precisions, recalls) # curba de precizie 
        ap_list.append(ap)
    if len(ap_list) == 0:
        return 0.0
    return torch.stack(ap_list).mean().item()

from typing import Dict, List, Tuple
import torch

LEVEL_THRESHOLDS = {
    "P3": 0,     # <= 64
    "P4": 64,    # 64 - 128
    "P5": 128,   # 128 - 256
    "P6": 256    # >256
}

LEVEL_ORDER = ["P3", "P4", "P5", "P6"]
STRIDES = {
    "P3": 8,
    "P4": 16,
    "P5": 32,
    "P6": 64
}


def select_level(m: float) -> str:

    if m <= 64:
        return "P3"
    elif m <= 128:
        return "P4"
    elif m <= 256:
        return "P5"
    else:
        return "P6"


def build_targets(batch_boxes: List[torch.Tensor],
                  batch_classes: List[torch.Tensor],
                  img_size: int,
                  device,
                  num_classes: int):

    targets = {}
    for lvl in LEVEL_ORDER:
        targets[lvl] = {
            "batch_idx": [],
            "x_idx": [],
            "y_idx": [],
            "tbox": [],
            "cls": [],
            "mask_ids": []
        }

    for b, (boxes_n, cls_n) in enumerate(zip(batch_boxes, batch_classes)):
        if boxes_n.numel() == 0:
            continue
        boxes_pix = boxes_n.clone()
        boxes_pix[:, 0] *= img_size
        boxes_pix[:, 1] *= img_size
        boxes_pix[:, 2] *= img_size
        boxes_pix[:, 3] *= img_size

        for i, (cx, cy, w, h) in enumerate(boxes_pix):
            m = max(w.item(), h.item())
            lvl = select_level(m)
            stride = STRIDES[lvl]
            gx = cx / stride
            gy = cy / stride
            gi = int(gx)
            gj = int(gy)
            tx = gx - gi
            ty = gy - gj
            tw = (w / stride).clamp(min=1e-4)
            th = (h / stride).clamp(min=1e-4)

            targets[lvl]["batch_idx"].append(b)
            targets[lvl]["x_idx"].append(gi)
            targets[lvl]["y_idx"].append(gj)
            targets[lvl]["tbox"].append([tx, ty, tw, th])
            targets[lvl]["cls"].append(int(cls_n[i].item()))
            targets[lvl]["mask_ids"].append(i)

    # convertește în tensori
    for lvl in LEVEL_ORDER:
        if len(targets[lvl]["tbox"]) == 0:
            # tensori goi
            targets[lvl]["batch_idx"] = torch.tensor([], dtype=torch.long, device=device)
            targets[lvl]["x_idx"] = torch.tensor([], dtype=torch.long, device=device)
            targets[lvl]["y_idx"] = torch.tensor([], dtype=torch.long, device=device)
            targets[lvl]["tbox"] = torch.zeros((0, 4), dtype=torch.float, device=device)
            targets[lvl]["cls"] = torch.tensor([], dtype=torch.long, device=device)
            targets[lvl]["mask_ids"] = []
        else:
            targets[lvl]["batch_idx"] = torch.tensor(targets[lvl]["batch_idx"], dtype=torch.long, device=device)
            targets[lvl]["x_idx"] = torch.tensor(targets[lvl]["x_idx"], dtype=torch.long, device=device)
            targets[lvl]["y_idx"] = torch.tensor(targets[lvl]["y_idx"], dtype=torch.long, device=device)
            targets[lvl]["tbox"] = torch.tensor(targets[lvl]["tbox"], dtype=torch.float, device=device)
            targets[lvl]["cls"] = torch.tensor(targets[lvl]["cls"], dtype=torch.long, device=device)

    return targets

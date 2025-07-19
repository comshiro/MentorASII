# val.py
import argparse
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from utils.data_loader import YoloDetSegDataset, collate_fn
from utils.targets import STRIDES
from utils.metrics import ap50
from utils.loss import box_cxcywh_to_xyxy

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="config/yolov8_sidewalk.yaml")
    ap.add_argument("--data_list", type=str, default="data/splits/val.txt")
    ap.add_argument("--img_dir", type=str, default="data/images/val")
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="cuda:0")
    return ap.parse_args()

def load_cfg(path):
    with open(path,"r") as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    cfg = load_cfg(args.cfg)
    num_classes = len(cfg["data"]["names"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    from models.yolo import YOLOSegModel
    model = YOLOSegModel(cfg["model"], num_classes=num_classes).to(device)
    ckpt = torch.load(args.weights, map_location=device) # ckpt - checkpoint
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = YoloDetSegDataset("data", split="val", img_size=args.imgsz, augment=False)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1, collate_fn=collate_fn)

    all_pred_boxes = []
    all_pred_scores = []
    all_pred_classes = []
    all_gt_boxes = []
    all_gt_classes = []

    with torch.no_grad():
        for images, batch_boxes, batch_classes, paths in dl:
            images = images.to(device)
            out = model(images)
            preds = out["preds"]

            # batchuri multiple de 2
            # reconstrucție box-uri brute per nivel
            for lvl, pred in preds.items():
                stride = STRIDES[lvl]
                box = pred["box"][0]  # (4,H,W)
                cls_logits = pred["cls"][0]  # (C,H,W)
                H, W = box.shape[-2], box.shape[-1]
                # grid
                ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
                xs = xs.to(device)
                ys = ys.to(device)
                tx, ty, tw, th = box[0], box[1], box[2], box[3]
                cx = (xs + tx).float() * stride
                cy = (ys + ty).float() * stride
                bw = (tw * stride).clamp(min=1e-3)
                bh = (th * stride).clamp(min=1e-3)
                # flatten
                cx = cx.view(-1)
                cy = cy.view(-1)
                bw = bw.view(-1)
                bh = bh.view(-1)
                cls_logits = cls_logits.view(num_classes, -1).transpose(0,1)  # (Ncells, C)
                scores, cls_ids = torch.max(torch.sigmoid(cls_logits), dim=1)

                topk = min(50, scores.shape[0])
                vals, idxs = torch.topk(scores, topk)
                cx_sel = cx[idxs]
                cy_sel = cy[idxs]
                bw_sel = bw[idxs]
                bh_sel = bh[idxs]
                cls_sel = cls_ids[idxs]
                # convertim la xyxy
                x1 = cx_sel - bw_sel/2
                y1 = cy_sel - bh_sel/2
                x2 = cx_sel + bw_sel/2
                y2 = cy_sel + bh_sel/2
                boxes_xyxy = torch.stack([x1,y1,x2,y2], dim=1)

                all_pred_boxes.append(boxes_xyxy.cpu())
                all_pred_scores.append(vals.cpu())
                all_pred_classes.append(cls_sel.cpu())

            # GT (convertim din normalized xywh)
            gt_b = batch_boxes[0]
            gt_c = batch_classes[0]
            if gt_b.numel() > 0:
                cx = gt_b[:,0] * args.imgsz
                cy = gt_b[:,1] * args.imgsz
                w  = gt_b[:,2] * args.imgsz
                h  = gt_b[:,3] * args.imgsz
                x1 = cx - w/2
                y1 = cy - h/2
                x2 = cx + w/2
                y2 = cy + h/2
                all_gt_boxes.append(torch.stack([x1,y1,x2,y2], dim=1).cpu())
                all_gt_classes.append(gt_c.cpu())
            else:
                all_gt_boxes.append(torch.zeros((0,4)))
                all_gt_classes.append(torch.zeros((0,), dtype=torch.long))

    # Concat
    pred_boxes = torch.cat(all_pred_boxes, dim=0)
    pred_scores = torch.cat(all_pred_scores, dim=0)
    pred_classes = torch.cat(all_pred_classes, dim=0)
    gt_boxes = torch.cat(all_gt_boxes, dim=0)
    gt_classes = torch.cat(all_gt_classes, dim=0)

    from utils.metrics import ap50
    ap = ap50(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes)
    print(f"AP@0.5 simplu: {ap:.4f}")

if __name__ == "__main__":
    main()

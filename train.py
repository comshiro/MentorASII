# train.py
import argparse
import yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import time

from models.yolo import YOLOSegModel
from utils.data_loader import YoloDetSegDataset, collate_fn
from utils.targets import build_targets
from utils.loss import DetectionSegLoss

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="config/yolov8_sidewalk.yaml")
    ap.add_argument("--data_list", type=str, default="data/splits/train.txt")
    ap.add_argument("--img_dir", type=str, default="data/images/train")
    ap.add_argument("--val_list", type=str, default="data/splits/val.txt")
    ap.add_argument("--val_img_dir", type=str, default="data/images/val")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--save_dir", type=str, default="experiments/debug_run")
    ap.add_argument("--debug", type=int, default=0)
    return ap.parse_args()

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    cfg = load_cfg(args.cfg)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_classes = len(cfg["data"]["names"])

    # Model
    model = YOLOSegModel(cfg["model"], num_classes)
    print("Model params:", sum(p.numel() for p in model.parameters())/1e6, "M")

    # Dataset
    train_ds = YoloDetSegDataset("data", split="train", img_size=args.imgsz, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=2, collate_fn=collate_fn)
    loss_fn = DetectionSegLoss(num_classes=num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scaler = GradScaler(enabled=True) 

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(args.epochs):
        t0 = time.time()
        running = 0.0
        for it, (images, batch_boxes, batch_classes, paths) in enumerate(train_loader):
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                outputs = model(images)  # {"preds":{...}, "proto":...}
                targets = build_targets(batch_boxes, batch_classes, args.imgsz, device, num_classes)
                loss, parts = loss_fn(outputs["preds"], targets, outputs["proto"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            if args.debug and it >= args.debug - 1:
                break

            if it % 10 == 0:
                print(f"[Epoch {epoch+1}/{args.epochs}] Iter {it} loss={loss.item():.4f} "
                      f"(box={parts['loss_box']:.3f} cls={parts['loss_cls']:.3f})")

        avg_loss = running / max(1, (it + 1))
        dt = time.time() - t0
        print(f"Epoch {epoch+1} done. Avg loss={avg_loss:.4f} time={dt:.1f}s")

        ckpt_path = save_dir / f"epoch_{epoch+1}.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch+1}, ckpt_path)

    print("Antrenare terminată.")

if __name__ == "__main__":
    main()


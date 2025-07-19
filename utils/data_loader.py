
from pathlib import Path
import torch
from torch.utils.data import Dataset
import cv2, numpy as np

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

class YoloDetSegDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=640, augment=False):

        self.root = Path(root_dir)
        self.split = split
        self.img_dir = self.root / "images" / split
        self.lab_dir = self.root / "labels" / split
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Lipsește folderul imagini: {self.img_dir}")
        if not self.lab_dir.exists():
            print(f"[WARN] Folder labels nu există încă: {self.lab_dir}")

        self.img_paths = sorted([p for p in self.img_dir.rglob("*")
                                 if p.suffix.lower() in VALID_EXTS])
        if len(self.img_paths) == 0:
            raise RuntimeError(f"Nu s-au găsit imagini în {self.img_dir}")

        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.img_paths)

    def _read(self, path):
        im = cv2.imread(str(path))
        if im is None:
            raise FileNotFoundError(path)
        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    def _resize_pad(self, img):
        h, w = img.shape[:2]
        s = self.img_size
        scale = min(s/h, s/w)
        nh, nw = int(h*scale), int(w*scale)
        resized = cv2.resize(img, (nw, nh), cv2.INTER_LINEAR)
        canvas = np.zeros((s, s, 3), dtype=np.uint8)
        canvas[:nh, :nw] = resized
        return canvas

    def _load_labels(self, img_path):
        lab = self.lab_dir / (img_path.stem + ".txt")
        boxes = []
        classes = []
        if lab.exists():
            with lab.open() as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    classes.append(cls)
                    boxes.append([cx, cy, w, h])
        return (torch.tensor(boxes, dtype=torch.float32),
                torch.tensor(classes, dtype=torch.long))

    def __getitem__(self, idx):
        p = self.img_paths[idx]
        img = self._read(p)
        img = self._resize_pad(img)
        boxes, classes = self._load_labels(p)
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        return {
            "image": img,
            "boxes": boxes,
            "classes": classes,
            "path": str(p)
        }

def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    boxes  = [b["boxes"] for b in batch]
    classes= [b["classes"] for b in batch]
    paths  = [b["path"] for b in batch]
    return images, boxes, classes, paths

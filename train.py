from ultralytics import YOLO

# Load a small pretrained segmentation model (auto-downloads first time)
model = YOLO("yolo11n-seg.pt")

# Train on your trains dataset
model.train(
    data="data.yaml",   # created by prepare_dataset.py
    epochs=100,
    imgsz=640,
    batch=8,
    device="cpu",       # you have the CPU build of torch
    project="runs",
    name="train_seg",
)
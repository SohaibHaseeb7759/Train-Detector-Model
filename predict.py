from ultralytics import YOLO

# Load your trained model
model = YOLO(r"runs\segment\runs\train_seg\weights\best.pt")

# Run prediction on a validation image (not seen during training)
results = model.predict(
    source=r"dataset\images\val\train_001.jpg",
    save=True,          # saves an annotated copy with the mask drawn
    conf=0.25,          # confidence threshold
    project="runs",
    name="predict",
    exist_ok=True,
)

# Print what it found
for r in results:
    n = 0 if r.masks is None else len(r.masks)
    print(f"\nTrains detected: {n}")
    if r.boxes is not None:
        for i, box in enumerate(r.boxes):
            conf = float(box.conf[0])
            print(f"  Train {i+1}: confidence = {conf:.2%}")
    print(f"\nAnnotated image saved in: runs\\predict\\")

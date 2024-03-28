from ultralytics import YOLO

model = YOLO("yolov8l-seg.pt")

results = model.train(
        batch=13,
        device="cuda",
        data="/home/kai/Documents/solar/15/data.yaml",
        epochs=180,
        imgsz=640,
    )
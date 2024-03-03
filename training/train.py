from ultralytics import YOLO

model = YOLO("yolov8m-seg.pt")

results = model.train(
        batch=18,
        device="cuda",
        data="/home/kai/Documents/solar/data.yaml",
        epochs=100,
        imgsz=640,
    )
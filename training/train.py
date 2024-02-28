from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt")

results = model.train(
        batch=8,
        device="cpu",
        data="/home/kai/Desktop/dataset/data.yaml",
        epochs=7,
        imgsz=640,
    )
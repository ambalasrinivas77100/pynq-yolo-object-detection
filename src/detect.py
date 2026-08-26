from ultralytics import YOLO

model = YOLO("models/yolo11n.pt")

results = model.predict(
    source="bus.jpg",
    conf=0.5,
    save=True
)

print("Object detection completed successfully.")
from ultralytics import YOLO
import time
import os

model = YOLO("models/yolo11n.pt")

image_folder = "dataset/test"

images = [
    os.path.join(image_folder, file)
    for file in os.listdir(image_folder)
    if file.lower().endswith((".jpg", ".jpeg", ".png"))
]

total_time = 0
total_images = 0

for image in images:
    start = time.time()

    model.predict(
        source=image,
        conf=0.5,
        verbose=False
    )

    end = time.time()

    total_time += end - start
    total_images += 1

average_time = total_time / total_images
fps = 1 / average_time

print("CPU Dataset Benchmark")
print("----------------------------")
print("Images tested:", total_images)
print("Average processing time:", round(average_time * 1000, 2), "ms")
print("Approximate FPS:", round(fps, 2))
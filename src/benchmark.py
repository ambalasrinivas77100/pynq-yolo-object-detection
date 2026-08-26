from ultralytics import YOLO
import time

model = YOLO("models/yolo11n.pt")

image = "bus.jpg"

runs = 10

times = []

for i in range(runs):
    start = time.time()

    model.predict(
        source=image,
        conf=0.5,
        verbose=False
    )

    end = time.time()

    times.append(end - start)

average_time = sum(times) / len(times)
fps = 1 / average_time

print("CPU Benchmark")
print("--------------------")
print("Number of runs:", runs)
print("Average inference time:", round(average_time * 1000, 2), "ms")
print("Approximate FPS:", round(fps, 2))
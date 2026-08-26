from ultralytics import YOLO
from pathlib import Path
from collections import Counter

model = YOLO("models/yolo11n.pt")

dataset_path = Path("dataset/test")

total_counts = Counter()

image_files = []

for ext in ["*.jpg", "*.jpeg", "*.png"]:
    image_files.extend(dataset_path.glob(ext))

print("\nOBJECT DETECTION RESULTS")
print("=" * 50)

for image in image_files:

    results = model.predict(
        source=str(image),
        imgsz=640,
        conf=0.25,
        verbose=False
    )

    counts = Counter()

    for result in results:
        for cls in result.boxes.cls:
            class_id = int(cls)
            class_name = model.names[class_id]

            counts[class_name] += 1
            total_counts[class_name] += 1

    print(f"\nImage: {image.name}")

    if counts:
        for name, count in counts.items():
            print(f"  {name}: {count}")

        print(f"  Total objects: {sum(counts.values())}")
    else:
        print("  No objects detected.")

print("\n")
print("=" * 50)
print("TOTAL OBJECT COUNTS")
print("=" * 50)

if total_counts:
    for name, count in total_counts.most_common():
        print(f"{name}: {count}")

    print("-" * 50)
    print(f"Total objects detected: {sum(total_counts.values())}")
else:
    print("No objects detected.")
import os
import shutil
import json
import glob
from ultralytics import YOLO
import numpy as np

test_dir = r"c:\Users\Kirthiggan\OneDrive\Documents\Rockmelon Leaf Disease\P&D Rock Melon\P&D Rock Melon\test"
coco_json = os.path.join(test_dir, "_annotations.coco.json")

yolo_test_dir = os.path.join(os.getcwd(), "yolo_test")
images_dir = os.path.join(yolo_test_dir, "images", "val")
labels_dir = os.path.join(yolo_test_dir, "labels", "val")

if os.path.exists(yolo_test_dir):
    shutil.rmtree(yolo_test_dir)

os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

with open(coco_json, "r") as f:
    coco = json.load(f)

img_lookup = {img["id"]: img for img in coco["images"]}
actual_paths = {}
for root, _, files in os.walk(test_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            actual_paths[file] = os.path.join(root, file)

for ann in coco["annotations"]:
    img_id = ann["image_id"]
    cat_id = ann["category_id"] - 1
    if cat_id < 0:
        continue
    bbox = ann["bbox"]
    img = img_lookup[img_id]
    img_filename = img["file_name"]
    dw = 1.0 / img["width"]
    dh = 1.0 / img["height"]
    x = bbox[0] + bbox[2] / 2.0
    y = bbox[1] + bbox[3] / 2.0
    w = bbox[2]
    h = bbox[3]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    label_path = os.path.join(labels_dir, os.path.splitext(os.path.basename(img_filename))[0] + ".txt")
    with open(label_path, "a") as f:
        f.write(f"{cat_id} {x} {y} {w} {h}\n")

for img_id, img in img_lookup.items():
    img_filename = img["file_name"]
    base = os.path.basename(img_filename)
    if img_filename in actual_paths:
        shutil.copy(actual_paths[img_filename], os.path.join(images_dir, base))
    elif base in actual_paths:
        shutil.copy(actual_paths[base], os.path.join(images_dir, base))

yaml_content = f"""
path: {yolo_test_dir}
train: images/val
val: images/val

names:
  0: aphid
  1: fungus
  2: leaf miner
  3: normal
  4: unknown
"""
with open(os.path.join(yolo_test_dir, "dataset.yaml"), "w") as f:
    f.write(yaml_content)

model = YOLO("best.pt")
metrics = model.val(data=os.path.join(yolo_test_dir, "dataset.yaml"), plots=True)

print("---------------- EVALUATION RESULTS ----------------")
print(f"mAP50 (Accuracy Proxy): {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
mean_p = np.mean(metrics.box.p)
mean_r = np.mean(metrics.box.r)
mean_f1 = np.mean(metrics.box.f1)
print(f"Precision: {mean_p:.4f}")
print(f"Recall: {mean_r:.4f}")
print(f"F1-Score: {mean_f1:.4f}")
print("----------------------------------------------------")
print(f"Save dir: {metrics.save_dir}")

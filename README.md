# Dental Quadrant Detection — YOLOv8

A YOLOv8-based object detection model that localizes and classifies the
dental quadrant in panoramic and periapical X-ray images. This model is
one component of a larger dental insurance fraud detection pipeline.

## Task Definition

Given a dental radiograph, the model predicts which of the four oral
quadrants is present in the image:

| Label | Quadrant     |  Range    |
|-------|--------------|-----------|
| UR    | Upper Right  |   1-8     |
| UL    | Upper Left   |           |
| LL    | Lower Left   |           |
| LR    | Lower Right  |           |

Each prediction includes a bounding box localizing the region and a
confidence score.

## Model Architecture

- Base: YOLOv8 (Ultralytics)
- Task: Object Detection
- Input: RGB image (resized to 640x640)
- Output: Bounding boxes with class label and confidence score

## Dataset

| Split | Images |
|-------|--------|
| Train | —      |
| Val   | —      |
| Test  | —      |

- Source: [Panoramic and periapical dental X-rays](https://huggingface.co/datasets/ibrahimhamamci/DENTEX)
- Annotation format: json .txt (normalized xywh)
- Classes: 4 (UR, UL, LL, LR)

## Training

yolo detect train \
  data=quadrant.yaml \
  model=yolov8n.pt \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  project=runs/quadrant \
  name=exp1

quadrant.yaml:

  path: ./datasets/quadrant
  train: images/train
  val:   images/val
  test:  images/test
  nc: 4
  names: ["UR", "UL", "LL", "LR"]

## Inference

from ultralytics import YOLO
import cv2

model = YOLO("quadrant_model.pt")
image = cv2.imread("xray.jpg")
results = model(image, conf=0.25)[0]

for box in results.boxes:
    class_id   = int(box.cls.item())
    confidence = float(box.conf.item())
    bbox       = box.xyxy[0].tolist()
    label      = results.names[class_id]
    print(f"Quadrant: {label} | Conf: {confidence:.2f} | BBox: {bbox}")


## Requirements

ultralytics>=8.0.0
opencv-python>=4.9.0
numpy>=1.26.0

## Part of

This model is used alongside:
- condition-detection: detects pathological findings
- tooth-number-detection: identifies tooth number within quadrant (1–8)

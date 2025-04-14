# Assignment 2: Object Detection using YOLOv5 on COCO Subset

##  Overview
This assignment demonstrates how to evaluate pretrained object detection models (YOLOv5s and YOLOv5m) using a small subset of the COCO 2017 validation dataset. The focus is on:

- Evaluating detection performance (mAP and recall)
- Analyzing inference time across different batch sizes
- Visualizing detections with bounding boxes
- Interpreting detection metrics for real-time/edge applications

---

## 1. Dataset Preparation

- **Dataset**: COCO 2017 val subset
- **Subset size**: 32 images only (due to memory constraints)
- **Annotations**: Uses `instances_val2017.json` or `person_keypoints_val2017.json`

Folder structure:

## 3. Evaluation Pipeline

###  Batch-wise Evaluation
- Supported batch sizes: **1, 4, 8**
- Metrics computed for each configuration

###  Metrics Collected
- `mAP@0.5`
- `mAP@0.5:0.95`
- `Recall@100`
- `Inference time per image`
- `Total inference time`

###  Tools Used
- `torchmetrics.detection.MeanAveragePrecision`
- `torchvision.ops.nms`
- `matplotlib` for visual output and graph plotting

---

## 4. Visualizations

Each evaluated image generates a result with:
- Ground-truth and predicted boxes
- Class labels and confidence scores

Saved in folders:

---

## 5. Plots and Analysis

Each evaluation run generates 3 main graphs:
- **mAP@0.5 vs Inference Time per Image**
- **mAP@0.5:0.95 vs Total Inference Time**
- **Recall@100 vs Inference Time per Image**

Saved as:
- `map50_vs_inference_time.png`
- `map_vs_total_inference_time.png`
- `mar100_vs_inference_time.png`

These graphs help analyze tradeoffs between **accuracy and latency** across batch sizes.

---

## 6. Code Highlights

### Custom `COCODataset` class handles:
- COCO annotation parsing
- Single image + label loading
- Resizing and tensor transforms

### `YOLOv5Evaluator` handles:
- Batch inference
- NMS + confidence threshold filtering
- Ground-truth alignment and metric updates
- Final visual results saving

---

## 7. Learning Outcomes

Students will:
- Understand the structure of a COCO-format dataset
- Evaluate pretrained models on real-world data
- Analyze inference speed and accuracy tradeoffs
- Visualize detections for reporting and debugging
- Compare performance of YOLOv5m vs YOLOv5s

---

## 8. References

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [TorchMetrics](https://torchmetrics.readthedocs.io/en/stable/detection/mean_average_precision.html)
- [COCO Dataset](https://cocodataset.org/#home)
- [StackOverflow](https://stackoverflow.com/)

> All code is thoroughly commented for clarity and educational use.

---

## 9. Appendix: Running Instructions

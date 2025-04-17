# Assignment: ImageNet Classification Using Pretrained Models and Quantization

##   Overview
This assignment is about the practical use of large-scale pretrained models (like ResNet, VGG, ViT, etc.) trained on ImageNet, and shows how to adapt them for research on edge computing use cases. The focus is on:

- Using pretrained models for benchmarking results
- Performing model quantization (dynamic and static)
- Working with a subset of ImageNet validation subset

All experiments are designed to give hands-on experience with model optimization and deployment-aware benchmarking.

---

## 1.  Using Pretrained ImageNet Models for Benchmarking Results 

###   Supported Architectures (Use any six of them)
- ResNet18, 34, 50, 101, 152
- VGG11-19, MobileNet, DenseNet, ViT, Swin, ConvNeXt, GoogLeNet

###   Evaluation Strategy
- Use single sample from each class (from ImageNet ILSVRC2012 val subset)
- Compare accuracy and total inference time

###   Metrics
- Accuracy across 500 classes
- Batch Size
- Inference time per model

###   Use Case
This helps to asses the tradeoffs of each model architecture for edge devices, real-time processing, and memory-bound applications.

---

## 2.  Quantization Analysis

Quantization reduces model size and improves inference time, especially useful for mobile/IoT.

###   Dynamic Quantization
- Uses `torch.quantization.quantize_dynamic`
- Target: `Conv2d` modules
- Compare batch sizes (1, 4, 8, 16)
- Plot Accuracy vs Inference Time
- Output: `batch_size_analysis_combined.png`

###   Static Quantization 
- Uses FX Graph Mode Quantization APIs
- Includes both:
  - Default config mapping (`fbgemm`, `qnnpack`, `x86`) (use any one of them) 
  - Custom `QConfig` with `HistogramObserver` + `MinMaxObserver`
- Calibrates on 1 batch of real data
- Measures impact of quantization engine on infrence time and accuracy

###   Evaluation Output
- Accuracy and inference plotted
- Output image: `static_quant_analysis.png/static_quant_analysis.png`
- Helps identify best quantization config for target hardware

---

## 3.  Working with  ImageNet ILSVRC2012 val Subset

Due to the full dataset size (~156GB), a smaller subset of the ImageNet **validation set** was selected:

- Uses 500 folders (each representing a class)
- Each folder contains 1 sample image (JPEG/PNG)
- Paths like: `C:/Users/.../val_subset/<class_folder>/image.jpg`

###  Dataset Notes
- Registered account at [ImageNet.org](https://image-net.org)
- Downloaded official `ILSVRC2012` val set
- Subset created to make experiments lightweight and reproducible
- You are encouraged to download the full dataset for future work

---

## 4. Results and Graphs

Plots generated:
- `batch_size_analysis_combined.png` (dynamic quantization)
- `static_quant_analysis.png` (static quantization)

Key metrics included:
- Inference time vs accuracy
- Comparison between batch sizes
- Model-specific tradeoffs

---

## 5. Learning Outcomes

By completing this assignment, one will:
- Understand how to use pretrained models from the PyTorch ecosystem
- Apply quantization strategies for efficient deployment
- Conduct benchmarks with real-world performance metrics
- Prepare models for edge/IoT applications

---

## 6. References & Tools

- **PyTorch Hub**: https://pytorch.org/vision/stable/models.html
- **Pytorch Quatization**: https://pytorch.org/docs/stable/quantization.html
- **StackOverflow**: For debugging and scripting references

> All code includes inline comments to explain functionality and logic for educational purposes.

# Assignment: Object Detection using YOLOv5 on COCO Subset

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
- **Annotations**: Uses `instances_val2017.json` 

## 2. Evaluation Pipeline

###  Batch-wise Evaluation
- Supported batch sizes: **1, 4, 8**
- Metrics computed for each configuration

###  Metrics Collected
- `mAP`
- `Recall`
- `Inference time per image`
- `Total inference time`

###  Tools Used
- `torchmetrics.detection.MeanAveragePrecision`
- `torchvision.ops.nms`
---

## 4. Visualizations

Each evaluated image generates a result with:
- Ground-truth and predicted boxes
- Class labels and confidence scores

Saved in folders:
- visualizations_batch_1/
- visualizations_batch_4/
- visualizations_batch_8/
---

## 5. Plots and Analysis

Each evaluation run generates 2 main graphs:
- **mAP@0.5 vs Inference Time per Image**

Saved as:
- `map50_vs_inference_time.png`
- `map_vs_total_inference_time.png`
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

One will:
- Understand the structure of a COCO-format dataset
- Evaluate pretrained models on real-world data
- Analyze inference speed and accuracy tradeoffs
- Visualize detections for reporting and debugging
- Compare performance of YOLOv5m vs YOLOv5s

---

## 8. References

- **Ultralytics YOLOv5**: https://github.com/ultralytics/yolov5
- **TorchMetrics**: https://torchmetrics.readthedocs.io/en/stable/detection/mean_average_precision.html)
- **COCO Dataset**: https://cocodataset.org/#home
- **StackOverflow**: https://stackoverflow.com/

> All code is thoroughly commented for clarity and reading.



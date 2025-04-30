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
This helps to assess the tradeoff of each model architecture for edge devices, real-time processing, and memory-bound applications.

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
  - Default config mapping (`fbgemm`, `qnnpack`) (use any one of them) 
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

# Assignment: Object Detection using YOLOv5 on COCO (Common Objetc in Context) Subset

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

# Assignment: Fine-Tuning Pretrained ResNet18 on CIFAR (Canadian Institute For Advanced Research)-100

## Overview

This assignment demonstrates how to leverage **transfer learning** and **fine-tuning** by adapting a **pretrained ResNet18** model (trained on ImageNet-1000 classes) to classify images from the **CIFAR-100** dataset (100 classes).

The focus is on:
- Using pretrained models efficiently for custom tasks
- Fine-tuning classification heads
- Selectively unfreezing layers for better adaptation
- Benchmarking model performance on a new dataset

---

## 1. Dataset Preparation

- **Dataset**: CIFAR-100
- **Classes**: 100 classes (animals, vehicles, etc.)
- **Image Size**: Resized to (224 × 224) to match ResNet18's input requirements
- **Automatic Download**: No need to manually download, the script handles it.

---

## 2. Model Adaptation and Fine-tuning

### Pretrained Model:
- **Model**: ResNet18 (from torchvision)
- **Pretraining**: Trained on ImageNet (1000 classes)

### Fine-tuning Strategy Via Transfer Learning::
- Earlier layers capture **general visual patterns**.
- Only the final layers need retraining to adapt to **specific categories** in CIFAR-100.
- This saves computation and speeds up convergence.
- Freeze all convolutional layers initially except for:
- Last residual block (`layer4`)
- Final fully connected layer (`fc`)
- For Example, replace the final classification layer from:
model.fc = nn.Linear(512, 1000)  # ImageNet has 1000 classes
to
model.fc = nn.Linear(512, 100)  # CIFAR has 100 classes
---

## 3. Training Pipeline

- **Optimizer**: SGD (learning rate = 0.01 (try this as well:1e-6), momentum = 0.9, weight decay = 5e-4)
- **Scheduler**: StepLR (reduce LR by half every 10 epochs)
- **Loss Function**: CrossEntropyLoss
- **Epochs**: 100

---

## 4. Observed Results
      
- Best Validation Accuracy | ~71%         |
- Training will Plateaud around 20-25 epochs 

---

### Important:

- With only `layer4` and `fc` trainable, accuracy **plateaus at ~71%**.
- To **improve further**:
-- Unfreeze **layer3** (and optionally layer2).
-- Allow mid-level features to adapt better to the CIFAR-100 dataset.

---

## 5. Fine tuning Strategies for Higher Accuracy

-	The provided code will give you ~71% validation accuracy.
-	By unfreezing more layers and adjusting optimizers/schedulers, you can fine tune to reach higher accuracy (75-85% range).
- To move beyond the initial ~71%:

- **Unfreeze more layers**:

for name, param in model.named_parameters():
    if 'layer3' in name or 'layer4' in name or 'fc' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
-	Use AdamW Optimizer:
-	AdamW with learning rate 1e-4 improves stability.
-	Switch Scheduler to CosineAnnealingLR:
-	Smooth learning rate decay helps avoid plateaus.
-	Train longer:
-	Increase total epochs to 150+ if needed.
-	Simplify Data Augmentation:
-	Use mild augmentations like RandomHorizontalFlip, RandomRotation (small).
-	**Note**: The code related to this section is not included in the assignment script; it is intended for self learning purposes to enhance skills and 
   knowledge
 	
---

## 6. Learning Outcomes
- Understand how to **adapt pretrained models** for new, custom datasets by modifying classification layers.
- Learn the **concept and practical application of transfer learning**, where knowledge from a large dataset (ImageNet) is applied to a smaller, different dataset (CIFAR-100).
- Gain experience in **fine-tuning pretrained networks**, including techniques like selective layer freezing, optimizer switching, and learning rate scheduling.
- Develop skills to **analyze model architecture** and decide strategically which layers to freeze or unfreeze for maximum performance.
- Understand the **trade-offs between training speed, convergence, and generalization** when leveraging transfer learning.
- Understand how to **benchmark model performance** (accuracy, loss, etc.) systematically after fine-tuning.
- Understand the importance of **data augmentation**, **optimizer selection**, and **learning rate management** in successful model transfer and adaptation.

---

## 7. References
-	**PyTorch Tansfer Learning Tutorial** :https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
-	**Transfer Learning CS231n Stanford**:https://cs231n.github.io/transfer-learning/
- **StackOverflow**: https://stackoverflow.com/
  
**Note**: Use the provided Jupyter Notebook file `finetune_resnet18_cifar100.ipynb` to run and reproduce result described in this assignment. Additionally, all code is thoroughly commented for clarity and 
  reading.



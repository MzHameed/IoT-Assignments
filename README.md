# Assignment 1: ImageNet Classification Using Pretrained Models, Fine-Tuning, and Quantization

##   Overview
This assignment guides through the practical use of large-scale pretrained models (like ResNet, VGG, ViT, etc.) trained on ImageNet, and shows how to adapt them for research or edge computing use cases. The focus is on:

- Fine-tuning pretrained models on custom datasets (CIFAR-100)
- Using pretrained models as it is for quick benchmarking
- Performing model quantization (dynamic and static)
- Working with a manageable ImageNet validation subset

All experiments are designed to give hands-on experience with transfer learning, model optimization, and deployment-aware benchmarking.

---

## 1.   Fine-Tuning a Pretrained Model on CIFAR-100

###   Goal
Demonstrate how to fine-tune a pretrained ImageNet model (ResNet-18) on a 100-class dataset (CIFAR-100). This is ideal for conducting research who want to leverage pretrained models without training from scratch.

###   Key Techniques
- Transfer Learning: Freeze early layers, train only the final block
- Modify final FC layer: `model.fc = nn.Linear(512, 100)` 
- Apply strong data augmentation (`RandomErasing`, `ColorJitter`)
- Use SGD optimizer with learning rate scheduling

###  Outcome
- Final model saved to `best_resnet18_cifar100_9001.pth`
- Prints accuracy for each epoch
- Highlights how one can reuse pretrained models for their own class setup (The aforementioned script is designed for the CIFAR-100 dataset, but to utilize a custom dataset, simply change the final fully connected (FC) layer according to the number of classes in the target dataset)

---

## 2.  Using Pretrained ImageNet Models for Benchmarking

###   Supported Architectures
- ResNet18, 34, 50, 101, 152
- VGG11-19, MobileNet, DenseNet, ViT, Swin, ConvNeXt, GoogLeNet

###   Evaluation Strategy
- Use single sample from each class (from ImageNet val subset)
- Compare accuracy and total inference time

###   Metrics
- Accuracy across 500 classes
- Batch Size
- Inference time per model

###   Use Case
This helps to asses the tradeoffs of each model architecture for edge devices, real-time processing, and memory-bound applications.

---

## 3.  Quantization Analysis

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
  - Default config mapping (`fbgemm`, `qnnpack`, `x86`)
  - Custom `QConfig` with `HistogramObserver` + `MinMaxObserver`
- Calibrates on 1 batch of real data
- Measures impact of quantization engine on speed and accuracy

###   Evaluation Output
- Accuracy and inference plotted
- Output image: `static_quant_analysis.png`
- Helps identify best quantization config for target hardware

---

## 4.  Working with ImageNet Subset

Due to the full dataset size (~156GB), a smaller subset of the ImageNet **validation set** was selected:

- Uses 500 folders (each representing a class)
- Each folder contains 1 sample image (JPEG/PNG)
- Paths like: `C:/Users/.../val_subset/<class_folder>/image.jpg`

###  Dataset Notes
- Registered account at [ImageNet.org](https://image-net.org)
- Downloaded official `ILSVRC2012` val set
- Subset created to make experiments lightweight and reproducible
- Students are encouraged to download the full dataset for future work

---

## 5. Results and Graphs

Plots generated:
- `batch_size_analysis_combined.png` (dynamic quantization)
- `static_quant_analysis.png` (static quantization)

Key metrics included:
- Inference time vs accuracy
- Comparison between batch sizes
- Model-specific tradeoffs

---

## 6. Learning Outcomes

By completing this assignment, students will:
- Understand how to use pretrained models from the PyTorch ecosystem
- Fine-tune for new tasks with custom class counts
- Apply quantization strategies for efficient deployment
- Conduct benchmarks with real-world performance metrics
- Prepare models for edge/IoT applications

---

## 7. References & Tools

- **PyTorch Hub**: https://pytorch.org/vision/stable/models.html
- **Ultralytics YOLOv5 GitHub**: https://github.com/ultralytics/yolov5 (used in Assignment 2)
- **StackOverflow**: For debugging and scripting references

> All code includes inline comments to explain functionality and logic for educational purposes.


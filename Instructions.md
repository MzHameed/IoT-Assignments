#  Instructions for Running Assignments 

This guide explains how to set up Python environments, run scripts for **ImagNet Classification** and **COCO Object Detection Assignmnets**, and visualize results using **Jupyter Notebook** via **Anaconda Navigator**.

---

## General Guidelines

- Place **all script files and datasets inside your user directory** (`C:/Users/YourName/IoT/`)
- **Update dataset paths** inside each script before running
- Make sure Python version = 3.9 (compatible with the provided environments)

---

##  Step 1: Download & Install Anaconda Navigator

Download: **Anaconda Official Site**: https://www.anaconda.com/download/success

After installation, open **Anaconda Prompt** (Windows) or terminal (Linux):

---

## Step 2: Create Conda Environments

###  Environment for ImagNet Classification Assignment
```
(base) > conda create -n torch_quant python=3.9 -y
(base) > conda activate torch_quant
(torch_quant) > pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
(torch_quant) > pip install numpy==1.23.5 pandas matplotlib
```

###  Check versions:
```python
>>> import torch
>>> import numpy as np
>>> print(torch.__version__)  # 1.13.1+cpu
>>> print(np.__version__)     # 1.23.5
>>> print(torch.backends.quantized.supported_engines)
# ['none', 'onednn']
```

---

###  Environment for Coco Datset Object Detection Assignment
```
(base) > conda deactivate
(base) > conda create -n torch_quant1 python=3.9 -y
(base) > conda activate torch_quant1
(torch_quant1) > pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
(torch_quant1) > pip install numpy==1.23.5 pandas matplotlib opencv-python torchmetrics pycocotools
```

---

## Step 3: Launch Jupyter Notebook

- Open **Anaconda Navigator**
- On the **top-left**, click the environment dropdown (default: `base (root)`)
- Select one of the created environments:
  - `torch_quant` → for **ImagNet Classification Assignment**
  - `torch_quant1` → for **Coco Dataset Object Detetcion Assignment**
- Install **Jupyter Notebook** from the list if not already installed
- Click **Launch** to open Jupyter in your browser

---

## Step 4: Run the Assignments

### ImageNet Classification Assignmnent
- Launch Jupyter using `torch_quant`
- Run `ImagNet_Calssification_Via_Pre-trained_Model.ipynb`
- Run `ImagNet_Calssification_Via_Quantized_Pre-trained_Model.ipynb`
- View the results in the console

### COCO Object Detection Assignment
- Launch Jupyter using `torch_quant1`
- Run `COCO_dataset_Object_Detection.ipynb`
- Visual output saved in:
  - `visualizations_batch_1/`
  - `visualizations_batch_4/`
  - `visualizations_batch_8/`

> Please refer to the attached .png images for additional clarity if needed

---

##  Troubleshooting

- Errors about already installed packages? Try restarting **Anaconda Navigator**
- If notebook doesn't launch, close browser and click **Launch** again
- Always ensure you're in the **correct environment** before starting

---

##  Summary

| Assignment         | Conda Env      | Location                  | Tool             |
|--------------------|----------------|---------------------------|------------------|
| Classification     | `torch_quant`  | IOT/Assignment1_Classification | Jupyter Notebook |
| Object Detection   | `torch_quant1` | IOT/Assignment2_ObjectDetection | Jupyter Notebook |

---

**You're Ready!**

Once you've run both assignments, you will:
- Understand pretrained model use for classification/detection
- Visualize performance metrics and tradeoffs
- Use results for edge/IoT deployment tasks



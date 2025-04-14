#  Instructions for Running Assignment 1 & Assignment 2

This guide explains how to set up Python environments, run scripts for **Assignment 1 (Classification)** and **Assignment 2 (Object Detection)**, and visualize results using **Jupyter Notebook** via **Anaconda Navigator**.

---

## General Guidelines

- Place **all script files and datasets inside your user directory** (e.g., `C:/Users/YourName/IOT/`)
- **Update dataset paths** inside each script before running
- Make sure Python version = 3.9 (compatible with the provided environments)

---

##  Step 1: Download & Install Anaconda Navigator

Download: **Anaconda Official Site**: https://www.anaconda.com/download/success

After installation, open **Anaconda Prompt** (Windows) or terminal (Linux):

---

## Step 2: Create Conda Environments

###  Environment for Assignment 1: Classification
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

###  Environment for Assignment 2: Object Detection
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
  - `torch_quant` → for **Assignment 1**
  - `torch_quant1` → for **Assignment 2**
- Install **Jupyter Notebook** from the list if not already installed
- Click **Launch** to open Jupyter in your browser

---

## Step 4: Run the Assignments

### Assignment 1: Classification
- In Jupyter, navigate to the `IOT/Assignment1_Classification` folder
- Open the `.ipynb` or `.py` script
- Click **Run** to simulate results
- Visual outputs like `static_quant_analysis.png` will be generated

### Assignment 2: Object Detection
- Launch Jupyter using `torch_quant1`
- Navigate to `IOT/Assignment2_ObjectDetection`
- Run the object detection script
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



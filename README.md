
# DiffWaferNet: A Lightweight Differential Attention Network for Wafermap Defect Classification

## Description
This work focuses on detecting wafer map defects using deep learning models. It proposes a lightweight CNN-based framework called **DiffWaferNet**, enhanced by a **Differential Attention Mechanism**, and addresses data imbalance using a **Conditional Variational Autoencoder (CVAE)**. The goal is to accurately classify rare and common defect patterns in semiconductor manufacturing, with minimal model size for real-time industrial applications.

## Dataset Information
- **Source**: WM-811K Wafer Map Dataset (publicly available via [Kaggle](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map))

- **Citation**:  
  Wu, Ming-Ju, Jyh-Shing R. Jang, and Jui-Long Chen.  “Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets.” IEEE Transactions on Semiconductor Manufacturing, vol. 28, no. 1, Feb. 2015, pp. 1–12. [DOI: 10.1109/TSM.2014.2364237](https://doi.org/10.1109/TSM.2014.2364237)

- **Classes**:
  - Center
  - Edge-Loc
  - Edge-Ring
  - Donut
  - Loc
  - Random
  - Scratch
  - Near-Full

- **Preprocessing Steps**:
  - Wafer maps resized to `32x32x3`
  - Class-wise channel encoding
  - Removal of unlabeled and irrelevant samples
  - Predefined train/test split from dataset
  - Normalization and data augmentation

## Code Information
The code includes the following:
1. **Data Loading**: Load `.pkl` formatted dataset generated from Kaggle data.
2. **Preprocessing**: Resize and encode wafer maps, apply transformations.
3. **Synthetic Data Generation**: CVAE used to generate realistic class-consistent samples for minority classes.
4. **Dataset Merging**: Merge synthetic data with real training data.
5. **Model Architecture**:
   - CNN Backbone
   - Differential Attention Layer
   - Fully connected layers with dropout and softmax
6. **Model Training**:
   - Optimizer: Adam
   - Loss: Categorical Crossentropy
   - Early stopping & LR scheduling
7. **Model Evaluation**:
   - Confusion matrix
   - Accuracy, Precision, Recall, F1-Score
8. **Ablation Study**: Comparisons across CNN-only, CNN with Standard Attention, and DiffWaferNet

## Usage Instructions

### Google Colab Setup
To run this on Google Colab:

1. **Upload Code and Dataset**:
```python
from google.colab import files
uploaded = files.upload()
```

2. **Install Dependencies**:
```bash
!pip install -r requirements.txt
```

3. **Load the Dataset**:
```python
import pandas as pd
import pickle
with open("data/LSWMD.pkl", "rb") as f:
    data = pickle.load(f)
```

4. **Train the Model**:
Open and run the notebook `code/diffwafernet.ipynb`


## Methodology

### Data Processing
- **EDA**: Class distribution inspection, wafer pattern diversity etc.
- **Missing Values**: only labeled classes used
- **Encoding**: Wafer maps encoded as 3-channel RGB-style arrays based on defect localization patterns.



### Synthetic Data with CVAE
- A Conditional Variational Autoencoder (CVAE) was trained using class-conditioned inputs to generate synthetic wafer maps.
- One-hot encoded labels were used to guide generation per defect class.

### DiffWaferNet Architecture
- **CNN Backbone**: 3 Conv layers (32, 64, 128 filters)
- **Differential Attention**: Learnable lambda weights across attention heads
- **Classifier**: GAP → Dense(32) → Dropout → Dense(8, Softmax)

## Evaluation Metrics
- Accuracy
- Macro Precision, Recall, F1-Score
- Confusion Matrix
- Saliency Maps

## Materials & Methods

- **Operating System**: Windows 11 (local) / Linux (Kaggle environment)
- **Hardware**:
  - Kaggle Notebook: Dual NVIDIA Tesla T4 GPUs
  - Local: Intel i7-1165G7 CPU, 16 GB RAM
- **Tools & Libraries**:
  - Jupyter Notebook (Kaggle Kernel)
  - TensorFlow (v2.10+)
  - NumPy, Matplotlib, scikit-learn

### Evaluation:
- **Accuracy**: Overall classification success
- **Confusion Matrix**: Class-specific insight
- **ROC**: Not used (multi-class setting)
- **Precision/Recall/F1**: Key for class imbalance

## Example Code
```python
import numpy as np
from tensorflow.keras.models import load_model

# Load wafer data
with open("LSWMD.pkl", "rb") as f:
    data = pickle.load(f)

# Example inference
model = load_model("diffwafernet.keras")
preds = model.predict(data["X_test"])
print("Predicted Class:", np.argmax(preds, axis=1))
```

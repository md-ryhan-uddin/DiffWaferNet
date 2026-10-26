# DiffWaferNet: A Lightweight Differential Attention-Based Network for Wafermap Defect Classification


## Description

DiffWaferNet is a lightweight deep learning framework for wafermap defect classification in semiconductor manufacturing. It combines a convolutional neural network (CNN), a differential attention mechanism, and a Conditional Variational Autoencoder (CVAE) to address classification challenges caused by severe class imbalance.

The framework aims to improve recognition of common and underrepresented wafer defect patterns while maintaining computational efficiency for potential industrial applications.

## Dataset Information

### Dataset Source

- **Dataset:** WM-811K Wafer Map Dataset
- **Source:** [WM-811K on Kaggle](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)

### Dataset Citation

Wu, M.-J., Jang, J.-S. R., & Chen, J.-L. (2015). Wafer map failure pattern recognition and similarity ranking for large-scale data sets. *IEEE Transactions on Semiconductor Manufacturing, 28*(1), 1–12. https://doi.org/10.1109/TSM.2014.2364237

### Defect Classes

The eight defect classes considered in the classification framework are:

1. Center
2. Edge-Loc
3. Edge-Ring
4. Donut
5. Loc
6. Random
7. Scratch
8. Near-Full

### Preprocessing

- Resize wafer maps to `32 × 32 × 3`.
- Encode wafer maps using three-channel representations.
- Remove unlabeled and irrelevant samples.
- Use the predefined train/test split where applicable.
- Apply normalization and data augmentation as configured in the experiments.

## Code Information

The implementation covers the following stages:

1. **Data Loading:** Load the wafermap dataset from a serialized `.pkl` file.
2. **Preprocessing:** Resize, encode, and transform wafer maps.
3. **Synthetic Data Generation:** Use a CVAE to generate class-conditioned samples for minority classes.
4. **Dataset Integration:** Combine synthetic samples with real training data.
5. **Model Architecture:**
   - CNN backbone
   - Differential attention layer
   - Fully connected classification layers
   - Dropout and softmax output
6. **Model Training:**
   - Optimizer: Adam
   - Loss: Categorical cross-entropy
   - Early stopping and learning-rate scheduling
7. **Model Evaluation:**
   - Confusion matrix
   - Accuracy, precision, recall, and F1-score
8. **Ablation Study:** Compare CNN-only, CNN with standard attention, and DiffWaferNet configurations.

## Methodology

### 1. Data Processing

Exploratory data analysis is used to inspect class distributions and wafer defect patterns. The preprocessing pipeline includes selecting labeled classes, converting wafer maps into three-channel representations, resizing maps to a consistent input shape, and preparing training, validation, and test sets.

### 2. Synthetic Data Generation with CVAE

A Conditional Variational Autoencoder generates synthetic wafer maps conditioned on defect classes. Class labels guide the generation of class-consistent samples, helping mitigate severe class imbalance. Synthetic samples are integrated into the training data to improve the representation of underrepresented classes.

### 3. DiffWaferNet Architecture

**CNN Backbone**
- Three convolutional layers with 32, 64, and 128 filters.
- Spatial feature extraction from wafermap representations.

**Differential Attention**
- Computes two attention distributions.
- Combines them using a learnable differential weighting mechanism.
- Emphasizes informative features for defect classification.

**Classification Head**
- Global Average Pooling (GAP)
- Dense layer with 32 units
- Dropout
- Eight-class softmax output

## Evaluation Metrics

- **Accuracy:** Overall classification performance.
- **Macro Precision:** Average precision across defect classes.
- **Macro Recall:** Average recall across defect classes.
- **Macro F1-score:** Balanced assessment of precision and recall across classes.
- **Confusion Matrix:** Class-specific classification performance.
- **Saliency Maps:** Visual interpretation of regions relevant to model predictions.

Macro-averaged metrics are particularly useful when defect classes have substantially different sample counts.

## Usage Instructions

### Google Colab Setup

#### 1. Upload the Required Files

Upload the notebook, dataset, and any other required project files to Google Colab. To upload local files manually:

```python
from google.colab import files

uploaded = files.upload()
```

#### 2. Install Dependencies

If a `requirements.txt` file is available, install dependencies:

```bash
pip install -r requirements.txt
```

In a Colab notebook cell, use `!pip install -r requirements.txt`.

#### 3. Load the Dataset

```python
import pickle

with open("data/LSWMD.pkl", "rb") as f:
    data = pickle.load(f)
```

Update the file path to match your dataset location.

#### 4. Train the Model

Open and run the notebook:

```text
code/diffwafernet.ipynb
```

Ensure that dataset paths, dependencies, and model settings are correct before training.

## Example Inference

This example illustrates a general inference workflow. Adapt the input format and preprocessing to match the saved model.

```python
import pickle
import numpy as np
import tensorflow as tf

# Load the dataset
with open("LSWMD.pkl", "rb") as f:
    data = pickle.load(f)

# Load the trained model
model = tf.keras.models.load_model(
    "diffwafernet.keras",
    compile=False
)

# Predict wafer defect classes
predictions = model.predict(data["X_test"])

# Obtain the predicted class index
predicted_classes = np.argmax(predictions, axis=1)

print("Predicted classes:", predicted_classes)
```

**Note:** This example assumes that the serialized dataset contains an `X_test` entry and that the saved model accepts the supplied input shape. Apply the same preprocessing used during training before inference.

## Materials and Methods

### Software and Libraries

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

### Computing Environment

The original project documentation identifies the following environments:

- **Operating systems:** Windows 11 and Linux-based notebook environments.
- **Notebook environment:** Kaggle Notebook.
- **Reported GPU configuration:** Dual NVIDIA Tesla T4 GPUs.
- **Local hardware:** Intel Core i7-1165G7 CPU with 16 GB RAM.

Actual training requirements depend on dataset size, batch size, model configuration, and synthetic data generation settings.

## Reproducibility

For reproducible experiments:

1. Use the same dataset version and preprocessing configuration.
2. Maintain consistent training, validation, and test partitions.
3. Record random seeds and model hyperparameters.
4. Document the CVAE configuration and synthetic sample-generation procedure.
5. Evaluate all model variants using the same test set.
6. Report class-wise metrics alongside overall accuracy.

## License and Attribution

Check the repository's license and the published article's license before redistributing the implementation or associated materials. Publication of a paper does not automatically establish an open-source license for its code.

When using this method or discussing its findings in academic work, cite the published article:

Uddin et al. (2026), *PeerJ Computer Science, 12*, e3947. https://doi.org/10.7717/peerj-cs.3947


## Cite This Work

**APA-style citation**

Uddin, M. R., Islam, M. S., Farid, F. A., Uddin, J., & Mansor, S. B. (2026). DiffWaferNet: A lightweight differential attention-based network for wafermap defect classification. *PeerJ Computer Science, 12*, e3947. https://doi.org/10.7717/peerj-cs.3947

**BibTeX**

```bibtex
@article{uddin2026diffwafernet,
  author  = {Uddin, Md Ryhan and
             Islam, Md Saiful and
             Farid, Fahmid Al and
             Uddin, Jia and
             Mansor, Sarina Binti},
  title   = {DiffWaferNet: A lightweight differential attention-based
             network for wafermap defect classification},
  journal = {PeerJ Computer Science},
  volume  = {12},
  pages   = {e3947},
  year    = {2026},
  doi     = {10.7717/peerj-cs.3947},
  url     = {https://doi.org/10.7717/peerj-cs.3947}
}
```

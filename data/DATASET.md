# WM-811K Dataset

**Dataset Name:** WM-811K (Wafer Map 811K Dataset)

**Original Source Paper:**
Wu, M.-J., Jang, J.-S. R., & Chen, J.-L. (2015).
*Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets.*
IEEE Transactions on Semiconductor Manufacturing.

**DOI (Dataset Reference Paper):**
https://doi.org/10.1109/TSM.2014.2364237

**Public Distribution:**
The dataset is publicly accessible via Kaggle:
https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map

---

## Description

The WM-811K dataset is a large-scale real semiconductor manufacturing dataset containing **811,457 wafer maps** used for wafer defect pattern recognition and machine learning research.

Each wafer map represents spatial pass/fail information of dies on semiconductor wafers.

### Defect Classes

* Center
* Donut
* Edge-Loc
* Edge-Ring
* Loc
* Random
* Scratch
* Near-Full
* None (Normal wafer)

Only a subset of wafers contains labeled defect patterns; the dataset is commonly used for supervised, semi-supervised, and anomaly detection research.

---

## License

The dataset is redistributed publicly for research and educational purposes.
Please refer to the Kaggle dataset page and the original publication for licensing and usage terms.

---

## Dataset Usage in This Repository

Dataset files are **not included** in this repository due to size limitations.

To reproduce experiments:

1. Download the dataset from Kaggle.
2. Extract the files locally.
3. Place the dataset in the project data directory as described in the setup instructions.

---

## Citation

If you use this project or dataset, please cite:

Wu, M.-J., Jang, J.-S. R., & Chen, J.-L. (2015).
*Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets.*
IEEE Transactions on Semiconductor Manufacturing.
https://doi.org/10.1109/TSM.2014.2364237

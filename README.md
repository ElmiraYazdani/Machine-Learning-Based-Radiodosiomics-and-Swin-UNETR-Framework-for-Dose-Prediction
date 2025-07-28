# Machine-Learning-Based-Radiodosiomics-and-Swin-UNETR-Framework-for-Dose-Prediction
This repository presents a radiodosiomics framework for personalized [¹⁷⁷Lu]Lu-PSMA-617 RLT in mCRPC. 
It includes feature selection and ML models using clinical, radiomic, and dosiomic features, plus nnU-Net and Swin UNETR DL models with SSL to predict Monte Carlo–based dose rate maps.

## 🔍 Machine Learning Pipeline
See [`radiodosiomics_ML.ipynb`](./radiodosiomics_ML.ipynb) for the full pipeline including:
- Feature selection (RFE, Boruta, LASSO, Mutual Information, and Elastic Net)
- Model training and evaluation
- Integration of clinical, radiomic, and dosiomic features


## 🔍 Deep Learning Pipeline




# Install Dependencies
Install dependencies using:
```bash
pip install -r requirements.txt
```
# Citation


# Acknowledgement
Models Implantation and SSL Pipeline are based on [MONAI](https://github.com/Project-MONAI/MONAI) and [This](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR) repository.

# Machine-Learning-Based-Radiodosiomics-and-Swin-UNETR-Framework-for-Dose-Prediction
This repository presents a radiodosiomics framework for personalized [¹⁷⁷Lu]Lu-PSMA-617 RLT in mCRPC. 
It includes feature selection and ML models using clinical, radiomic, and dosiomic features, plus nnU-Net and Swin UNETR DL models with SSL to predict Monte Carlo–based dose rate maps.

## 🤖 Machine Learning Pipeline 🤖
See [`radiodosiomics_ML.ipynb`](./radiodosiomics_ML.ipynb) for the full pipeline including:
- Feature selection (RFE, Boruta, LASSO, Mutual Information, and Elastic Net)
- Model training and evaluation
- Integration of clinical, radiomic, and dosiomic features


## 🧠 Deep Learning Pipeline 🧠
## 🧹 1. Preprocessing
Before pretraining and fine-tuning, data (PET and CT images) should be preprocessed:
```bash
python preprocess.py --in_dir=<Input-directory(PET and CT)> --out_dir=<Output-directory>
```

## 🏋️ 2. Pre-Training
Pre-Train Swin UNETR encoder on unlabeled data
```bash
python main.py --exp=<Experiment Name> --in_channels=2 --data_dir=<Data-Path> --json_list=<Json List Path> \
--lr=6e-6 --lrdecay --batch_size=<Batch Size> --num_steps=<Number of Steps>
```

## 3. 🛠️ Fine-Tuning
Fine-Tuning Swin UNETR on labeled data:
```bash
python main.py --exp=<Experiment Name> --data_dir=<Data-Path> --json_list=<Json List Path> --in_channels=2 --out_channels=1 \
--pretrained_model_name=<Pretrained Encoder Name> --batch_size=<Batch Size> --max_epochs=<Epochs> --use_ssl_pretrained \
--ssl_pretrained_path=<Pretrained Model Path> --use_checkpoint
```

## 📊 4. Evaluation
Evaluating Swin UNETR
```bash

python test.py --pretrained_dir=<Pretrained Model Path> --data_dir=<Data-Path> --exp_name=<Experiment Name> \
--json_list=<Json List Path> --pretrained_model_name=<Pretrained Model Name> --save
```


## ⚙️ Install Dependencies
Install dependencies using:
```bash
pip install -r requirements.txt
```
## 📚 Citation


## Acknowledgement
Models Implantation and SSL Pipeline are based on [MONAI](https://github.com/Project-MONAI/MONAI) and [This](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR) repository.

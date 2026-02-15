🏜️ Robust Semantic Segmentation for Offroad Desert Environments
Duality AI – Offroad Semantic Scene Segmentation Challenge

Team: SS CODERS

📌 Overview

This project presents a semantic segmentation model trained on synthetic desert environment data generated using Falcon digital twin environments.

The objective was to build a model capable of accurately segmenting offroad desert scenes and generalizing effectively to unseen desert environments.

The solution includes:

End-to-end training pipeline

Data preprocessing & augmentation

Validation evaluation

Optimized inference pipeline

Reproducible training scripts

🧠 Problem Statement

Train a semantic segmentation model to classify each pixel into predefined desert environment classes and ensure generalization to unseen test environments.

Classes
Class ID	Label
100	Trees
200	Lush Bushes
300	Dry Grass
500	Dry Bushes
550	Ground Clutter
600	Flowers
700	Logs
800	Rocks
7100	Landscape
10000	Sky
🏗️ Project Structure
├── train.py
├── test.py
├── model/
├── data/
├── configs/
├── runs/
├── README.md

⚙️ Installation
1️⃣ Clone Repository
git clone <your-repo-url>
cd <repo-name>

2️⃣ Create Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

3️⃣ Install Dependencies
pip install -r requirements.txt

🚀 Training
python train.py


Training pipeline includes:

RGB normalization

Mask label mapping

Data augmentation

Weighted loss handling

Validation IoU monitoring

Early stopping

🧪 Evaluation
python test.py


The evaluation script:

Loads trained model (.pkl)

Performs inference on unseen desert environment

Generates segmentation outputs

Computes IoU (if ground truth available)

📊 Performance

Due to runtime/environment constraints, final training and evaluation scripts could not be executed at submission time. Therefore, quantitative metrics (IoU and inference speed) are not included in this repository.

However, the implemented architecture, augmentation pipeline, optimization strategy, and validation workflow are designed to achieve strong generalization performance on unseen desert environments.

🛠️ Techniques Used

Data Augmentation (flip, rotation, brightness, crop)

Class-weighted loss for imbalance

Adam optimizer

Early stopping & LR scheduling

Efficient inference pipeline

🔬 Challenges & Solutions
Class Imbalance

Handled using weighted loss and targeted augmentation.

Overfitting

Mitigated using augmentation, dropout, and early stopping.

📦 Submission Includes

Trained Model (.pkl)

train.py

test.py

Configuration files

Report

README

💡 Future Improvements

Domain adaptation techniques

Transformer-based segmentation models

Attention-based architectures

Self-supervised pretraining

Real-time deployment optimization
# Pneumonia-Detection-using-X---ray

---

# 🩺 Pneumonia Detection from Chest X-ray Images

This project uses deep learning to classify chest X-ray images as either **"NORMAL"** or **"PNEUMONIA"**. It includes:

* ✅ A **custom CNN** model built from scratch
* 🚀 A **transfer learning model** using pre-trained **VGG16**
* 🖼️ An **interactive UI** to upload and classify your own X-ray images
* 📈 Evaluation metrics to compare model performance

---

## 📂 Dataset

**Chest X-Ray Images (Pneumonia)** from Kaggle
📎 [Dataset Link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

* Total Images: **5,863**
* Organized into: `train/`, `test/`, and `val/` folders
* Classes: `NORMAL` and `PNEUMONIA`

---

## 🧠 Models Used

### 1️⃣ Custom CNN (Built from Scratch)

* 3 Convolutional Layers
* ReLU activation, MaxPooling
* Flatten → Dense → Output
* Lightweight baseline model

### 2️⃣ VGG16 Transfer Learning

* Pre-trained on **ImageNet**
* Used as a feature extractor
* Fine-tuned top layers for pneumonia detection
* Improved **accuracy and stability**

---

## 💻 How to Use (Google Colab)

### ▶️ [Open in Google Colab](https://colab.research.google.com/)

> The easiest way to try this project!

### 🔑 Step 1: Add Your Kaggle API Key

1. Download `kaggle.json` from your [Kaggle account](https://www.kaggle.com/account).
2. Upload it to the notebook when prompted.

### 🧲 Step 2: Download the Dataset

```python
!pip install -q kaggle
!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
```

### 🔗 Step 3: Mount Google Drive (for dataset & model storage)

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 🏗️ Step 4: Train the Models

Run the training cells to train both the Custom CNN and the VGG16-based model.
Evaluation metrics and plots will be shown after training.

---

## 🚀 Using the Pre-trained Model (Without Retraining)

If you just want to **make predictions** without training:

1. Mount Google Drive
2. Run the cell that **loads the saved model (.h5 file)**
3. Use the **interactive UI** to upload an image and see the prediction

```python
from tensorflow.keras.models import load_model
model = load_model('/content/drive/MyDrive/pneumonia_model.h5')
```

---

## 📊 Results Summary

| Model          | Accuracy | Precision | Recall | F1 Score |
| -------------- | -------- | --------- | ------ | -------- |
| Custom CNN     | 85%      | 84%       | 86%    | 85%      |
| VGG16 Transfer | **93%**  | 92%       | 94%    | **93%**  |

---
Dataset is available under [Kaggle’s Terms of Use](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

Note :- 
1. To run the model you have to train the ephocs in your system.
2. This is just a prototype model not recommended for medical and industrial Use. 

---

Let me know if you want me to:

* Add Colab badges
* Auto-link the `.ipynb` notebook
* Write deployment instructions (Hugging Face Spaces or Streamlit)

Just say the word.

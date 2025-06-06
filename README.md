
# Real-Time Emotion Recognition with PyTorch 🎭

This project performs real-time **emotion recognition** using a webcam feed and a trained **ResNet18 model** in PyTorch.

## 📁 Project Structure

```
emotion-recognition/
├── train.py              # Script to train the emotion recognition model
├── predict.py            # Real-time webcam inference
├── emotion_model.pth     # Trained model weights (optional for inference)
├── .gitignore            # To exclude dataset and cache files from Git
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation (you are here)
```

## 🧠 Model Info

- Uses **ResNet18** pretrained on ImageNet
- Custom classification head for emotion categories
- Trained using `torchvision.datasets.ImageFolder` format

## 💡 Dataset Format

Organize the dataset like:

```
dataset/
├── train/
│   ├── happy/
│   ├── sad/
│   └── ... (other classes)
├── test/
│   ├── happy/
│   ├── sad/
│   └── ...
```

> Dataset not included in this repository (ignored by `.gitignore`)

## 🚀 How to Use

### 🔧 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 🏋️‍♂️ 2. Train the Model
```bash
python train.py
```

### 🎥 3. Run Webcam Inference
```bash
python predict.py
```

Make sure your camera is connected and accessible.

## 📦 Requirements

- torch
- torchvision
- opencv-python
- scikit-learn
- wandb

You can install them via:

```bash
pip install torch torchvision opencv-python scikit-learn wandb
```

## 📌 Notes

- You can modify `predict.py` to use your **secondary camera** (change `VideoCapture(0)` to `VideoCapture(1)`).
- The `.gitignore` ensures large datasets or generated files aren't committed.



---

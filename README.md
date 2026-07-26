# 🛡️ Fake Detection System

An AI-powered deepfake image detection system built using **TensorFlow**, **Keras**, and **EfficientNetB3**. The project provides a complete pipeline for dataset preprocessing, model training, image prediction, and a Flask-based web application for real-time inference.

---

## 📌 Overview

With the rapid advancement of AI-generated media, distinguishing authentic images from manipulated ones has become increasingly important. This project leverages transfer learning with **EfficientNetB3** to classify images as **Real** or **Fake** with high accuracy.

---

## ✨ Features

- 📂 Dataset preprocessing and organization
- 🤖 Deepfake image generation utilities
- 🧠 Transfer Learning using EfficientNetB3
- 📈 Model training with TensorFlow/Keras
- 🔍 Image prediction and inference
- 🌐 Flask web application for easy deployment
- ⚡ Clean and modular project structure

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Model | EfficientNetB3 |
| Computer Vision | OpenCV |
| Web Framework | Flask |
| Frontend | HTML, CSS |
| Data Processing | NumPy, Pandas |

---

## 📁 Project Structure

```text
fake-detection-system/
│
├── dataset/                 # Training dataset
├── models/                  # Saved trained models
├── src/
│   ├── train.py             # Model training
│   └── predict.py           # Prediction script
│
├── static/                  # CSS, JS, Images
├── templates/
│   └── index.html           # Web interface
│
├── app.py                   # Flask application
├── generate_fakes.py        # Fake image generation
├── flatten_dataset.py       # Dataset preprocessing
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/saadirfan15/fake-detection-system.git
```

### 2. Navigate to the project

```bash
cd fake-detection-system
```

### 3. Create a virtual environment

Windows

```bash
python -m venv .venv
```

Activate it

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

Run:

```bash
python src/train.py
```

The trained model will be saved inside the **models/** directory.

---

## 🔍 Predict an Image

```bash
python src/predict.py
```

---

## 🌐 Run the Web Application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 🧠 Model Architecture

- EfficientNetB3 (Pre-trained on ImageNet)
- Global Average Pooling
- Batch Normalization
- Dense Layers
- Dropout Regularization
- Softmax Output Layer

---

## 📊 Workflow

```text
Dataset
   │
   ▼
Data Preprocessing
   │
   ▼
Model Training
   │
   ▼
Saved Model
   │
   ▼
Prediction
   │
   ▼
Flask Web Application
```

---

## 📸 Screenshots

You can add screenshots of the web application here.

Example:

```
screenshots/
├── home.png
├── prediction.png
└── result.png
```

Then include:

```markdown
![Home](screenshots/home.png)

![Prediction](screenshots/prediction.png)
```

---

## 📈 Future Improvements

- Video deepfake detection
- Live webcam detection
- REST API integration
- Docker support
- Cloud deployment
- Model optimization for faster inference

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a new branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push the branch

```bash
git push origin feature-name
```

5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Saad Irfan**

- GitHub: https://github.com/saadirfan15

---

⭐ If you found this project helpful, consider giving it a **Star** on GitHub!
# 🛡️ Fake Detection System

An AI-powered **Deepfake Image Detection System** built using **TensorFlow**, **Keras**, and **EfficientNetB3**. This project provides a complete pipeline for dataset preprocessing, model training, image prediction, and a Flask-based web application for detecting whether an image is **Real** or **Fake**.

---

## 📖 Overview

The increasing use of AI-generated images has made it difficult to distinguish authentic content from manipulated media. This project leverages **transfer learning** with **EfficientNetB3** to build a robust deepfake detection model capable of classifying images into **Real** and **Fake** categories.

The project includes utilities for dataset preprocessing, fake image generation, model training, inference, and an easy-to-use web interface for predictions.

---

## ✨ Features

- 🧠 Deepfake image classification using EfficientNetB3
- 📂 Dataset preprocessing utilities
- 🎭 Fake image generation script
- 📈 Model training with TensorFlow/Keras
- 🔍 Predict real vs fake images
- 🌐 Flask web application
- 📁 Organized project structure
- ⚡ Easy to extend and customize

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Model | EfficientNetB3 |
| Computer Vision | OpenCV |
| Data Processing | NumPy, Pandas |
| Web Framework | Flask |
| Frontend | HTML, CSS |

---

# 📁 Project Structure

```text
fake-detection-system/
│
├── dataset/
│   ├── pkr_fake/
│   └── pkr_real/
│
├── models/
│   └── currency_model.h5
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── preprocess.py
│   └── utils.py
│
├── static/
│   ├── css/
│   └── uploads/
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── app.py
├── generate_fakes.py
├── flatten_dataset.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/saadirfan15/fake-detection-system.git
```

## 2. Navigate to the project

```bash
cd fake-detection-system
```

## 3. Create a virtual environment

### Windows

```bash
python -m venv .venv
```

Activate it

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Training the Model

Run the training script:

```bash
python src/train.py
```

After training, the model will be saved in the **models/** directory.

---

# 🔍 Predict an Image

Run:

```bash
python src/predict.py
```

The script will classify an input image as:

- ✅ Real
- ❌ Fake

---

# 🌐 Run the Web Application

Start the Flask server:

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

Upload an image through the web interface and view the prediction result.

---

# 🧠 Model Architecture

The model is based on **EfficientNetB3** with transfer learning.

Architecture:

- EfficientNetB3 (ImageNet Pretrained)
- Global Average Pooling
- Batch Normalization
- Dense (256, ReLU)
- Dropout (0.4)
- Dense (128, ReLU)
- Dropout (0.3)
- Softmax Output Layer

---

# 🔄 Workflow

```text
Raw Dataset
      │
      ▼
Dataset Preprocessing
      │
      ▼
Fake Image Generation
      │
      ▼
Model Training
      │
      ▼
Saved Model (.h5)
      │
      ▼
Prediction
      │
      ▼
Flask Web Application
```

---

# 📸 Application Preview

Add screenshots of your application inside a folder named:

```
screenshots/
```

Example:

```
screenshots/
├── home.png
├── upload.png
└── result.png
```

Then display them like:

```markdown
## Home Page

![Home](screenshots/home.png)

## Prediction Result

![Result](screenshots/result.png)
```

---

# 📌 Future Improvements

- 🎥 Deepfake video detection
- 📷 Live webcam inference
- ☁️ Cloud deployment
- 🐳 Docker support
- 📱 Responsive UI improvements
- ⚡ Faster inference using TensorFlow Lite
- 📊 Training metrics visualization

---

# 🤝 Contributing

Contributions are welcome.

1. Fork this repository.
2. Create a feature branch.

```bash
git checkout -b feature-name
```

3. Commit your changes.

```bash
git commit -m "Add new feature"
```

4. Push to your branch.

```bash
git push origin feature-name
```

5. Open a Pull Request.

---

# 📄 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

**Saad Irfan**

- GitHub: https://github.com/saadirfan15

---

## ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub. Your support helps motivate future improvements and makes the project more discoverable for others.
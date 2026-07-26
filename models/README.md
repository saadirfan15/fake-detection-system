# Fake Currency Detection

A computer vision project that uses deep learning to detect fake currency notes. The system employs transfer learning with ResNet50 for image classification and provides a web interface for easy prediction.

## Project Description

This project implements an AI-powered fake currency detection system using Convolutional Neural Networks (CNNs). It can classify currency images as either genuine ("REAL") or counterfeit ("FAKE") with high accuracy. The model is trained on a dataset of real and fake currency images and deployed as a Flask web application for user-friendly predictions.

The system preprocesses images, applies data augmentation, trains a ResNet50-based model, and provides both command-line and web-based prediction capabilities.

## Features

- **Image Preprocessing**: Automatic resizing, RGB conversion, and normalization
- **Data Augmentation**: Rotation, brightness adjustment, and horizontal flipping for robust training
- **Transfer Learning**: Uses pre-trained ResNet50 model for feature extraction
- **Binary Classification**: Distinguishes between real and fake currency
- **Web Interface**: Bootstrap-styled Flask app for easy image upload and prediction
- **Confidence Scoring**: Provides prediction confidence percentages
- **Suspicious Region Highlighting**: Draws bounding boxes on low-confidence predictions
- **Model Persistence**: Saves trained models for reuse

## Tech Stack

- **Programming Language**: Python 3.8+
- **Deep Learning Framework**: TensorFlow/Keras
- **Computer Vision**: OpenCV
- **Web Framework**: Flask
- **Data Processing**: NumPy, scikit-learn
- **Visualization**: Matplotlib
- **Object Detection**: Ultralytics YOLOv8 (for future enhancements)
- **Frontend**: Bootstrap 5

## Folder Structure

```
fake-currency-detection/
│
├── dataset/
│   ├── real/          # Real currency images
│   └── fake/          # Fake currency images
│
├── models/
│   ├── currency_model.h5    # Trained model file
│   └── app.py               # Flask web application
│
├── src/
│   ├── preprocess.py        # Data preprocessing and augmentation
│   ├── train.py             # Model training script
│   ├── predict.py           # Prediction script
│   └── utils.py             # Utility functions
│
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning)

### Step-by-Step Installation

1. **Clone or download the project**:
   ```bash
   git clone <repository-url>
   cd fake-currency-detection
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the dataset**:
   - Place real PKR currency images in `dataset/PKR_real/`
   - Place fake PKR currency images in `dataset/PKR_fake/`
   - Supported formats: JPG, JPEG, PNG

## Usage

### Training the Model

1. Ensure your dataset is properly organized in `dataset/PKR_real/` and `dataset/PKR_fake/`

2. Run the training script:
   ```bash
   python src/train.py
   ```

   This will:
   - Load and preprocess the data
   - Train EfficientNetB3 with a 2-phase fine-tuning schedule
   - Save the trained model to `models/currency_model.h5`
   - Display training curves (accuracy and loss)

### Running the Web App

1. Ensure the trained model exists at `models/currency_model.h5`

2. Start the Flask application:
   ```bash
   python app.py
   ```

3. Open your browser and navigate to `http://localhost:5000`

4. Upload a currency image and click "Detect" to get the prediction

### Command-Line Prediction

For single image prediction via command line:

```bash
python src/predict.py path/to/currency_image.jpg
```

This will output the prediction result and save an annotated image if the confidence is low.

## Expected Results

### Model Performance

- **Training Accuracy**: Typically 95-98% after 20 epochs
- **Validation Accuracy**: 90-95% depending on dataset quality
- **Test Accuracy**: 85-95% on unseen data

### Prediction Output

- **High Confidence (>80%)**: "REAL with 95.67% confidence" or "FAKE with 92.34% confidence"
- **Low Confidence (<80%)**: Same output + annotated image with red bounding box saved

### Web Interface Features

- Responsive Bootstrap design
- Drag-and-drop file upload
- Real-time prediction display
- Color-coded results (green for real, red for fake)
- Confidence percentage display

## Configuration

### Model Parameters

- **Input Size**: 224x224 pixels
- **Batch Size**: 32
- **Epochs**: 20
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy

### Data Augmentation

- Rotation: ±30 degrees
- Brightness: 50-150%
- Horizontal flip: Applied randomly

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Model Not Found**: Run training script first to generate `currency_model.h5`
3. **Dataset Errors**: Check that `dataset/real/` and `dataset/fake/` contain valid images
4. **Port Already in Use**: Change port in `app.py` if 5000 is occupied

### Performance Tips

- Use GPU for faster training (install TensorFlow-GPU)
- Increase epochs for better accuracy (monitor for overfitting)
- Ensure balanced dataset for optimal performance

## Future Enhancements

- Integration with YOLOv8 for region-specific fake detection
- Mobile app development
- Real-time video stream analysis
- Multi-currency support
- Advanced security features

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ResNet50 pre-trained weights from ImageNet
- TensorFlow/Keras for deep learning framework
- OpenCV for computer vision utilities
- Flask for web framework
- Bootstrap for UI styling
import cv2
import matplotlib.pyplot as plt
import os

def load_image(image_path):
    """
    Load an image from given path using OpenCV
    Return the image in RGB format
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def save_image(image, output_path):
    """
    Save a given image to the specified output path using OpenCV
    """
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img_bgr)

def plot_training_history(history):
    """
    Accept a Keras training history object
    Plot accuracy and loss curves (training vs validation)
    Save the plot as models/training_plot.png using matplotlib
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    output_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'training_plot.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Training plot saved to {output_path}")

def get_class_labels():
    """
    Return a dictionary: {0: "FAKE", 1: "REAL"}
    """
    return {0: "FAKE", 1: "REAL"}

def confidence_color(confidence):
    """
    Accept a confidence float (0.0 to 1.0)
    Return "green" if confidence >= 0.80
    Return "orange" if confidence >= 0.60
    Return "red" if below 0.60
    """
    if confidence >= 0.80:
        return "green"
    elif confidence >= 0.60:
        return "orange"
    else:
        return "red"

def log_prediction(image_path, prediction, confidence):
    """
    Log the prediction details
    """
    print(f"Image: {image_path}, Prediction: {prediction}, Confidence: {confidence:.2f}")
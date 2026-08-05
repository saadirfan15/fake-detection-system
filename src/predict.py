import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

def predict_currency(image_path):
    """
    Load the saved model, preprocess the input image,
    make prediction, and highlight suspicious regions if confidence < 80%.
    Returns prediction string and saves annotated image if applicable.
    """
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'currency_model.h5')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    custom_objects = {'InputLayer': tf.keras.layers.InputLayer}
    model = load_model(model_path, compile=False, custom_objects=custom_objects)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (300, 300))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    pred = model.predict(img_input)[0]
    label_index = int(np.argmax(pred))
    confidence = float(pred[label_index]) * 100
    label = "REAL" if label_index == 1 else "FAKE"

    if confidence < 80:
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, h), (0, 0, 255), 2)
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_annotated{ext}"
        cv2.imwrite(output_path, img)
        print(f"Annotated image saved to {output_path}")
    else:
        print("High confidence prediction, no annotation needed")

    return f"{label} with {confidence:.2f}% confidence"

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    result = predict_currency(image_path)
    print(result)

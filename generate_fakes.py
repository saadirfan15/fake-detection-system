import cv2
import numpy as np
import os

def create_fake_sample(img):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
    _, enc = cv2.imencode('.jpg', img, encode_param)
    img = cv2.imdecode(enc, 1)
    noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
    return cv2.add(img, noise)

input_folder = 'dataset/PKR_real'
output_folder = 'dataset/PKR_fake'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img = cv2.imread(os.path.join(input_folder, filename))
        if img is None:
            continue
        fake = create_fake_sample(img)
        cv2.imwrite(os.path.join(output_folder, f'fake_{filename}'), fake)

print("Fake images generated successfully.")

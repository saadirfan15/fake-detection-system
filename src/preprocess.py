import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def preprocess_data(dataset_path=None, augmentation_factor=1):
    """
    Load images from PKR_real and PKR_fake folders,
    preprocess them for EfficientNetB3, and return train/test splits.
    """
    if dataset_path is None:
        dataset_path = os.path.join(os.path.dirname(__file__), '..', 'dataset')

    images = []
    labels = []
    folder_paths = [
        (1, 'PKR_real'),
        (0, 'PKR_fake'),
    ]

    for label, relative_path in folder_paths:
        path = os.path.join(dataset_path, relative_path)
        if not os.path.exists(path):
            print(f"Warning: Path {path} does not exist, skipping...")
            continue

        for file in os.listdir(path):
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(path, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (300, 300))
                img = img / 255.0
                images.append(img)
                labels.append(label)

    if not images:
        raise ValueError("No images found in the dataset folders")

    images = np.array(images)
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        horizontal_flip=False
    )

    X_train_aug = []
    y_train_aug = []
    target_size = len(X_train) * augmentation_factor
    batch_size = 32

    for batch_x, batch_y in datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True):
        X_train_aug.extend(batch_x)
        y_train_aug.extend(batch_y)
        if len(X_train_aug) >= target_size:
            break

    X_train_aug = np.array(X_train_aug[:target_size])
    y_train_aug = np.array(y_train_aug[:target_size])

    return X_train_aug, X_test, y_train_aug, y_test

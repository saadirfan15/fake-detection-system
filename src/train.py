import importlib.util
import os
import shutil
import subprocess
import os
import tensorflow as tf
# ✅ Keras 3.x compatible imports
from keras.applications import EfficientNetB3
from keras import layers, Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.src.legacy.preprocessing.image import ImageDataGenerator

print(tf.config.list_physical_devices('GPU'))


def build_model(num_classes=2):
    base = EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_shape=(300, 300, 3)
    )
    base.trainable = False
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=output)
    return model, base


def main():
    dataset_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f'Dataset path not found: {dataset_dir}')

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        horizontal_flip=False,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(300, 300),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        seed=42
    )
    val_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(300, 300),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        seed=42
    )

    print('Class indices:', train_gen.class_indices)

    if train_gen.num_classes < 2:
        raise ValueError('Training directory must contain at least two classes: PKR_real and PKR_fake')

    model, base = build_model(num_classes=train_gen.num_classes)

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.3, patience=3),
        ModelCheckpoint('best_model.h5', save_best_only=True)
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print('Starting phase 1 training...')
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=5,
        callbacks=callbacks
    )

    base.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print('Starting phase 2 fine-tuning...')
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=callbacks
    )

    model_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'models', 'currency_model.h5'))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f'Model saved to {model_path}')


if __name__ == '__main__':
    main()

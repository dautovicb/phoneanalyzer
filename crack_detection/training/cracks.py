import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import confusion_matrix, classification_report

# config
base_path = 'dataset_updated'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# data loading

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(base_path, 'train'),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(base_path, 'valid'),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(base_path, 'test'),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False
)

# optimization

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# augmentation

data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomBrightness(factor=0.3),
    layers.RandomContrast(0.2),
    layers.RandomZoom(0.1),
])

# loading convnext tiny

base_model = tf.keras.applications.ConvNeXtTiny(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

# training

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Recall()]
)

print("\n--- Starting Training ---")
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# evaluation

def run_project_evaluation(model, history, test_ds):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy Evolution')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Evolution')
    plt.legend()
    plt.show()

    # Confusion Matrix
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_pred = (model.predict(test_ds) > 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Clean', 'Cracked'], yticklabels=['Clean', 'Cracked'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix: Cracked Phone Detection')
    plt.show()

    print("\n--- Final Test Metrics ---")
    print(classification_report(y_true, y_pred, target_names=['Clean', 'Cracked']))

run_project_evaluation(model, history, test_ds)

model.save('cracked_phone_detector_model.keras')
print("Model saved successfully to 'cracked_phone_detector_model.keras'")

import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models
keras = tf.keras
import matplotlib.pyplot as plt
import json
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=r"D:\\Code\\Code\\DataSetProject\\DataSet", help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout1', type=float, default=0.3)
    parser.add_argument('--dropout2', type=float, default=0.15)
    parser.add_argument('--freeze_epochs', type=int, default=5, help='Epochs to freeze base before fine-tuning')
    return parser.parse_args()

args = parse_args()
DATASET_DIR = args.data
img_height, img_width = 224, 224
batch_size = args.batch_size
epochs = args.epochs
learning_rate = args.lr
dropout1 = args.dropout1
dropout2 = args.dropout2
freeze_epochs = args.freeze_epochs

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Class names
class_names = train_ds.class_names
num_classes = len(class_names)
print("CLASS_NAME:", class_names)

# Save class_indices for inference
with open("class_indices.json", "w", encoding="utf-8") as f:
    json.dump({i: c for i, c in enumerate(class_names)}, f, ensure_ascii=False, indent=2)

# ======================

# 2. Data Augmentation (tăng cường mạnh hơn)
# ======================
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.20),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.15),
    layers.RandomBrightness(0.1),
    layers.RandomTranslation(0.10, 0.10),
    # Removed RandomCrop for less aggressive augmentation
])

# ======================
# 3. Chuẩn hóa & Tối ưu pipeline
# ======================
AUTOTUNE = tf.data.AUTOTUNE

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

train_ds = train_ds.map(
    lambda x, y: (preprocess_input(data_augmentation(x)), y),
    num_parallel_calls=AUTOTUNE
).cache().prefetch(AUTOTUNE)

val_ds = val_ds.map(
    lambda x, y: (preprocess_input(x), y),
    num_parallel_calls=AUTOTUNE
).cache().prefetch(AUTOTUNE)

# ======================

# 4. Sử dụng MobileNetV2 (Transfer Learning, fine-tune toàn bộ)
# ======================

# Progressive fine-tuning: freeze base for a few epochs, then unfreeze
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(dropout1),
    layers.Dense(128, activation='relu'),
    layers.Dropout(dropout2),
    layers.Dense(num_classes, activation='softmax')
])

# ======================


# 5. Compile Model (initial: freeze base, higher lr)
# ======================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ======================

# 6. Tính class_weight để giảm lệch class
from collections import Counter
labels = np.concatenate([y for x, y in train_ds], axis=0)
counter = Counter(labels)
total = sum(counter.values())
class_weight = {i: total/(len(counter)*c) for i, c in counter.items()}


# 7. Train Model (progressive fine-tuning)
# ======================
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True
)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'model_best.keras', monitor='val_loss', save_best_only=True, verbose=1
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1
)

# Phase 1: freeze base
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=freeze_epochs,
    callbacks=[early_stop, checkpoint, reduce_lr],
    class_weight=class_weight
)

# Phase 2: unfreeze all layers, lower lr
base_model.trainable = True
finetune_lr = min(learning_rate, 5e-5)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=finetune_lr),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs - freeze_epochs,
    callbacks=[early_stop, checkpoint, reduce_lr],
    class_weight=class_weight,
    initial_epoch=freeze_epochs
)

# Merge histories for plotting
history = history1
for k in history2.history:
    if k in history.history:
        history.history[k] += history2.history[k]
    else:
        history.history[k] = history2.history[k]

# ======================



# 8. Evaluate
# ======================
loss, acc = model.evaluate(val_ds)
print(f"Validation Accuracy: {acc*100:.2f}%")

# Classification report & confusion matrix
y_true = np.concatenate([y for x, y in val_ds], axis=0)
y_pred = np.argmax(model.predict(val_ds), axis=1)
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
cm = confusion_matrix(y_true, y_pred)
print(json.dumps(report, indent=2, ensure_ascii=False))
print("Confusion matrix:")
print(cm)

# Save report and confusion matrix to file
with open("model_metrics.json", "w", encoding="utf-8") as f:
    json.dump({"val_accuracy": acc*100, "report": report, "confusion_matrix": cm.tolist()}, f, ensure_ascii=False, indent=2)

# Print per-class accuracy
print("Per-class accuracy:")
for i, name in enumerate(class_names):
    total = cm[i].sum()
    correct = cm[i, i]
    acc_cls = correct / total if total > 0 else 0.0
    print(f"{name}: {acc_cls*100:.2f}% ({correct}/{total})")

# ======================

# 8. Save Model
# ======================
model.save("model_mobilenetv2.keras")
print("Model saved: model_mobilenetv2.keras")

# ======================

# 9. Plot Accuracy & Loss
# ======================
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.legend()
plt.title("Accuracy")

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.legend()
plt.title("Loss")

plt.show()


#cd d:\Code\Code\Project\Waste-Classification-Web-main


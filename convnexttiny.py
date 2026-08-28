import os

import cv2

import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras import layers, models

from tensorflow.keras.applications import ConvNeXtTiny

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
base_path = "/content/dataset/"
coye_base_path = "/content/dataset_coye/"
train_csv = os.path.join(base_path, "train_1_ori.csv")
train_dir = os.path.join(base_path, "train_images", "train_images")
train_coye_dir = os.path.join(coye_base_path, "train_images", "train_images")
val_csv = os.path.join(base_path, "valid_ori.csv")
val_dir = os.path.join(base_path, "val_images", "val_images")
val_coye_dir = os.path.join(coye_base_path, "val_images", "val_images")
test_csv = os.path.join(base_path, "test_ori.csv")
test_dir = os.path.join(base_path, "test_images", "test_images")
test_coye_dir = os.path.join(coye_base_path, "test_images", "test_images")


def load_and_preprocess(plain_path, coye_path, label):
    plain_img = tf.io.read_file(plain_path)
    plain_img = tf.image.decode_image(plain_img, channels=3, expand_animations=False)
    plain_img = tf.image.resize(plain_img, [224, 224])
    coye_img = tf.io.read_file(coye_path)
    coye_img = tf.image.decode_image(coye_img, channels=3, expand_animations=False)
    coye_img = tf.image.resize(coye_img, [224, 224])
    return ((plain_img, coye_img), label)


def get_all_data(csv_file, plain_dir, coye_dir):
    df = pd.read_csv(csv_file)
    plain_paths = []
    coye_paths = []
    labels = []

    for idx, row in df.iterrows():
        filename = str(row.iloc[0])

        if not filename.endswith(".png") and (not filename.endswith(".jpeg")):
            filename += ".png"
        plain_paths.append(os.path.join(plain_dir, filename))
        coye_paths.append(os.path.join(coye_dir, filename))
        labels.append(row.iloc[1])
    return (plain_paths, coye_paths, labels)


all_plain_paths, all_coye_paths, all_labels = ([], [], [])

for csv_f, p_dir, c_dir in [
    (train_csv, train_dir, train_coye_dir),
    (val_csv, val_dir, val_coye_dir),
    (test_csv, test_dir, test_coye_dir),
]:
    p, c, l = get_all_data(csv_f, p_dir, c_dir)
    all_plain_paths.extend(p)
    all_coye_paths.extend(c)
    all_labels.extend(l)
all_paths_zipped = list(zip(all_plain_paths, all_coye_paths))
train_paths_zip, temp_paths_zip, train_labels, temp_labels = train_test_split(
    all_paths_zipped, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)
val_paths_zip, test_paths_zip, val_labels, test_labels = train_test_split(
    temp_paths_zip, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)
train_plain_paths, train_coye_paths = zip(*train_paths_zip)
val_plain_paths, val_coye_paths = zip(*val_paths_zip)
test_plain_paths, test_coye_paths = zip(*test_paths_zip)

print(f"Total Combined Data: {len(all_labels)}")

print(f"-> Training set (80%): {len(train_labels)}")

print(f"-> Validation set (10%): {len(val_labels)}")

print(f"-> Testing set (10%): {len(test_labels)}")


def build_dataset_from_list(
    plain_paths, coye_paths, labels, is_training=False, batch_size=16
):
    ds = tf.data.Dataset.from_tensor_slices(
        (list(plain_paths), list(coye_paths), labels)
    )

    if is_training:
        ds = ds.shuffle(buffer_size=len(labels))
    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = build_dataset_from_list(
    train_plain_paths, train_coye_paths, train_labels, is_training=True
)
val_ds = build_dataset_from_list(
    val_plain_paths, val_coye_paths, val_labels, is_training=False
)
test_ds = build_dataset_from_list(
    test_plain_paths, test_coye_paths, test_labels, is_training=False
)

print("\nTraining Data Distribution:")
counts = pd.Series(train_labels).value_counts().sort_index()

for label, count in counts.items():

    print(f"  Class {label}: {count} images")
class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(train_labels), y=train_labels
)
class_weight_dict = dict(enumerate(class_weights))

import math

class_weight_dict = {k: math.sqrt(v) for k, v in class_weight_dict.items()}

print(
    "\nComputed Class Weights (Higher weight = higher priority for minority classes):"
)

for k, v in class_weight_dict.items():

    print(f"  Class {k}: {v:.4f}")


class SynchronizedAugmentation(layers.Layer):

    def __init__(self):
        super(SynchronizedAugmentation, self).__init__()
        self.augmenter = tf.keras.Sequential(
            [
                layers.RandomZoom(height_factor=(-0.2, 0.2)),
                layers.RandomBrightness(factor=0.2),
                layers.RandomContrast(factor=0.2),
            ]
        )

    def call(self, plain_img, coye_img, training=False):

        if not training:
            return (plain_img, coye_img)
        combined = tf.concat([plain_img, coye_img], axis=-1)
        augmented = self.augmenter(combined, training=True)
        aug_plain = augmented[:, :, :, :3]
        aug_coye = augmented[:, :, :, 3:]
        return (aug_plain, aug_coye)


def create_dual_stream_convnext_model(num_classes=5):
    input_plain = layers.Input(shape=(224, 224, 3), name="plain_input")
    input_coye = layers.Input(shape=(224, 224, 3), name="coye_input")
    aug_layer = SynchronizedAugmentation()
    aug_plain, aug_coye = aug_layer(input_plain, input_coye)
    stream1_base = tf.keras.Sequential(
        [ConvNeXtTiny(include_top=False, weights="imagenet", pooling="avg")]
    )
    feat1 = stream1_base(aug_plain)
    stream2_base = tf.keras.Sequential(
        [ConvNeXtTiny(include_top=False, weights="imagenet", pooling="avg")]
    )
    feat2 = stream2_base(aug_coye)
    feat2_reduced = layers.Dense(64, activation="relu", name="coye_reduced_features")(
        feat2
    )
    fused = layers.Concatenate(axis=-1)([feat1, feat2_reduced])
    x = layers.Dropout(0.5)(fused)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(
        inputs=[input_plain, input_coye],
        outputs=output,
        name="DualStream_ConvNeXtTiny_Main_Sub",
    )
    return model


model = create_dual_stream_convnext_model(num_classes=5)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("Starting Training with Callbacks (80/10/10 Split)...")
num_epochs = 20
save_path = "/content/drive/MyDrive/DR/best_dual_stream_convnexttiny_80_10_10_coye_reducedoat.keras"
callbacks = [
    EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        filepath=save_path, monitor="val_loss", save_best_only=True, verbose=1
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=num_epochs,
    callbacks=callbacks,
    class_weight=class_weight_dict,
)

print("Training finished!")
model.load_weights(save_path)
model.optimizer = None
model.save(save_path)

print(f"Best model safely optimized and saved to {save_path}")
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
plt.plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss", linewidth=2)
plt.plot(history.history["val_loss"], label="Val Loss", linewidth=2)
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

print("\nEvaluating model on Internal Test Set (10%)...")
y_true = np.array(test_labels)
outputs = model.predict(test_ds)
y_pred = np.argmax(outputs, axis=1)
CLASS_NAMES = [
    "No DR (0)",
    "Mild (1)",
    "Moderate (2)",
    "Severe (3)",
    "Proliferative (4)",
]
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
)
plt.title("Confusion Matrix on Test Set (10%)")
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.show()

print("\n" + "=" * 50)

print("CLASSIFICATION REPORT (Test Set)")

print("=" * 50)

print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

try:
    auc_score = roc_auc_score(y_true, outputs, multi_class="ovr")

    print("=" * 50)

    print(f"AUC ROC Score (OVR): {auc_score:.4f}")

    print("=" * 50)
except Exception as e:

    print("Could not calculate AUC:", e)

from sklearn.metrics import roc_curve, RocCurveDisplay

plt.figure(figsize=(10, 8))

for i, class_name in enumerate(CLASS_NAMES):
    fpr, tpr, _ = roc_curve(y_true == i, outputs[:, i])
    RocCurveDisplay(
        fpr=fpr,
        tpr=tpr,
        roc_auc=roc_auc_score(y_true == i, outputs[:, i]),
        estimator_name=class_name,
    ).plot(ax=plt.gca())
plt.plot([0, 1], [0, 1], "k--", label="Chance (AUC = 0.5)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve - Multi-class OVR")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
"""
train_lstm.py  — Train LSTM on MediaPipe hand landmarks

Architecture:
    Input : (batch, 30, 42)  ← 30 frames × (21 landmarks × 2 coords)
    LSTM  : 2 layers of 128 units + 1 layer of 64 units
    Output: num_classes (softmax)

Data augmentation:
    - Gaussian noise added to landmark coordinates
    - Horizontal flip (mirror hand)

Split:
    80% train / 10% val / 10% test

Validation:
    - Sequence-level F1 score
    - Confusion matrix (saved to confusion_matrix.png)
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import L2
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import os

# Reduce TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ------------------------------------------------------------------ #
# CONFIG                                                               #
# ------------------------------------------------------------------ #
SEQUENCE_LENGTH = 30      # Increased for better gesture context
FEATURE_DIM     = 42      # 21 landmarks × 2 coords (x, y)
NUM_CLASSES     = None    # set after loading data
EPOCHS          = 150
BATCH_SIZE      = 64      # Increased for 1650Ti/CPU efficiency
NOISE_STD       = 0.012   # Balanced augmentation
L2_REG          = 0.0005  # Slight regularization
SAVE_DIR        = "app/models/lstm_model"

# ------------------------------------------------------------------ #
# Device Check                                                         #
# ------------------------------------------------------------------ #
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"[Device] GPUs detected: {len(gpus)}. Using GPU for training.")
else:
    print("[Device] No GPUs found. Using CPU for training.")

# ------------------------------------------------------------------ #
# Load landmark sequences                                              #
# Expected: X.npy shape (N, 15, 42), y.npy shape (N,) int class ids  #
# ------------------------------------------------------------------ #
print("[Train] Loading data...")
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
X = np.load(os.path.join(DATA_DIR, "X.npy"))      # shape: (N, SEQUENCE_LENGTH, FEATURE_DIM)
y = np.load(os.path.join(DATA_DIR, "y.npy"))      # shape: (N,) — integer class labels


NUM_CLASSES = len(np.unique(y))
print(f"[Train] Samples: {len(X)}, Classes: {NUM_CLASSES}")

# ------------------------------------------------------------------ #
# Augmentation                                                         #
# ------------------------------------------------------------------ #
def add_gaussian_noise(X, std=NOISE_STD):
    return X + np.random.normal(0, std, X.shape).astype(np.float32)

def horizontal_flip(X):
    """Flip x-coordinates for single hand."""
    X_flipped = X.copy()
    # 1. Negate relative x-coordinates within the hand buffer.
    X_flipped[:, :, 0:42:2] = -X_flipped[:, :, 0:42:2]
    return X_flipped

print("[Train] Applying aggressive augmentation (noise + flip + double-noise)...")
X_noise   = add_gaussian_noise(X)
X_noise2  = add_gaussian_noise(X)  # Second independent noise sample
X_flipped = horizontal_flip(X)
X_flipped_noise = add_gaussian_noise(horizontal_flip(X))  # Flipped + noise
X_aug = np.concatenate([X, X_noise, X_noise2, X_flipped, X_flipped_noise], axis=0)
y_aug = np.concatenate([y, y,       y,        y,         y              ], axis=0)
print(f"[Train] After augmentation: {len(X_aug)} samples")

# ------------------------------------------------------------------ #
# One-hot encode labels                                                #
# ------------------------------------------------------------------ #
y_ohe = tf.keras.utils.to_categorical(y_aug, NUM_CLASSES)

# ------------------------------------------------------------------ #
# Split: 80 / 10 / 10                                                  #
# ------------------------------------------------------------------ #
X_train, X_temp, y_train, y_temp = train_test_split(
    X_aug, y_ohe, test_size=0.20, stratify=y_aug, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)
print(f"[Train] Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}")

# ------------------------------------------------------------------ #
# Model with strong regularization                                     #
# ------------------------------------------------------------------ #
model = Sequential([
    BatchNormalization(input_shape=(SEQUENCE_LENGTH, FEATURE_DIM)),
    LSTM(128, return_sequences=True,
         kernel_regularizer=L2(L2_REG), recurrent_regularizer=L2(L2_REG)),
    BatchNormalization(),
    Dropout(0.4),
    LSTM(128, return_sequences=True,
         kernel_regularizer=L2(L2_REG), recurrent_regularizer=L2(L2_REG)),
    BatchNormalization(),
    Dropout(0.4),
    LSTM(64, return_sequences=False,
         kernel_regularizer=L2(L2_REG), recurrent_regularizer=L2(L2_REG)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation="relu", kernel_regularizer=L2(L2_REG)),
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)
model.summary()

# ------------------------------------------------------------------ #
# Callbacks                                                            #
# ------------------------------------------------------------------ #
callbacks = [
    EarlyStopping(patience=20, restore_best_weights=True, verbose=1, min_delta=0.0005),
    ModelCheckpoint(SAVE_DIR + "_best.h5", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6, verbose=1)
]

# ------------------------------------------------------------------ #
# Training                                                             #
# ------------------------------------------------------------------ #
print("[Train] Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# ------------------------------------------------------------------ #
# Evaluation — sequence-level F1 + confusion matrix                   #
# ------------------------------------------------------------------ #
print("[Train] Evaluating on test set...")
y_pred_ohe = model.predict(X_test, verbose=0)
y_pred     = np.argmax(y_pred_ohe, axis=1)
y_true     = np.argmax(y_test,    axis=1)

f1 = f1_score(y_true, y_pred, average="weighted")
print(f"[Train] Test F1 (weighted): {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar(im, ax=ax)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("[Train] Confusion matrix saved to confusion_matrix.png")

# ------------------------------------------------------------------ #
# Save final model                                                     #
# ------------------------------------------------------------------ #
os.makedirs(SAVE_DIR, exist_ok=True)
model.save(os.path.join(SAVE_DIR, "lstm_model.h5"))
print(f"[Train] Model saved to {SAVE_DIR}/lstm_model.h5")

# Save labels map (class_id → label)
# Assumes a labels.json with {"0": "HELLO", "1": "THANK_YOU", ...}
# is already present. If not, create it here.
print("[Train] Done.")

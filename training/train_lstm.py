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
SEQUENCE_LENGTH = 20
FEATURE_DIM     = 42
NUM_CLASSES     = None
EPOCHS          = 150
BATCH_SIZE      = 32    # Smaller batch for better generalization
NOISE_STD       = 0.015  # Increased jitter
L2_REG          = 0.001  # Increased regularization
SAVE_DIR        = "app/models/lstm_model" 

# ------------------------------------------------------------------ #
# Load landmark sequences                                              #
# ------------------------------------------------------------------ #
print("[Train] Loading data...")
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
X = np.load(os.path.join(DATA_DIR, "X.npy"))      # shape: (N, 20, 42)
y = np.load(os.path.join(DATA_DIR, "y.npy"))      # shape: (N,)

NUM_CLASSES = len(np.unique(y))
print(f"[Train] Samples: {len(X)}, Classes: {NUM_CLASSES}")

# ------------------------------------------------------------------ #
# Augmentation                                                         #
# ------------------------------------------------------------------ #
def add_gaussian_noise(X, std=NOISE_STD):
    return X + np.random.normal(0, std, X.shape).astype(np.float32)

def horizontal_flip(X):
    X_flipped = X.copy()
    # Mirror x-coordinates. Assumes normalized landmarks where x=0 is center.
    X_flipped[:, :, 0:42:2] = -X_flipped[:, :, 0:42:2]
    return X_flipped

def random_rotate(X, angle_range=12):
    """Rotate landmarks by random angle."""
    X_rotated = X.copy()
    for i in range(len(X_rotated)):
        angle = np.random.uniform(-angle_range, angle_range)
        rad = np.radians(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        seq = X_rotated[i].reshape(SEQUENCE_LENGTH, 21, 2)
        rotated_seq = np.array([np.dot(frame, rot_mat.T) for frame in seq])
        X_rotated[i] = rotated_seq.reshape(SEQUENCE_LENGTH, FEATURE_DIM)
    return X_rotated

def random_scale(X, scale_range=(0.9, 1.1)):
    """Scale landmarks randomly."""
    scales = np.random.uniform(scale_range[0], scale_range[1], (len(X), 1, 1))
    return X * scales

def random_translate(X, range=0.05):
    """Slightly shift landmarks to simulate hand position variance."""
    offsets = np.random.uniform(-range, range, (len(X), 1, FEATURE_DIM))
    return X + offsets

print("[Train] Applying LIGHTWEIGHT model augmentation...")
X_noise   = add_gaussian_noise(X, 0.012)
X_rot     = random_rotate(X, 10)
X_scale   = random_scale(X, (0.92, 1.08))
X_trans   = random_translate(X, 0.03)
X_flipped = horizontal_flip(X)
X_flipped_rot = random_rotate(X_flipped, 8)

# Combine original + variations
X_aug = np.concatenate([X, X_noise, X_rot, X_scale, X_trans, X_flipped, X_flipped_rot], axis=0).astype(np.float32)
y_aug = np.concatenate([y, y,       y,     y,       y,       y,         y          ], axis=0)
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
# Model: Optimized GRU (Faster Inference + Anti-Overfit)              #
# ------------------------------------------------------------------ #
from tensorflow.keras.layers import GRU

model = Sequential([
    BatchNormalization(input_shape=(SEQUENCE_LENGTH, FEATURE_DIM)),
    
    # Unidirectional GRU is much faster for real-time and enough for alphabets
    GRU(64, return_sequences=True, kernel_regularizer=L2(L2_REG)),
    BatchNormalization(),
    Dropout(0.4),
    
    GRU(32, return_sequences=False, kernel_regularizer=L2(L2_REG)),
    BatchNormalization(),
    Dropout(0.4),
    
    Dense(64, activation="relu", kernel_regularizer=L2(L2_REG)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2), # Increased smoothing
    metrics=["accuracy"]
)
model.summary()

# ------------------------------------------------------------------ #
# Callbacks                                                            #
# ------------------------------------------------------------------ #
callbacks = [
    EarlyStopping(patience=20, restore_best_weights=True, verbose=1),
    ModelCheckpoint(SAVE_DIR + "_best.keras", save_best_only=True, verbose=1),
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
# Save and Convert to TFLite (for Performance)                        #
# ------------------------------------------------------------------ #
os.makedirs(SAVE_DIR, exist_ok=True)
model.save(os.path.join(SAVE_DIR, "lstm_model.keras"))


print("[Train] Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, # enable TFLite ops.
    tf.lite.OpsSet.SELECT_TF_OPS # enable TF ops.
]
converter._experimental_lower_tensor_list_ops = False
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(os.path.join(SAVE_DIR, "model.tflite"), "wb") as f:
    f.write(tflite_model)
print(f"[Train] Model saved to {SAVE_DIR}/model.tflite")


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

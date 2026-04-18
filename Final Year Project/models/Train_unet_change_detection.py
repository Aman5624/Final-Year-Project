"""
train_unet_change_detection.py
================================
Trains a U-Net model for building change detection.

MODEL:   unet_building_change_detection.h5
INPUT:   (512, 512, 3) per-pixel absolute difference of before/after images
OUTPUT:  (512, 512, 1) binary change mask (sigmoid)

Training pipeline:
  1. Loads 10 sample drone/street images of informal construction.
  2. On-the-fly synthetic pair generation via a Keras Sequence generator
     (memory-safe: only 1 batch in RAM at a time).
  3. Trains with combined BCE + Dice loss.
  4. Saves best checkpoint to models/unet_building_change_detection.h5
"""

import os, random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, UpSampling2D, concatenate,
    BatchNormalization, Dropout, Activation
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

# ── CONFIG ────────────────────────────────────────────────────────────────────
IMG_SIZE     = 512
BATCH_SIZE   = 2
EPOCHS       = 50
TRAIN_STEPS  = 40
VAL_STEPS    = 10
SEED         = 42
MODEL_DIR    = "models"
MODEL_PATH   = os.path.join(MODEL_DIR, "unet_building_change_detection.h5")

IMG_PATHS = [
    "C:/Users/Aman/Downloads/Final Year Project-Main/Final Year Project/Sample_Data_Set/Image_upz07tupz07tupz0.png",
    *[f"C:/Users/Aman/Downloads/Final Year Project-Main/Final Year Project/Sample_Data_Set/Image_upz07tupz07tupz0__{i}_.png"
      for i in range(1, 10)]
]

os.makedirs(MODEL_DIR, exist_ok=True)
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


# ── IMAGE HELPERS ─────────────────────────────────────────────────────────────
def load_and_resize(path, size=IMG_SIZE):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (size, size))


def random_change_mask(h, w):
    mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(random.randint(2, 6)):
        x1 = random.randint(0, w - 60)
        y1 = random.randint(0, h - 60)
        x2 = random.randint(x1 + 20, min(x1 + 180, w))
        y2 = random.randint(y1 + 20, min(y1 + 180, h))
        mask[y1:y2, x1:x2] = 1.0
    return mask


def apply_change(img, mask):
    after = img.copy().astype(np.float32)
    m = mask.astype(bool)
    grey = np.mean(after[m], axis=-1, keepdims=True)
    concrete = np.concatenate([grey * 0.88, grey * 0.86, grey * 0.80], axis=-1)
    after[m] = np.clip(concrete + np.random.uniform(-12, 12, concrete.shape), 0, 255)
    return after.astype(np.uint8)


def random_augment(b, a, m):
    if random.random() > 0.5:
        b, a, m = np.flipud(b), np.flipud(a), np.flipud(m)
    if random.random() > 0.5:
        b, a, m = np.fliplr(b), np.fliplr(a), np.fliplr(m)
    k = random.randint(0, 3)
    return np.rot90(b, k), np.rot90(a, k), np.rot90(m, k)


# ── GENERATOR ─────────────────────────────────────────────────────────────────
class CDGenerator(Sequence):
    def __init__(self, imgs, batch_size, steps):
        self.imgs  = imgs
        self.batch = batch_size
        self.steps = steps

    def __len__(self):
        return self.steps

    def __getitem__(self, _):
        X_b, Y_b = [], []
        for _ in range(self.batch):
            src  = random.choice(self.imgs)
            mask = random_change_mask(IMG_SIZE, IMG_SIZE)
            after = apply_change(src, mask)
            b, a, m = random_augment(src, after, mask)
            diff = np.abs(b.astype(np.float32) - a.astype(np.float32)) / 255.0
            X_b.append(diff)
            Y_b.append(m[..., np.newaxis])
        return np.array(X_b, np.float32), np.array(Y_b, np.float32)


# ── U-NET ─────────────────────────────────────────────────────────────────────
def conv_block(x, f, drop=0.0):
    for _ in range(2):
        x = Conv2D(f, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    if drop:
        x = Dropout(drop)(x)
    return x


def build_unet(sz=IMG_SIZE):
    inp = Input((sz, sz, 3), name="diff_input")
    c1 = conv_block(inp, 32);         p1 = MaxPooling2D(2)(c1)
    c2 = conv_block(p1,  64, 0.1);    p2 = MaxPooling2D(2)(c2)
    c3 = conv_block(p2, 128, 0.2);    p3 = MaxPooling2D(2)(c3)
    c4 = conv_block(p3, 256, 0.2);    p4 = MaxPooling2D(2)(c4)
    c5 = conv_block(p4, 512, 0.3)
    u6 = concatenate([UpSampling2D(2)(c5), c4]); c6 = conv_block(u6, 256, 0.2)
    u7 = concatenate([UpSampling2D(2)(c6), c3]); c7 = conv_block(u7, 128, 0.2)
    u8 = concatenate([UpSampling2D(2)(c7), c2]); c8 = conv_block(u8,  64, 0.1)
    u9 = concatenate([UpSampling2D(2)(c8), c1]); c9 = conv_block(u9,  32)
    out = Conv2D(1, 1, activation="sigmoid", name="change_mask")(c9)
    return Model(inp, out, name="unet_change_detection")


# ── LOSS ──────────────────────────────────────────────────────────────────────
def dice_coef(y_true, y_pred, smooth=1.0):
    yf, pf = tf.keras.backend.flatten(y_true), tf.keras.backend.flatten(y_pred)
    return (2.0 * tf.keras.backend.sum(yf * pf) + smooth) / (
        tf.keras.backend.sum(yf) + tf.keras.backend.sum(pf) + smooth)

def bce_dice(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + (1 - dice_coef(y_true, y_pred))


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  U-Net Building Change Detection – Training")
    print("=" * 65)

    valid = [p for p in IMG_PATHS if os.path.exists(p)]
    print(f"\n[DATA] {len(valid)}/{len(IMG_PATHS)} source images found")
    imgs = [load_and_resize(p) for p in valid]
    print(f"  Loaded {len(imgs)} images at {IMG_SIZE}×{IMG_SIZE}")

    train_gen = CDGenerator(imgs, BATCH_SIZE, TRAIN_STEPS)
    val_gen   = CDGenerator(imgs, BATCH_SIZE, VAL_STEPS)

    model = build_unet()
    model.compile(optimizer=Adam(1e-4), loss=bce_dice,
                  metrics=[dice_coef, "accuracy"])
    print(f"\n[MODEL] {model.count_params():,} parameters | "
          f"in={model.input_shape} out={model.output_shape}\n")

    cbs = [
        ModelCheckpoint(MODEL_PATH, monitor="val_dice_coef",
                        save_best_only=True, mode="max", verbose=1),
        EarlyStopping(monitor="val_dice_coef", patience=10,
                      mode="max", restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-7, verbose=1),
    ]

    history = model.fit(train_gen, validation_data=val_gen,
                        epochs=EPOCHS, callbacks=cbs, verbose=1)

    if not os.path.exists(MODEL_PATH):
        model.save(MODEL_PATH)

    best = int(np.argmax(history.history["val_dice_coef"]))
    print("\n" + "=" * 65)
    print(f"  ✓ Saved → {MODEL_PATH}")
    print(f"  Best epoch    : {best+1}")
    print(f"  Val Dice      : {history.history['val_dice_coef'][best]:.4f}")
    print(f"  Val Accuracy  : {history.history['val_accuracy'][best]:.4f}")
    print("=" * 65)
    print("\n[USAGE in app.py]")
    print("  img1 = preprocess_image_cd(before_path)  # → (1,512,512,3) float32 /255")
    print("  img2 = preprocess_image_cd(after_path)")
    print("  diff = np.abs(img1 - img2)               # element-wise diff")
    print("  pred = model.predict(diff)[0]            # → (512,512,1)")


if __name__ == "__main__":
    main()
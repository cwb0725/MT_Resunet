# train_512.py  (ResUNet only, no CBAM)
import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, Activation, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.activations import relu

from res_blocks import (
    resunet_down_block,
    resunet_up_block,
    resunet_identity_block
)

# =========================
# Base paths (stable)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NPY_DIR   = os.path.join(BASE_DIR, "npy_data")    
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
LOG_DIR   = os.path.join(BASE_DIR, "..", "log")

os.makedirs(NPY_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

MEAN_PATH = os.path.join(NPY_DIR, "train_mean.npy")

# =========================
# Training config
# =========================
PATCH = 512
BATCH_SIZE = 1

EPOCHS = 50
STEPS_PER_EPOCH = 1000
VAL_STEPS = 200

LEARNING_RATE = 1e-4

# Foreground-aware patch sampling (IMPORTANT)
FG_PROB = 0.6          # 60% of patches try to contain foreground
FG_RATIO_MIN = 0.005   # >=0.5% foreground pixels inside patch
MAX_TRIES = 50

# Loss: focal for imbalanced segmentation
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.75

# Device
FORCE_CPU = False      # True -> CPU only

GPU_ID = "0"           # use GPU0 by default


def setup_device(force_cpu=False, gpu_id="0"):
    # Must be set before TF init ideally, but still helps
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("[Device] CPU only")
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            print(f"[Device] Using GPU:{gpu_id} (memory growth enabled)")
        except Exception as e:
            print("[Device] Failed to set memory growth:", e)
    else:
        print("[Device] No GPU found, fallback to CPU")


def load_npy_data():
    img_path = os.path.join(NPY_DIR, "imgs_train.npy")
    msk_path = os.path.join(NPY_DIR, "imgs_mask_train.npy")

    if not os.path.exists(img_path) or not os.path.exists(msk_path):
        print("[DEBUG] NPY_DIR =", NPY_DIR)
        if os.path.isdir(NPY_DIR):
            print("[DEBUG] files in NPY_DIR:")
            for f in sorted(os.listdir(NPY_DIR))[:200]:
                print("   ", f)
        raise FileNotFoundError(f"Missing: {img_path} or {msk_path}")

    imgs = np.load(img_path)
    masks = np.load(msk_path)

    imgs = np.asarray(imgs)
    masks = np.asarray(masks)

    if imgs.ndim == 3:
        imgs = imgs[..., np.newaxis]
    if masks.ndim == 3:
        masks = masks[..., np.newaxis]

    assert imgs.ndim == 4 and masks.ndim == 4, (imgs.shape, masks.shape)
    assert imgs.shape[:3] == masks.shape[:3], (imgs.shape, masks.shape)
    assert imgs.shape[-1] == 1 and masks.shape[-1] == 1, (imgs.shape, masks.shape)

    print("[Data] imgs:", imgs.shape, imgs.dtype, "min/max:", float(imgs.min()), float(imgs.max()))
    print("[Data] masks:", masks.shape, masks.dtype, "unique:", np.unique(masks)[:10])
    print("[Data] foreground ratio (global):", float((masks > 0.5).mean()))

    return imgs, masks


def normalize_and_mean_sub(imgs, save_mean=True):
    x = imgs.astype(np.float32)

    # If imgs are 0~255, scale to 0~1
    if x.max() > 1.5:
        x /= 255.0

    mean = x.mean(axis=0)  # (H,W,1)

    if save_mean:
        np.save(MEAN_PATH, mean.astype(np.float32))
        print("[Mean] Saved:", MEAN_PATH, "shape:", mean.shape)

    x = x - mean
    return x, mean


def patch_generator(imgs, masks, patch=512, batch_size=1,
                    fg_ratio_min=0.005, fg_prob=0.6, max_tries=50):
    """
    imgs, masks: (N,H,W,1)
    masks expected 0/1 float32
    """
    N, H, W, C = imgs.shape
    assert C == 1

    while True:
        bx = np.zeros((batch_size, patch, patch, 1), dtype=np.float32)
        by = np.zeros((batch_size, patch, patch, 1), dtype=np.float32)

        for b in range(batch_size):
            idx = np.random.randint(0, N)
            want_fg = (np.random.rand() < fg_prob)

            y0 = x0 = None
            for _ in range(max_tries):
                y = np.random.randint(0, H - patch + 1)
                x = np.random.randint(0, W - patch + 1)

                m_patch = masks[idx, y:y+patch, x:x+patch, 0]
                fg_ratio = float(m_patch.mean())

                if (not want_fg) or (fg_ratio >= fg_ratio_min):
                    y0, x0 = y, x
                    break

            if y0 is None:
                y0 = np.random.randint(0, H - patch + 1)
                x0 = np.random.randint(0, W - patch + 1)

            bx[b] = imgs[idx, y0:y0+patch, x0:x0+patch, :]
            by[b] = masks[idx, y0:y0+patch, x0:x0+patch, :]

        yield bx, by


def build_resunet(input_size=(512, 512, 1)):
    """
    ResUNet (no CBAM)
    Encoder: resunet_down_block (with pooling inside)
    Bottleneck: resunet_identity_block x2
    Decoder: upsample + concat skip + resunet_up_block
    """
    inputs = Input(input_size)

    # Stem (512)
    x0 = Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)
    x0 = Activation(relu)(x0)
    skip0 = x0

    # Encoder (downsample inside each down_block)
    d1 = resunet_down_block(skip0, 32, 32, 64, 64)        # 256
    skip1 = d1

    d2 = resunet_down_block(skip1, 64, 64, 128, 128)      # 128
    skip2 = d2

    d3 = resunet_down_block(skip2, 128, 128, 256, 256)    # 64
    skip3 = d3

    d4 = resunet_down_block(skip3, 256, 256, 512, 512)    # 32

    # Bottleneck
    b = resunet_identity_block(d4, 256, 256, 512)
    b = resunet_identity_block(b, 256, 256, 512)

    # Decoder
    u3 = UpSampling2D((2, 2))(b)                           # 64
    u3 = Concatenate(axis=-1)([u3, skip3])
    u3 = resunet_up_block(u3, 256, 256, 256, 256)

    u2 = UpSampling2D((2, 2))(u3)                          # 128
    u2 = Concatenate(axis=-1)([u2, skip2])
    u2 = resunet_up_block(u2, 128, 128, 128, 128)

    u1 = UpSampling2D((2, 2))(u2)                          # 256
    u1 = Concatenate(axis=-1)([u1, skip1])
    u1 = resunet_up_block(u1, 64, 64, 64, 64)

    u0 = UpSampling2D((2, 2))(u1)                          # 512
    u0 = Concatenate(axis=-1)([u0, skip0])
    u0 = resunet_up_block(u0, 32, 32, 64, 64)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(u0)

    model = Model(inputs, outputs, name="ResUNet_patch512")

    loss_fn = tf.keras.losses.BinaryFocalCrossentropy(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=loss_fn,
        metrics=[]
    )
    return model


def main():
    setup_device(force_cpu=FORCE_CPU, gpu_id=GPU_ID)

    imgs_raw, masks_raw = load_npy_data()

    # Normalize & mean-sub (save mean for test)
    imgs, _ = normalize_and_mean_sub(imgs_raw, save_mean=True)

    # Masks -> 0/1 float32
    masks = (masks_raw > 0.5).astype(np.float32)

    # Split by image index (90/10)
    N = imgs.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)

    split = max(1, int(N * 0.9))
    tr_idx = idx[:split]
    va_idx = idx[split:] if split < N else idx[:1]

    x_tr, y_tr = imgs[tr_idx], masks[tr_idx]
    x_va, y_va = imgs[va_idx], masks[va_idx]

    print("[Split] train:", x_tr.shape, "val:", x_va.shape)

    train_gen = patch_generator(
        x_tr, y_tr,
        patch=PATCH,
        batch_size=BATCH_SIZE,
        fg_ratio_min=FG_RATIO_MIN,
        fg_prob=FG_PROB,
        max_tries=MAX_TRIES
    )

    val_gen = patch_generator(
        x_va, y_va,
        patch=PATCH,
        batch_size=BATCH_SIZE,
        fg_ratio_min=FG_RATIO_MIN,
        fg_prob=FG_PROB,
        max_tries=MAX_TRIES
    )

    model = build_resunet(input_size=(PATCH, PATCH, 1))
    model.summary()

    model_path = os.path.join(MODEL_DIR, "ResUNet_patch512.keras")
    ckpt = ModelCheckpoint(
        filepath=model_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        mode="min",
        verbose=1
    )

    csvlog = CSVLogger(os.path.join(LOG_DIR, "train_resunet_patch512.csv"), append=False)

    rlop = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
    )

    es = EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
    )

    model.fit(
        train_gen,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=VAL_STEPS,
        callbacks=[ckpt, csvlog, rlop, es],
        verbose=1
    )

    print("[Done] Best model saved to:", model_path)
    print("[Done] Mean saved to:", MEAN_PATH)


if __name__ == "__main__":
    main()

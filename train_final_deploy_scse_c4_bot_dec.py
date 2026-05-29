# -*- coding: utf-8 -*-
import os
import csv
import argparse
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, CSVLogger, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import mixed_precision

from res_blocks_leakyrelu import (
    resunet_down_block,
    resunet_up_block,
    resunet_identity_block,
)
from attention_module import scSE


# =========================
# Config and initialization
# =========================
def set_global_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)


os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
mixed_precision.set_global_policy("mixed_float16")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NPY_DIR = os.path.join(BASE_DIR, "npy_data")
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")
LOG_DIR = os.path.join(BASE_DIR, "..", "log")

os.makedirs(NPY_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

PATCH = 512
INFER_STRIDE = 256
BATCH_SIZE = 8
EPOCHS = 120
STEPS_PER_EPOCH = 300
LEARNING_RATE = 1e-4

FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.75
W_FOCAL = 0.4
W_DICE = 0.6

TH = 0.5
THRESHOLD_CANDIDATES = [0.35, 0.40, 0.45, 0.50, 0.55]


# =========================
# Losses and metrics
# =========================
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 0.0, 1.0)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    return tf.reduce_mean((2.0 * intersection + smooth) / (union + smooth))


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def iou_metric(y_true, y_pred, threshold=TH, eps=1e-6):
    y_true = tf.cast(y_true >= 0.5, tf.float32)
    y_pred = tf.cast(y_pred >= threshold, tf.float32)
    inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3]) - inter
    return tf.reduce_mean((inter + eps) / (union + eps))


def combined_loss(y_true, y_pred):
    focal = tf.keras.losses.BinaryFocalCrossentropy(
        gamma=FOCAL_GAMMA,
        alpha=FOCAL_ALPHA,
    )
    return W_FOCAL * focal(y_true, y_pred) + W_DICE * dice_loss(y_true, y_pred)


# =========================
# Data pipeline
# =========================
def tf_augment_pair(x, y):
    c = tf.shape(x)[-1]
    xy = tf.concat([x, y], axis=-1)
    xy = tf.image.random_flip_left_right(xy)
    xy = tf.image.random_flip_up_down(xy)
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    xy = tf.image.rot90(xy, k=k)
    x_aug = xy[..., :c]
    y_aug = xy[..., c:]
    return x_aug, y_aug


def extract_balanced_patches(imgs, masks, patch=512, stride=256, fg_ratio_min=0.05):
    fg_x, fg_y, bg_x, bg_y = [], [], [], []
    n_imgs, h, w, _ = imgs.shape

    for i in range(n_imgs):
        img = imgs[i]
        mask = masks[i]
        for y in range(0, h - patch + 1, stride):
            for x in range(0, w - patch + 1, stride):
                p_img = img[y:y + patch, x:x + patch]
                p_mask = mask[y:y + patch, x:x + patch]
                fg_ratio = np.sum(p_mask > 0) / float(patch * patch)
                if fg_ratio >= fg_ratio_min:
                    fg_x.append(p_img)
                    fg_y.append(p_mask)
                else:
                    bg_x.append(p_img)
                    bg_y.append(p_mask)

    return (
        np.array(fg_x, dtype=np.float32),
        np.array(fg_y, dtype=np.float32),
    ), (
        np.array(bg_x, dtype=np.float32),
        np.array(bg_y, dtype=np.float32),
    )


def create_balanced_train_dataset(
    x_train,
    y_train,
    patch=512,
    batch_size=4,
    fg_weight=0.8,
    fg_ratio_min=0.05,
):
    (fg_x, fg_y), (bg_x, bg_y) = extract_balanced_patches(
        x_train,
        y_train,
        patch=patch,
        stride=256,
        fg_ratio_min=fg_ratio_min,
    )

    print(
        "[Data] Memory cropping extracted {} foreground patches and {} background patches".format(
            len(fg_x), len(bg_x)
        )
    )

    if len(fg_x) == 0:
        raise RuntimeError("No foreground patches were extracted. Check masks or fg_ratio_min.")
    if len(bg_x) == 0:
        raise RuntimeError("No background patches were extracted. Check masks or fg_ratio_min.")

    fg_ds = tf.data.Dataset.from_tensor_slices((fg_x, fg_y)).shuffle(len(fg_x)).repeat()
    bg_ds = tf.data.Dataset.from_tensor_slices((bg_x, bg_y)).shuffle(len(bg_x)).repeat()

    train_ds = tf.data.Dataset.sample_from_datasets(
        [fg_ds, bg_ds],
        weights=[fg_weight, 1.0 - fg_weight],
    )
    train_ds = train_ds.map(tf_augment_pair, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds


# =========================
# Model
# =========================
def decoder_up_block(x, skip, out_channels, use_scse=True):
    x = UpSampling2D(size=(2, 2))(x)
    x = Concatenate(axis=-1)([x, skip])
    inner = max(out_channels // 4, 1)
    x = resunet_up_block(x, inner, inner, out_channels, out_channels)
    if use_scse:
        x = scSE(ratio=1 / 16.0)(x)
    return x


def build_model(input_shape=(PATCH, PATCH, 1), compile_model=True):
    inputs = Input(input_shape)
    x0 = Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal")(inputs)

    c1 = resunet_identity_block(x0, 16, 16, 64)
    p1 = resunet_down_block(c1, 32, 32, 128, 128)

    c2 = resunet_identity_block(p1, 32, 32, 128)
    p2 = resunet_down_block(c2, 64, 64, 256, 256)

    c3 = resunet_identity_block(p2, 64, 64, 256)
    p3 = resunet_down_block(c3, 128, 128, 512, 512)

    c4 = resunet_identity_block(p3, 128, 128, 512)
    c4 = scSE(ratio=1 / 16.0)(c4)
    p4 = resunet_down_block(c4, 256, 256, 1024, 1024)

    c5 = resunet_identity_block(p4, 256, 256, 1024)
    c5 = scSE(ratio=1 / 16.0)(c5)

    c6 = decoder_up_block(c5, c4, 512, use_scse=True)
    c7 = decoder_up_block(c6, c3, 256, use_scse=True)
    c8 = decoder_up_block(c7, c2, 128, use_scse=False)
    c9 = decoder_up_block(c8, c1, 64, use_scse=False)

    outputs = Conv2D(1, 1, activation="sigmoid", dtype="float32")(c9)
    model = Model(inputs, outputs, name="resunet_scse_c4_bottleneck_decoder")

    if compile_model:
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.0),
            loss=combined_loss,
            metrics=[
                dice_coef,
                iou_metric,
                tf.keras.metrics.Precision(name="prec", thresholds=TH),
                tf.keras.metrics.Recall(name="rec", thresholds=TH),
            ],
        )

    return model


# =========================
# Full-image inference and metrics
# =========================
def make_start_positions(length, patch, stride):
    if length <= patch:
        return [0]
    positions = list(range(0, max(length - patch, 1), stride))
    last = length - patch
    if positions[-1] != last:
        positions.append(last)
    return positions


def pad_to_min_size(img, min_h, min_w):
    h, w, c = img.shape
    pad_h = max(0, min_h - h)
    pad_w = max(0, min_w - w)

    if pad_h == 0 and pad_w == 0:
        return img, (0, 0)

    img_pad = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    return img_pad, (pad_h, pad_w)


def predict_full_image_by_tiles(model, img, patch=512, stride=256):
    img_pad, (pad_h, pad_w) = pad_to_min_size(img, patch, patch)
    h, w, _ = img_pad.shape

    ys = make_start_positions(h, patch, stride)
    xs = make_start_positions(w, patch, stride)

    pred_sum = np.zeros((h, w, 1), dtype=np.float32)
    pred_count = np.zeros((h, w, 1), dtype=np.float32)

    for y in ys:
        for x in xs:
            tile = img_pad[y:y + patch, x:x + patch, :]
            tile_in = np.expand_dims(tile, axis=0)
            tile_pred = model.predict(tile_in, verbose=0)[0]
            pred_sum[y:y + patch, x:x + patch, :] += tile_pred
            pred_count[y:y + patch, x:x + patch, :] += 1.0

    pred = pred_sum / np.maximum(pred_count, 1.0)

    if pad_h or pad_w:
        pred = pred[:img.shape[0], :img.shape[1], :]

    return pred


def compute_metrics_np(y_true, y_prob, threshold=0.5, eps=1e-6):
    y_true = (y_true >= 0.5).astype(np.float32)
    y_pred = (y_prob >= threshold).astype(np.float32)

    inter = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - inter

    dice = (2.0 * inter + eps) / (np.sum(y_true) + np.sum(y_pred) + eps)
    iou = (inter + eps) / (union + eps)

    tp = inter
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


def summarize_threshold_metrics(preds, y_true, thresholds):
    threshold_rows = []

    for th in thresholds:
        per_image = []
        for i in range(len(preds)):
            m = compute_metrics_np(y_true[i], preds[i], threshold=th)
            m["image_index"] = int(i)
            per_image.append(m)

        row = {
            "threshold": float(th),
            "dice": float(np.mean([m["dice"] for m in per_image])),
            "iou": float(np.mean([m["iou"] for m in per_image])),
            "precision": float(np.mean([m["precision"] for m in per_image])),
            "recall": float(np.mean([m["recall"] for m in per_image])),
        }
        threshold_rows.append(row)

    best_row = max(threshold_rows, key=lambda x: x["dice"])
    best_threshold = float(best_row["threshold"])

    best_per_image = []
    for i in range(len(preds)):
        m = compute_metrics_np(y_true[i], preds[i], threshold=best_threshold)
        m["image_index"] = int(i)
        best_per_image.append(m)

    return best_row, best_per_image, threshold_rows


def evaluate_on_full_images(model, x_data, y_data, patch=512, stride=256, thresholds=None):
    if thresholds is None:
        thresholds = [TH]

    preds = []
    for i in range(len(x_data)):
        pred = predict_full_image_by_tiles(model, x_data[i], patch=patch, stride=stride)
        preds.append(pred)

    return summarize_threshold_metrics(preds, y_data, thresholds)


class FullImageBestSaver(Callback):
    def __init__(
        self,
        x_data,
        y_data,
        weights_path,
        keras_path,
        metrics_csv_path,
        patch=512,
        stride=256,
        thresholds=None,
    ):
        super().__init__()
        self.x_data = x_data
        self.y_data = y_data
        self.weights_path = weights_path
        self.keras_path = keras_path
        self.metrics_csv_path = metrics_csv_path
        self.patch = patch
        self.stride = stride
        self.thresholds = thresholds if thresholds is not None else [TH]

        self.best_fullimg_dice = -1.0
        self.best_epoch = 0
        self.best_threshold = TH
        self.best_metrics = None
        self.history_rows = []

    def on_epoch_end(self, epoch, logs=None):
        best_row, _, threshold_rows = evaluate_on_full_images(
            self.model,
            self.x_data,
            self.y_data,
            patch=self.patch,
            stride=self.stride,
            thresholds=self.thresholds,
        )

        fullimg_dice = float(best_row["dice"])
        best_threshold = float(best_row["threshold"])

        if logs is not None:
            logs["train_fullimg_dice"] = fullimg_dice
            logs["train_fullimg_iou"] = float(best_row["iou"])
            logs["train_fullimg_precision"] = float(best_row["precision"])
            logs["train_fullimg_recall"] = float(best_row["recall"])
            logs["train_fullimg_best_threshold"] = best_threshold

        threshold_text = ", ".join(
            ["{:.2f}:{:.4f}".format(r["threshold"], r["dice"]) for r in threshold_rows]
        )
        print(
            "\nEpoch {}: train_fullimg_dice={:.4f}, best_threshold={:.2f}".format(
                epoch + 1,
                fullimg_dice,
                best_threshold,
            )
        )
        print("Epoch {}: threshold_dice = {}".format(epoch + 1, threshold_text))

        self.history_rows.append({
            "epoch": int(epoch + 1),
            "best_threshold": float(best_threshold),
            "fullimg_dice": float(best_row["dice"]),
            "fullimg_iou": float(best_row["iou"]),
            "fullimg_precision": float(best_row["precision"]),
            "fullimg_recall": float(best_row["recall"]),
        })
        self._flush_history_csv()

        if fullimg_dice > self.best_fullimg_dice:
            self.best_fullimg_dice = fullimg_dice
            self.best_epoch = int(epoch + 1)
            self.best_threshold = best_threshold
            self.best_metrics = {
                "dice": float(best_row["dice"]),
                "iou": float(best_row["iou"]),
                "precision": float(best_row["precision"]),
                "recall": float(best_row["recall"]),
                "threshold": float(best_threshold),
            }
            self.model.save_weights(self.weights_path)
            self.model.save(self.keras_path)
            print(
                "Epoch {}: full-image Dice improved to {:.4f} at threshold {:.2f}; saved best weights to {} and best model to {}".format(
                    epoch + 1,
                    fullimg_dice,
                    best_threshold,
                    self.weights_path,
                    self.keras_path,
                )
            )

    def _flush_history_csv(self):
        with open(self.metrics_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "best_threshold",
                    "fullimg_dice",
                    "fullimg_iou",
                    "fullimg_precision",
                    "fullimg_recall",
                ],
            )
            writer.writeheader()
            writer.writerows(self.history_rows)


# =========================
# Main
# =========================
def load_training_data(npy_dir):
    imgs = np.load(os.path.join(npy_dir, "imgs_train.npy")).astype(np.float32)
    masks = np.load(os.path.join(npy_dir, "imgs_mask_train.npy")).astype(np.float32)

    if imgs.ndim == 3:
        imgs = imgs[..., np.newaxis]
    if masks.ndim == 3:
        masks = masks[..., np.newaxis]

    masks = (masks > 0.5).astype(np.float32)

    if imgs.max() > 1.5:
        imgs /= 255.0

    img_means = imgs.mean(axis=(1, 2, 3), keepdims=True)
    imgs = imgs - img_means

    return imgs, masks


def save_final_summary(summary_path, summary_dict):
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_dict.keys()))
        writer.writeheader()
        writer.writerow(summary_dict)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--npy_dir", type=str, default=NPY_DIR, help="Directory with imgs_train.npy and imgs_mask_train.npy")
    p.add_argument("--model_dir", type=str, default=MODEL_DIR, help="Directory to save model files")
    p.add_argument("--log_dir", type=str, default=LOG_DIR, help="Directory to save logs")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    p.add_argument("--steps_per_epoch", type=int, default=STEPS_PER_EPOCH, help="Steps per epoch")
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    p.add_argument("--patch", type=int, default=PATCH, help="Training patch size")
    p.add_argument("--infer_stride", type=int, default=INFER_STRIDE, help="Stride for tiled full-image inference")
    p.add_argument("--fg_weight", type=float, default=0.8, help="Sampling weight for foreground patches")
    p.add_argument("--fg_ratio_min", type=float, default=0.05, help="Minimum foreground ratio used in patch extraction")
    p.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    p.add_argument(
        "--run_name",
        type=str,
        default="final_Mito2_resunet_scse",
        help="Prefix used for output files",
    )
    p.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=THRESHOLD_CANDIDATES,
        help="Threshold candidates for full-image self-check",
    )
    p.add_argument(
        "--early_stop_patience",
        type=int,
        default=20,
        help="Early stopping patience monitored on train_fullimg_dice",
    )
    p.add_argument(
        "--reduce_lr_patience",
        type=int,
        default=5,
        help="ReduceLROnPlateau patience monitored on train_fullimg_dice",
    )
    return p.parse_args()


def main(args):
    global PATCH, INFER_STRIDE, BATCH_SIZE, EPOCHS, STEPS_PER_EPOCH, LEARNING_RATE

    PATCH = int(args.patch)
    INFER_STRIDE = int(args.infer_stride)
    BATCH_SIZE = int(args.batch_size)
    EPOCHS = int(args.epochs)
    STEPS_PER_EPOCH = int(args.steps_per_epoch)
    LEARNING_RATE = float(args.lr)

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    set_global_seed(args.seed)
    tf.keras.backend.clear_session()

    imgs, masks = load_training_data(args.npy_dir)

    print("[Data] imgs :", imgs.shape, imgs.dtype, "min/max:", float(imgs.min()), float(imgs.max()))
    print("[Data] masks:", masks.shape, masks.dtype, "foreground ratio:", float(masks.mean()))
    print(
        "[Train] patch={}, infer_stride={}, batch_size={}, steps_per_epoch={}, epochs={}".format(
            PATCH,
            INFER_STRIDE,
            BATCH_SIZE,
            STEPS_PER_EPOCH,
            EPOCHS,
        )
    )
    print("[Eval] thresholds={}".format(args.thresholds))

    train_dataset = create_balanced_train_dataset(
        imgs,
        masks,
        patch=PATCH,
        batch_size=BATCH_SIZE,
        fg_weight=args.fg_weight,
        fg_ratio_min=args.fg_ratio_min,
    )

    model = build_model(input_shape=(PATCH, PATCH, 1), compile_model=True)

    weights_path = os.path.join(args.model_dir, "{}.weights.h5".format(args.run_name))
    keras_path = os.path.join(args.model_dir, "{}.keras".format(args.run_name))
    csv_log_path = os.path.join(args.log_dir, "{}_train_log.csv".format(args.run_name))
    fullimg_csv_path = os.path.join(args.log_dir, "{}_fullimg_history.csv".format(args.run_name))
    summary_csv_path = os.path.join(args.log_dir, "{}_summary.csv".format(args.run_name))
    threshold_csv_path = os.path.join(args.log_dir, "{}_threshold_sweep.csv".format(args.run_name))
    per_image_csv_path = os.path.join(args.log_dir, "{}_per_image.csv".format(args.run_name))

    best_saver = FullImageBestSaver(
        x_data=imgs,
        y_data=masks,
        weights_path=weights_path,
        keras_path=keras_path,
        metrics_csv_path=fullimg_csv_path,
        patch=PATCH,
        stride=INFER_STRIDE,
        thresholds=args.thresholds,
    )

    callbacks = [
        best_saver,
        CSVLogger(csv_log_path),
        ReduceLROnPlateau(
            monitor="train_fullimg_dice",
            mode="max",
            factor=0.5,
            patience=args.reduce_lr_patience,
            min_lr=1e-7,
            verbose=1,
        ),
        EarlyStopping(
            monitor="train_fullimg_dice",
            mode="max",
            patience=args.early_stop_patience,
            restore_best_weights=False,
            verbose=1,
        ),
    ]

    model.fit(
        train_dataset,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    best_model = build_model(input_shape=(PATCH, PATCH, 1), compile_model=False)
    best_model.load_weights(weights_path)

    best_row, per_image_rows, threshold_rows = evaluate_on_full_images(
        best_model,
        imgs,
        masks,
        patch=PATCH,
        stride=INFER_STRIDE,
        thresholds=args.thresholds,
    )

    with open(threshold_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["threshold", "dice", "iou", "precision", "recall"],
        )
        writer.writeheader()
        writer.writerows(threshold_rows)

    with open(per_image_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_index", "dice", "iou", "precision", "recall"],
        )
        writer.writeheader()
        writer.writerows(per_image_rows)

    summary_row = {
        "run_name": args.run_name,
        "best_epoch": int(best_saver.best_epoch),
        "best_threshold": float(best_saver.best_threshold),
        "fullimg_dice": float(best_row["dice"]),
        "fullimg_iou": float(best_row["iou"]),
        "fullimg_precision": float(best_row["precision"]),
        "fullimg_recall": float(best_row["recall"]),
        "weights_path": weights_path,
        "keras_path": keras_path,
    }
    save_final_summary(summary_csv_path, summary_row)

    print("\n" + "=" * 80)
    print("Final deployment training finished")
    print("Best epoch      : {}".format(best_saver.best_epoch))
    print("Best threshold  : {:.2f}".format(best_saver.best_threshold))
    print("Full-image Dice : {:.6f}".format(best_row["dice"]))
    print("Full-image IoU  : {:.6f}".format(best_row["iou"]))
    print("Precision       : {:.6f}".format(best_row["precision"]))
    print("Recall          : {:.6f}".format(best_row["recall"]))
    print("[Saved] {}".format(weights_path))
    print("[Saved] {}".format(keras_path))
    print("[Saved] {}".format(csv_log_path))
    print("[Saved] {}".format(fullimg_csv_path))
    print("[Saved] {}".format(threshold_csv_path))
    print("[Saved] {}".format(per_image_csv_path))
    print("[Saved] {}".format(summary_csv_path))
    print("=" * 80)


if __name__ == "__main__":
    args = parse_args()
    main(args)

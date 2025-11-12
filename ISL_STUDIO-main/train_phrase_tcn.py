import os
import json
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


def extract_both_hands_keypoints(frame_bgr, holistic) -> np.ndarray:
    """Return 126-dim vector: [right(63), left(63)] with zeros if missing."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = holistic.process(rgb)
    def to63(hand_landmarks):
        if not hand_landmarks:
            return np.zeros(21 * 3, dtype=np.float32)
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32).reshape(-1)
    right = to63(res.right_hand_landmarks)
    left = to63(res.left_hand_landmarks)
    return np.concatenate([right, left]).astype(np.float32)


def load_video_dataset_both_hands(video_dir: Path, max_frames: int = 30) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Scan class subfolders for .mp4/.mov videos and extract (N, T, 126) and string labels."""
    mp = None
    import mediapipe as mp_mod
    mp = mp_mod
    holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.5)

    X, y = [], []
    classes = []
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    for label in sorted([d for d in os.listdir(video_dir) if (video_dir / d).is_dir()]):
        classes.append(label)
        class_dir = video_dir / label
        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith((".mp4", ".mov", ".mkv")):
                continue
            path = str(class_dir / fname)
            cap = cv2.VideoCapture(path)
            frames = []
            while cap.isOpened() and len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                kp = extract_both_hands_keypoints(frame, holistic)
                frames.append(kp)
            cap.release()
            if len(frames) == 0:
                continue
            # pad/truncate to max_frames
            if len(frames) < max_frames:
                pad = [np.zeros(126, dtype=np.float32) for _ in range(max_frames - len(frames))]
                frames.extend(pad)
            elif len(frames) > max_frames:
                frames = frames[:max_frames]
            arr = np.asarray(frames, dtype=np.float32)
            if arr.shape == (max_frames, 126):
                X.append(arr)
                y.append(label)
    holistic.close()
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    return X, y, sorted(set(y.tolist()))


def standardize_per_sample(X: np.ndarray) -> np.ndarray:
    Xf = X.astype(np.float32)
    mean = Xf.mean(axis=(1, 2), keepdims=True)
    std = Xf.std(axis=(1, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (Xf - mean) / std


def augment_sequence_noise(X: np.ndarray, factor: int = 2) -> np.ndarray:
    if factor <= 1:
        return X
    rng = np.random.default_rng(42)
    out = [X]
    for _ in range(factor - 1):
        noise = rng.normal(0, 0.01, size=X.shape).astype(np.float32)
        out.append(X + noise)
    return np.concatenate(out, axis=0)


def build_tcn(input_shape: tuple, num_classes: int, lr: float = 1e-3) -> tf.keras.Model:
    inp = layers.Input(shape=input_shape)  # (T, F)
    x = inp
    filters = 128
    drop = 0.2
    dilations = [1, 2, 4, 8]
    for d in dilations:
        res = x
        x = layers.Conv1D(filters, kernel_size=3, padding='causal', dilation_rate=d,
                          kernel_regularizer=regularizers.l2(1e-6))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(drop)(x)
        x = layers.Conv1D(filters, kernel_size=3, padding='causal', dilation_rate=d,
                          kernel_regularizer=regularizers.l2(1e-6))(x)
        x = layers.BatchNormalization()(x)
        # match residual dims
        if res.shape[-1] != x.shape[-1]:
            res = layers.Conv1D(filters, kernel_size=1, padding='same')(res)
        x = layers.Add()([x, res])
        x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train both-hands TCN phrase model.")
    parser.add_argument('--video_dir', type=str, required=True, help='Dataset root with class folders containing videos.')
    parser.add_argument('--out', type=str, default='isl_model_tcn.keras', help='Output model path (.keras).')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_frames', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--augment', type=int, default=1, help='Augmentation replication factor.')
    args = parser.parse_args()

    os.environ["PYTHONHASHSEED"] = "42"
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)

    video_dir = Path(args.video_dir)
    print(f"Loading videos from {video_dir} ...")
    X, y, label_names = load_video_dataset_both_hands(video_dir, max_frames=args.max_frames)
    if len(X) == 0:
        raise SystemExit("No samples found. Please check the dataset path.")
    print(f"Dataset: X={X.shape}, y={y.shape}, classes={len(label_names)}")

    X = standardize_per_sample(X)
    if args.augment and args.augment > 1:
        X_aug = augment_sequence_noise(X, factor=args.augment)
        y_aug = np.repeat(y, args.augment, axis=0)
    else:
        X_aug, y_aug = X, y

    # Label encoding
    label_map = {label: idx for idx, label in enumerate(sorted(set(y_aug.tolist())))}
    y_idx = np.array([label_map[v] for v in y_aug], dtype=np.int64)
    y_oh = tf.keras.utils.to_categorical(y_idx, num_classes=len(label_map))

    X_train, X_val, y_train, y_val = train_test_split(
        X_aug, y_oh, test_size=0.2, stratify=y_idx, random_state=42
    )

    class_weights = compute_class_weight(
        class_weight='balanced', classes=np.arange(len(label_map)), y=np.argmax(y_train, axis=1)
    )
    class_weights = {i: float(w) for i, w in enumerate(class_weights)}

    model = build_tcn(input_shape=(X.shape[1], X.shape[2]), num_classes=len(label_map), lr=args.lr)
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(args.out, monitor='val_accuracy', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1),
    ]

    print("Training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cbs,
        class_weight=class_weights,
        verbose=1,
    )

    print("Evaluating...")
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc:.3f}")

    y_true = np.argmax(y_val, axis=1)
    y_prob = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    report = classification_report(y_true, y_pred, target_names=[k for k, _ in sorted(label_map.items(), key=lambda x: x[1])], digits=3)
    cm = confusion_matrix(y_true, y_pred)

    out_path = Path(args.out)
    with open(out_path.with_suffix('.label_map.json'), 'w') as f:
        json.dump({i: name for name, i in label_map.items()}, f, indent=2)
    with open(out_path.with_suffix('.report.txt'), 'w') as f:
        f.write(f"Validation accuracy: {val_acc:.3f}\n\n")
        f.write(report)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(cm.shape[1]))
        ax.set_yticks(range(cm.shape[0]))
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
        fig.tight_layout()
        fig.savefig(str(out_path.with_suffix('.cm.png')), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Could not save confusion matrix plot: {e}")

    print(f"Saved TCN model to: {out_path}")


if __name__ == '__main__':
    main()


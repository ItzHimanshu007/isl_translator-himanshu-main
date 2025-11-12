import os
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from typing import List, Optional


def calc_landmark_list(image, landmarks):
    h, w = image.shape[0], image.shape[1]
    landmark_point = []
    for lm in landmarks.landmark:
        x = float(np.clip(lm.x * w, 0.0, w - 1.0))
        y = float(np.clip(lm.y * h, 0.0, h - 1.0))
        landmark_point.append([x, y])
    return landmark_point


def pre_process_landmark(landmark_list, handedness_label: Optional[str] = None):
    """Center on wrist, rotate to canonical orientation, mirror left to right, normalize to [-1,1]."""
    pts = np.asarray(landmark_list, dtype=np.float32).copy()  # (21,2)
    # Center at wrist (id 0)
    base = pts[0].copy()
    pts -= base
    # Compute orientation using wrist -> middle_mcp (id 9)
    v = pts[9].copy()
    angle = np.arctan2(v[1], v[0])
    c, s = np.cos(-angle), np.sin(-angle)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    pts = pts @ rot.T
    # Mirror left to match right-hand canonical if label says Left
    if handedness_label is not None and handedness_label.lower().startswith('left'):
        pts[:, 0] = -pts[:, 0]
    # Normalize by max abs
    flat = pts.reshape(-1)
    max_val = float(np.max(np.abs(flat))) if flat.size else 1.0
    if max_val < 1e-6:
        max_val = 1.0
    flat = (flat / max_val).astype(np.float32)
    return flat


def load_dataset_from_images(data_dir: Path, classes: List[str], min_detect_conf=0.5, max_per_class: Optional[int] = None):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=min_detect_conf,
    )

    X, y, kept, skipped = [], [], 0, 0
    for label in classes:
        class_dir = data_dir / label
        if not class_dir.exists():
            print(f"[warn] Missing class folder: {class_dir}")
            continue
        file_names = [f for f in sorted(os.listdir(class_dir)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if max_per_class is not None and max_per_class > 0:
            file_names = file_names[:max_per_class]
        for fname in file_names:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = str(class_dir / fname)
            img = cv2.imread(fpath)
            if img is None:
                skipped += 1
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)
            if not result.multi_hand_landmarks:
                skipped += 1
                continue
            lm = result.multi_hand_landmarks[0]
            lm_list = calc_landmark_list(img, lm)
            # Handedness label if available
            handed = None
            try:
                if result.multi_handedness:
                    handed = result.multi_handedness[0].classification[0].label  # 'Left' or 'Right'
            except Exception:
                handed = None
            if len(lm_list) != 21:
                skipped += 1
                continue
            feat = pre_process_landmark(lm_list, handed)  # 42 features
            X.append(feat)
            y.append(label)
            kept += 1
    hands.close()
    print(f"Loaded samples: {kept} (skipped: {skipped})")
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    return X, y


def build_classifier(input_dim: int, num_classes: int, lr: float = 1e-3) -> tf.keras.Model:
    from tensorflow.keras import layers, regularizers
    model = tf.keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-6)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-6)),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def augment_features(X: np.ndarray, y: np.ndarray, y_idx: np.ndarray, target_classes: List[int], factor: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create augmented copies for selected classes with light rotation/scale/noise."""
    if not target_classes or factor <= 1:
        return X, y, y_idx
    rng = np.random.default_rng(42)
    X_list = [X]
    y_list = [y]
    yidx_list = [y_idx]
    sel_mask = np.isin(y_idx, np.array(target_classes))
    X_sel = X[sel_mask]
    y_sel = y[sel_mask]
    yidx_sel = y_idx[sel_mask]
    for _ in range(factor - 1):
        aug = []
        for feat in X_sel:
            pts = feat.reshape(21, 2)
            # small random rotation
            deg = rng.uniform(-12, 12)
            rad = np.deg2rad(deg)
            c, s = np.cos(rad), np.sin(rad)
            rot = np.array([[c, -s], [s, c]], dtype=np.float32)
            pts2 = pts @ rot.T
            # scale and noise
            scale = rng.uniform(0.9, 1.1)
            pts2 *= scale
            pts2 += rng.normal(0, 0.01, pts2.shape)
            # re-normalize
            flat = pts2.reshape(-1)
            m = np.max(np.abs(flat))
            if m < 1e-6:
                m = 1.0
            flat = (flat / m).astype(np.float32)
            aug.append(flat)
        X_list.append(np.array(aug, dtype=np.float32))
        y_list.append(y_sel.copy())
        yidx_list.append(yidx_sel.copy())
    X_aug = np.concatenate(X_list, axis=0)
    y_aug = np.concatenate(y_list, axis=0)
    yidx_aug = np.concatenate(yidx_list, axis=0)
    return X_aug, y_aug, yidx_aug


def main():
    parser = argparse.ArgumentParser(description="Train ISL character model from image dataset using MediaPipe landmarks.")
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset root (folders 1-9 and A-Z).')
    parser.add_argument('--out', type=str, default=str(Path(__file__).resolve().parent / 'models' / 'model.h5'), help='Output model path (.h5).')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val_size', type=float, default=0.2)
    parser.add_argument('--max_per_class', type=int, default=400, help='Limit images per class (0 or negative = no limit).')
    parser.add_argument('--es_patience', type=int, default=8, help='EarlyStopping patience.')
    parser.add_argument('--rlrop_patience', type=int, default=3, help='ReduceLROnPlateau patience.')
    parser.add_argument('--rlrop_factor', type=float, default=0.5, help='ReduceLROnPlateau factor.')
    parser.add_argument('--min_lr', type=float, default=1e-5, help='Minimum learning rate for scheduling.')
    parser.add_argument('--out_keras', type=str, default=None, help='Optional additional path to save native Keras format (.keras).')
    parser.add_argument('--boost_classes', type=str, default='A,B,H,J,Q,R', help='Comma-separated list of class labels to oversample/augment.')
    parser.add_argument('--boost_factor', type=int, default=3, help='Augmentation replication factor for boosted classes.')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Dataset not found: {data_dir}")

    # Expected class order must match the app's inference mapping
    classes = [str(i) for i in range(1, 10)] + [chr(c) for c in range(ord('A'), ord('Z') + 1)]
    present = [c for c in classes if (data_dir / c).exists()]
    missing = [c for c in classes if c not in present]
    if missing:
        print(f"[warn] Missing classes (will be ignored): {missing}")
    if not present:
        raise SystemExit("No class folders found under dataset root.")

    print("Extracting landmarks and building dataset...")
    mpc = None if args.max_per_class is None or args.max_per_class <= 0 else int(args.max_per_class)
    X, y = load_dataset_from_images(data_dir, present, max_per_class=mpc)
    if len(X) == 0:
        raise SystemExit("No samples extracted. Check dataset and MediaPipe installation.")

    # Map labels to indices following the defined order (present subset only)
    label_to_idx = {label: i for i, label in enumerate(present)}
    y_idx = np.array([label_to_idx[label] for label in y], dtype=np.int64)

    # One-hot
    num_classes = len(present)
    y_oh = tf.keras.utils.to_categorical(y_idx, num_classes=num_classes)

    # Targeted class boosting (oversample + augment)
    boost_labels = [lab.strip() for lab in (args.boost_classes or '').split(',') if lab.strip()]
    boost_indices = [label_to_idx[lab] for lab in boost_labels if lab in label_to_idx]
    if boost_indices:
        print(f"Boosting classes {boost_labels} by factor {args.boost_factor}")
        X, y_oh, y_idx = augment_features(X, y_oh, y_idx, boost_indices, factor=max(1, int(args.boost_factor)))

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_oh, test_size=args.val_size, random_state=42, stratify=y_idx
    )

    # Class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(num_classes),
        y=np.argmax(y_train, axis=1)
    )
    class_weights = {i: float(w) for i, w in enumerate(class_weights)}

    print("Building model...")
    model = build_classifier(input_dim=X.shape[1], num_classes=num_classes, lr=args.lr)

    ckpt_path = args.out
    os.makedirs(Path(ckpt_path).parent, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.es_patience, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=args.rlrop_factor, patience=args.rlrop_patience, min_lr=args.min_lr, verbose=1),
    ]

    print("Training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    print("Evaluating...")
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc:.3f}")

    # Reports
    y_true = np.argmax(y_val, axis=1)
    y_prob = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    report = classification_report(y_true, y_pred, target_names=present, digits=3)
    cm = confusion_matrix(y_true, y_pred)

    # Save artifacts next to model
    out_dir = Path(ckpt_path).parent
    (out_dir / 'char_label_map.json').write_text(json.dumps({i: lab for lab, i in label_to_idx.items()}, indent=2))
    (out_dir / 'char_training_report.txt').write_text(
        f"Validation accuracy: {val_acc:.3f}\n\n{report}\n"
    )
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(len(present)))
        ax.set_yticks(range(len(present)))
        ax.set_xticklabels(present, rotation=90)
        ax.set_yticklabels(present)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
        fig.tight_layout()
        fig.savefig(str(out_dir / 'char_confusion_matrix.png'), dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Could not save confusion matrix plot: {e}")

    # Optionally save native Keras format
    if args.out_keras:
        try:
            model.save(args.out_keras)
            print(f"Saved native Keras model to: {args.out_keras}")
        except Exception as e:
            print(f"Could not save native Keras model: {e}")

    print(f"Saved model to: {ckpt_path}")


if __name__ == '__main__':
    main()

import os
import json
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from data_preparation import process_videos_to_dataset, normalize_landmarks, augment_landmarks, prepare_labels

def build_model(input_shape, num_classes):
    """
    Build a CNN-LSTM model for video classification.
    """
    model = Sequential([
        LSTM(128, return_sequences=True, activation='relu', input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, activation='relu', kernel_regularizer=l2(1e-5)),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=l2(1e-5)),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def main():
    # Reproducibility
    os.environ["PYTHONHASHSEED"] = "42"
    random.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    video_dir = 'Greetings'  # Replace with your video directory path
    model_save_path = 'isl_model.keras'
    max_frames = 30
    #num_classes = 9
    batch_size = 16
    epochs = 50

    print("Loading and preprocessing data...")
    X_data, y_data = process_videos_to_dataset(video_dir, max_frames)

    X_data = normalize_landmarks(X_data)

    # Augment data
    X_data_augmented = augment_landmarks(X_data)
    y_data_augmented = np.repeat(y_data, 3, axis=0)  # Repeat labels for each augmented sample

    # One-hot encode labels
    label_names = sorted(set(y_data.tolist()))
    num_classes = len(label_names)
    y_data_augmented = prepare_labels(y_data_augmented, num_classes)

    # Split into train and test sets
    from sklearn.model_selection import train_test_split
    #print(f"X_data shape: {X_data.shape}, y_data shape: {len(y_data)}")
    X_train, X_test, y_train, y_test = train_test_split(X_data_augmented, y_data_augmented, test_size=0.2, random_state=42)
    #print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    #print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

    print("Building model...")
    input_shape = (X_train.shape[1], X_train.shape[2]) #X_train.shape[1:]  # (timesteps, features)
    model = build_model(input_shape, num_classes)

    # Callbacks for saving the model and early stopping
    checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)

    # Use more advanced callbacks
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=3, 
        min_lr=0.00001
    )
    
    # Add class weights if slightly imbalanced
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(np.argmax(y_train, axis=1)), 
        y=np.argmax(y_train, axis=1)
    )
    class_weights = dict(enumerate(class_weights))

    print("Starting training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint, early_stopping, reduce_lr],
        class_weight=class_weights
    )

    print(f"Training complete. Model saved to {model_save_path}.")
    print("Evaluating model on test data...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.3f}")

    # Detailed report
    y_true = np.argmax(y_test, axis=1)
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    report = classification_report(y_true, y_pred, target_names=label_names, digits=3)
    cm = confusion_matrix(y_true, y_pred)

    # Persist artifacts
    with open('training_report.txt', 'w') as f:
        f.write(f"Test accuracy: {test_accuracy:.3f}\n\n")
        f.write(report)

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(len(label_names)))
        ax.set_yticks(range(len(label_names)))
        ax.set_xticklabels(label_names, rotation=45, ha='right')
        ax.set_yticklabels(label_names)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
        fig.tight_layout()
        fig.savefig('confusion_matrix.png', dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Could not save confusion matrix plot: {e}")

    # Save label mapping
    with open('label_map.json', 'w') as f:
        json.dump({i: name for i, name in enumerate(label_names)}, f, indent=2)

if __name__ == "__main__":
    main()


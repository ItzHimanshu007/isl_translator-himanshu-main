import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from data_preparation import process_videos_to_dataset, normalize_landmarks, augment_landmarks, prepare_labels'

def build_model(input_shape, num_classes):
    """
    Build a CNN-LSTM model for video classification.
    """
    model = Sequential([
        LSTM(128, return_sequences=True, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def main():
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
    num_classes = len(set(y_data))
    y_data_augmented= prepare_labels(y_data_augmented, num_classes)

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
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

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
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.2f}")

if __name__ == "__main__":
    main()



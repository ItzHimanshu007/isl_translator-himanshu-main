import cv2
import numpy as np
import os
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks_from_video(video_path, max_frames=30):
    """
    Extract normalized hand landmarks from a video file using MediaPipe.
    """
    cap = cv2.VideoCapture(video_path)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    
    landmarks_list = []
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            # Flatten the landmarks from both hands
            for hand_landmarks in result.multi_hand_landmarks:
                single_hand_landmarks = []
                for lm in hand_landmarks.landmark:
                    single_hand_landmarks.extend([lm.x, lm.y, lm.z])
                landmarks_list.append(single_hand_landmarks)

        frame_count += 1
        if frame_count >= max_frames:
            break

    cap.release()
    hands.close()

    # Convert landmarks to numpy array
    landmarks_array = np.array(landmarks_list)

    return landmarks_array if len(landmarks_array) > 0 else None

def pad_or_truncate_landmarks(landmarks, max_frames):
    """
    Ensure the landmarks have a fixed number of frames (pad or truncate).
    """
    if landmarks is None or len(landmarks) == 0:
        return np.zeros((max_frames, 63))  # Assuming 21 landmarks * 3 coordinates

    # Truncate
    if len(landmarks) > max_frames:
        return landmarks[:max_frames]
    
    # Pad with zeros
    padding = np.zeros((max_frames - len(landmarks), landmarks.shape[1]))
    return np.vstack([landmarks, padding])

def preprocess_landmarks(landmarks, max_frames=30):
    """
    Reshape landmarks to (timesteps, features).
    """
    if landmarks is None or len(landmarks) == 0:
        # Return zero-padded array with shape (max_frames, 63) for 21 landmarks * 3 coordinates
        return np.zeros((max_frames, 63))
    
    # Reshape landmarks to ensure (samples, max_frames, features)
    reshaped_landmarks = []
    for sample in landmarks:
        # Truncate or pad each sample
        padded_sample = pad_or_truncate_landmarks(sample.reshape(-1, 63), max_frames)
        reshaped_landmarks.append(padded_sample)
    
    return np.array(reshaped_landmarks)

def process_videos_to_dataset(video_dir, max_frames=30):
    """
    Process videos in a directory to create a dataset of landmarks and labels.
    """
    hands = mp_hands.Hands()
    
    X_data = []
    y_data = []

    # Ensure the video directory exists
    if not os.path.exists(video_dir):
        raise ValueError(f"Video directory '{video_dir}' does not exist.")
    
    for label in os.listdir(video_dir):
        video_path = os.path.join(video_dir, label)
        
        # Ensure the directory contains videos
        if not os.path.isdir(video_path):
            continue
        
        for video_file in os.listdir(video_path):
            if video_file.endswith('.MOV') or video_file.endswith('.mp4'):  # Adjust extensions as needed
                video_file_path = os.path.join(video_path, video_file)
                print(f"Processing video: {video_file_path}")
                
                cap = cv2.VideoCapture(video_file_path)
                frames = []
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Convert the frame to RGB (MediaPipe requires RGB format)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Get hand landmarks
                    results = hands.process(frame_rgb)
                    frame_landmarks = []
                    if results.multi_hand_landmarks:
                        # Assuming one hand per frame, you can handle multiple if needed
                        landmarks = results.multi_hand_landmarks[0]
                        frame_landmarks = []
                        for landmark in landmarks.landmark:
                            frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
                        frames.append(frame_landmarks)
                    else:
                        frames.append([0]*63)  # 21 landmarks with 3 coordinates each (x, y, z)
                
                    # Break if we've reached max_frames
                    if len(frames) == max_frames:
                        break
                cap.release()

                if len(frames) > 0:
                    # Ensure exactly max_frames
                    if len(frames) < max_frames:
                        # Pad with zero frames
                        padding = [[0]*63] * (max_frames - len(frames))
                        frames.extend(padding)
                    elif len(frames) > max_frames:
                        # Truncate
                        frames = frames[:max_frames]
                    
                    # Ensure shape is exactly (max_frames, 63)
                    frames = np.array(frames)
                    
                    if frames.shape == (max_frames, 63):
                        X_data.append(frames)
                        y_data.append(label)
                    else:
                        print(f"Skipping video {video_file_path}: Incorrect frame shape {frames.shape}")

    X_data = np.array(X_data)
    y_data = np.array(y_data)

    # Check shapes before reshaping or preprocessing
    print(f"Shape of X_data before preprocessing: {X_data.shape}")
    print(f"Shape of y_data before preprocessing: {y_data.shape}")

    # Flatten X_data along the last two dimensions (if necessary) while retaining the sample count
    return X_data, y_data

def normalize_landmarks(X_data):
    """
    Standardize landmarks per sample to zero mean and unit variance.
    More robust than global max-scaling and avoids leakage across splits.
    Shape: (samples, timesteps, features)
    """
    X = X_data.astype(np.float32)
    mean = X.mean(axis=(1, 2), keepdims=True)
    std = X.std(axis=(1, 2), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (X - mean) / std

def prepare_labels(y_data, num_classes):
    """
    Convert numerical labels into one-hot encoding.
    """
    from keras.utils import to_categorical
    # Map string labels to numeric labels
    label_mapping = {label: idx for idx, label in enumerate(sorted(set(y_data)))}
    
    # Convert string labels to numeric labels
    y_data_numeric = [label_mapping[label] for label in y_data]
    
    # Convert numeric labels to one-hot encoding
    return to_categorical(y_data_numeric, num_classes=num_classes)

def augment_landmarks(X_data, augmentation_factor=2):
    """
    Simple data augmentation for landmark data
    """
    augmented_data = []
    for sample in X_data:
        # Original sample
        augmented_data.append(sample)
        
        # Slight noise addition
        noisy_sample = sample + np.random.normal(0, 0.01, sample.shape)
        augmented_data.append(noisy_sample)
        
        # Slight scaling
        if augmentation_factor > 1:
            scaled_sample = sample * (1 + np.random.uniform(-0.1, 0.1, 1)[0])
            augmented_data.append(scaled_sample)
    
    return np.array(augmented_data)

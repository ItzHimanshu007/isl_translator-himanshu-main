import os
import numpy as np
import mediapipe as mp
import cv2

class VideoPreprocessor:
    def __init__(self, target_frames=20):
        self.target_frames = target_frames
        self.mp_holistic = mp.solutions.holistic

    def extract_landmarks(self, video_path, target_size=(100, 100)):
        """
        Extract landmarks from a video using MediaPipe Holistic.
        
        Args:
            video_path (str): Path to the video file.
            target_size (tuple): Resizing dimensions for the video frame.
        
        Returns:
            np.ndarray: Extracted landmarks (frames x landmarks x 3).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file {video_path}")

        landmarks_list = []

        with self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=True,
        ) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize and convert to RGB
                frame = cv2.resize(frame, target_size)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                results = holistic.process(frame_rgb)
                frame_landmarks = []

                # Define the expected number of landmarks
                expected_pose = 33  # Pose landmarks
                expected_face = 468  # Face landmarks
                expected_hand = 21  # Hand landmarks

                # Extract pose landmarks
                pose_landmarks = (
                    [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
                    if results.pose_landmarks
                    else [[0, 0, 0]] * expected_pose
                )
                frame_landmarks.extend(pose_landmarks)

                # Extract face landmarks
                face_landmarks = (
                    [[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]
                    if results.face_landmarks
                    else [[0, 0, 0]] * expected_face
                )
                frame_landmarks.extend(face_landmarks)

                # Extract hand landmarks
                left_hand_landmarks = (
                    [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
                    if results.left_hand_landmarks
                    else [[0, 0, 0]] * expected_hand
                )
                right_hand_landmarks = (
                    [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
                    if results.right_hand_landmarks
                    else [[0, 0, 0]] * expected_hand
                )
                frame_landmarks.extend(left_hand_landmarks)
                frame_landmarks.extend(right_hand_landmarks)

                # Append frame landmarks
                landmarks_list.append(frame_landmarks)

        cap.release()

        # Ensure fixed number of frames by truncating or padding
        if len(landmarks_list) > self.target_frames:
            landmarks_list = landmarks_list[:self.target_frames]
        else:
            padding = [np.zeros_like(landmarks_list[0])] * (self.target_frames - len(landmarks_list))
            landmarks_list.extend(padding)

        return np.array(landmarks_list)

    def preprocess_videos(self, input_dir, output_dir):
        """
        Process all videos in the input directory and save landmarks as .npy files.
        
        Args:
            input_dir (str): Path to directory containing sign videos.
            output_dir (str): Path to save preprocessed .npy files.
        """
        os.makedirs(output_dir, exist_ok=True)

        for label in os.listdir(input_dir):
            label_path = os.path.join(input_dir, label)
            if not os.path.isdir(label_path):
                continue

            output_label_dir = os.path.join(output_dir, label)
            os.makedirs(output_label_dir, exist_ok=True)

            for video_file in os.listdir(label_path):
                video_path = os.path.join(label_path, video_file)
                try:
                    landmarks = self.extract_landmarks(video_path)
                    output_file = os.path.join(output_label_dir, f"{os.path.splitext(video_file)[0]}.npy")
                    np.save(output_file, landmarks)
                    print(f"Processed {video_file} -> {output_file}")
                except Exception as e:
                    print(f"Failed to process {video_file}: {e}")

# Example usage
if __name__ == "__main__":
    preprocessor = VideoPreprocessor(target_frames=20)
    preprocessor.preprocess_videos(input_dir="Greetings", output_dir="Keypoints")

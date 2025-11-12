import streamlit as st
import cv2
import mediapipe as mp

def count_fingers(hand_landmarks):
    """Count number of fingers raised"""
    # Finger tip and base landmarks
    finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    finger_base = [3, 6, 10, 14, 18]
    
    fingers_raised = 0
    
    # Special check for thumb (different from other fingers)
    thumb_is_raised = hand_landmarks[finger_tips[0]].y < hand_landmarks[finger_base[0]].y
    
    # Check other fingers
    for i in range(1, 5):
        tip = hand_landmarks[finger_tips[i]]
        base = hand_landmarks[finger_base[i]]
        finger_is_raised = tip.y < base.y
        
        if finger_is_raised:
            fingers_raised += 1
    
    # Add thumb if raised
    if thumb_is_raised:
        fingers_raised += 1
    
    return fingers_raised

def main():
    st.title("Finger Counter")
    
    # MediaPipe hands setup
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7
    )
    
    # Video capture
    run = st.checkbox('Start Camera')
    FRAME_WINDOW = st.image([])
    finger_count_display = st.empty()
    
    if run:
        camera = cv2.VideoCapture(0)
        while run:
            ret, frame = camera.read()
            if not ret:
                break

            # Flip and convert frame
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Count fingers
                    fingers = count_fingers(hand_landmarks.landmark)
                    
                    # Display finger count
                    finger_count_display.write(f"Fingers Raised: {fingers}")
                    
                    # Optional: Visualize landmarks
                    for landmark in hand_landmarks.landmark:
                        h, w, _ = frame.shape
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            
            # Display frame
            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        camera.release()

if __name__ == "__main__":
    main()
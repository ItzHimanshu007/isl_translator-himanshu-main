import cv2
import numpy as np
import os
import mediapipe as mp
import time
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from keras.callbacks import TensorBoard
import threading

# Initialize mediapipe and other variables
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
model = load_model('isl_model.keras')

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

actions = np.array(['Alright', 'Good afternoon', 'Good evening', 'Good morning', 'Good night', 'Hello', 'How are you', 'Pleased', 'Thank you'])
signs = {'English': actions}

def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image_rgb)
    return image, results

def draw_styled_landmarks(image, results):
    # Only draw landmarks if they exist
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4), 
                                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    # Safely extract keypoints, handling potential None values
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    keypoints = np.concatenate([pose, face, lh, rh])[:63]
    return keypoints

def prob_viz(res, actions, image):
    colors = [(245,117,16) for _ in range(len(actions))]
    for num, prob in enumerate(res):
        cv2.rectangle(image, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
        cv2.putText(image, f"{actions[num]}: {prob:.2f}", (0, 85 + num * 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return image

def run_openCV(language):
    global sequence, sentence, predictions, translation, last_sign

    # Initialize sequence and variables
    sequence = []
    threshold = 0.7  # Confidence threshold
    predictions = []
    sentence = []
    translation = []
    last_sign = None
    start_time = time.time()

    # Create translation window
    translation_window = tk.Toplevel(root)
    translation_window.title("Translated Sign Words")
    
    # Calculate window dimensions and position
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = screen_width // 2
    window_height = 100
    window_x = (screen_width - window_width) // 2
    window_y = screen_height - window_height - 50
    
    translation_window.geometry(f"{window_width}x{window_height}+{window_x}+{window_y}")
    translation_window.resizable(False, False)
    translation_window.attributes("-topmost", True)
    
    translated_label = tk.Label(translation_window, text="", font=("Arial", 14))
    translated_label.pack(pady=20)

    # Try multiple camera indices
    camera_indices = [0, 1, 2]
    cap = None
    for idx in camera_indices:
        #cap = cv2.VideoCapture(idx)
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            break
    
    if not cap or not cap.isOpened():
        messagebox.showerror("Error", "Could not open camera!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with mp_holistic.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            
            image, results = mediapipe_detection(frame, holistic)
            
            # Only process if hand landmarks are detected
            if results.left_hand_landmarks or results.right_hand_landmarks:
                draw_styled_landmarks(image, results)
                
                # Extract and process keypoints
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]  # Keep last 30 frames
                
                if len(sequence) == 30:
                    # Reshape sequence for prediction
                    input_sequence = np.array(sequence).reshape(1, 30, 63)
                    
                    # Predict
                    res = model.predict(input_sequence)[0]
                    predictions.append(np.argmax(res))
                    
                    # Stabilize predictions
                    if len(np.unique(predictions[-10:])) == 1:
                        if res[np.argmax(res)] > threshold:
                            current_time = time.time()
                            
                            # Add to sentence after a delay
                            if current_time - start_time > 2:
                                current_sign = actions[np.argmax(res)]
                                if current_sign != last_sign:
                                    sentence.append(current_sign)
                                    translation.append(current_sign)
                                    last_sign = current_sign
                                    start_time = current_time
                    
                    # Update translation label
                    translation_display = ' '.join(translation[-5:])
                    translated_label.config(text=translation_display)
                    translated_label.update_idletasks()
                    
                    # Visualize probabilities
                    image = prob_viz(res, actions, image)
            
            # Display current sentence
            cv2.rectangle(image, (0, 0), (image.shape[1], 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence[-5:]), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('OpenCV Feed', image)

            key = cv2.waitKey(10)
            if key & 0xFF == ord('q'):  # Exit when 'q' is pressed
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        translation_window.destroy()
        root.deiconify()

def on_button_click():
    language = language_var.get()
    messagebox.showinfo("Info", f"{language} Sign Language Translation Started.")
    root.withdraw()  # Hide the initial Tkinter UI
    threading.Thread(target=run_openCV, args=(language,), daemon=True).start()

# Main GUI Setup
root = tk.Tk()
root.title("Neuronest Studio")
root.geometry("800x600")
root.configure(bg="#2C2C2B")

language_var = tk.StringVar(root)
language_var.set("English")  # Default language

language_label = tk.Label(root, text="Choose Your Language to Translate", 
                           font=("poppins", 18, "bold"), 
                           fg="white", bg="#2C2C2B")
language_label.pack(pady=20)

language_menu = tk.OptionMenu(root, language_var, "English", "Malayalam", "Tamil", "Hindi", "Telugu")
language_menu.config(font=("poppins", 14), bg="#EFEEE7", fg="#000000")
language_menu.pack(pady=10)

button = tk.Button(root, text="Start Translation", 
                   command=on_button_click, 
                   font=("Arial", 16), 
                   bg="#2980b9", 
                   fg="white", 
                   height=2)
button.pack(pady=20)

root.mainloop()

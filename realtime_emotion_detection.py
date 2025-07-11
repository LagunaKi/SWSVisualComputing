import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class RealtimeEmotionDetector:
    def __init__(self, model_path='emotion_model.pkl'):
        """Initialize the emotion detector with trained model"""
        # Load the trained model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.emotions = model_data['emotions']
        
        # Initialize MediaPipe Face Mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.fps = 0
        
    def extract_landmarks(self, frame):
        """Extract facial landmarks from frame"""
        # Convert to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Process image
        results = self.face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            # Get the first face landmarks
            landmarks = results.multi_face_landmarks[0]
            
            # Extract landmark coordinates
            landmark_coords = []
            for landmark in landmarks.landmark:
                landmark_coords.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmark_coords), landmarks
        else:
            return None, None
    
    def predict_emotion(self, landmarks):
        """Predict emotion from landmarks"""
        # Reshape landmarks for prediction
        landmarks_reshaped = landmarks.reshape(1, -1)
        
        # Scale landmarks
        landmarks_scaled = self.scaler.transform(landmarks_reshaped)
        
        # Predict emotion
        emotion_idx = self.model.predict(landmarks_scaled)[0]
        emotion_prob = self.model.predict_proba(landmarks_scaled)[0]
        
        return self.emotions[emotion_idx], emotion_prob[emotion_idx]
    
    def draw_emotion_info(self, frame, emotion, confidence, landmarks):
        """Draw emotion information on frame"""
        # Draw face mesh
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        )
        
        # Draw emotion text
        emotion_text = f"Emotion: {emotion.upper()}"
        confidence_text = f"Confidence: {confidence:.2f}"
        fps_text = f"FPS: {self.fps:.1f}"
        
        # Set text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        
        # Get text sizes
        (emotion_width, emotion_height), _ = cv2.getTextSize(emotion_text, font, font_scale, thickness)
        (conf_width, conf_height), _ = cv2.getTextSize(confidence_text, font, font_scale, thickness)
        (fps_width, fps_height), _ = cv2.getTextSize(fps_text, font, font_scale, thickness)
        
        # Draw background rectangles
        cv2.rectangle(frame, (10, 10), (10 + emotion_width + 20, 10 + emotion_height + 20), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 40), (10 + conf_width + 20, 40 + conf_height + 20), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 70), (10 + fps_width + 20, 70 + fps_height + 20), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, emotion_text, (20, 30), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(frame, confidence_text, (20, 60), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(frame, fps_text, (20, 90), font, font_scale, (255, 255, 255), thickness)
        
        # Color code based on emotion
        emotion_colors = {
            'happy': (0, 255, 0),    # Green
            'sad': (255, 0, 0),      # Blue
            'angry': (0, 0, 255),    # Red
            'surprise': (0, 255, 255), # Yellow
            'fear': (128, 0, 128),   # Purple
            'disgust': (0, 128, 128), # Teal
            'neutral': (128, 128, 128) # Gray
        }
        
        # Draw emotion indicator
        color = emotion_colors.get(emotion, (255, 255, 255))
        cv2.circle(frame, (frame.shape[1] - 50, 50), 30, color, -1)
        cv2.circle(frame, (frame.shape[1] - 50, 50), 30, (255, 255, 255), 2)
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def run(self):
        """Run real-time emotion detection"""
        print("Starting real-time emotion detection...")
        print("Press 'q' to quit")
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
            
            # Apply mirror effect
            frame = cv2.flip(frame, 1)
            
            # Extract landmarks
            landmarks, face_landmarks = self.extract_landmarks(frame)
            
            if landmarks is not None and face_landmarks is not None:
                # Predict emotion
                emotion, confidence = self.predict_emotion(landmarks)
                
                # Draw emotion information
                self.draw_emotion_info(frame, emotion, confidence, face_landmarks)
            else:
                # No face detected
                cv2.putText(frame, "No face detected", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Update FPS
            self.update_fps()
            
            # Display frame
            cv2.imshow('Real-time Emotion Detection', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        self.face_mesh.close()
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()

def main():
    # Check if model exists
    model_path = 'emotion_model.pkl'
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        print("Please run train_emotion_model.py first to train the model.")
        return
    
    # Initialize and run emotion detector
    detector = RealtimeEmotionDetector(model_path)
    detector.run()

if __name__ == "__main__":
    import os
    main() 
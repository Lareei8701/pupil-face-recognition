import cv2
import numpy as np

class PupilDetector:
    def __init__(self):
        # Load Haar cascade classifiers
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    def detect_pupils_in_frame(self, frame):
        """Detect pupils in a single frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face (optional)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Region of interest for eyes (upper half of face)
            roi_gray = gray[y:y+h//2, x:x+w]
            roi_color = frame[y:y+h//2, x:x+w]
            
            # Detect eyes in the face region
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5, minSize=(30, 30))
            
            for (ex, ey, ew, eh) in eyes:
                # Draw rectangle around eye (optional)
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
                # Extract eye region
                eye_gray = roi_gray[ey:ey+eh, ex:ex+ew]
                
                # Enhance contrast for better pupil detection
                eye_gray = cv2.equalizeHist(eye_gray)
                
                # Apply Gaussian blur
                eye_blurred = cv2.GaussianBlur(eye_gray, (5, 5), 0)
                
                # Apply HoughCircles ONLY within eye region
                circles = cv2.HoughCircles(
                    eye_blurred,
                    cv2.HOUGH_GRADIENT,
                    dp=1,
                    minDist=20,
                    param1=50,
                    param2=15,  # Lower threshold for better detection
                    minRadius=5,
                    maxRadius=min(ew, eh)//4  # Reasonable pupil size
                )
                
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    
                    # Take only the most central circle (likely the pupil)
                    eye_center_x, eye_center_y = ew//2, eh//2
                    best_circle = None
                    min_dist = float('inf')
                    
                    for (cx, cy, r) in circles:
                        dist = np.sqrt((cx - eye_center_x)**2 + (cy - eye_center_y)**2)
                        if dist < min_dist:
                            min_dist = dist
                            best_circle = (cx, cy, r)
                    
                    if best_circle:
                        cx, cy, r = best_circle
                        # Convert coordinates back to original frame
                        center_x = cx + ex + x
                        center_y = cy + ey + y
                        
                        # Draw pupil
                        cv2.circle(frame, (center_x, center_y), r, (0, 0, 255), 2)
                        cv2.circle(frame, (center_x, center_y), 2, (0, 255, 255), 2)
        
        return frame

def process_video_file():
    detector = PupilDetector()
    # Open the video file
    cap = cv2.VideoCapture('faces.mp4')
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        print("Make sure the video file path is correct")
        return
    
    print("Processing video... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or failed to read frame")
            break
        
        # Detect pupils in the frame
        frame = detector.detect_pupils_in_frame(frame)
        
        # Display the frame
        cv2.imshow('Video Pupil Detection', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release everything
    cap.release()
    cv2.destroyAllWindows()
    print("Video processing complete")

if __name__ == "__main__":
    process_video_file()
COMP 4630 - Final Project

# ✋ American Sign Language Recognition using YOLO-Pose & OpenCV

### 5. Final Pipeline: YOLO + CNN Hybrid System
The final system combines YOLO-based detection with CNN-based classification:

1. YOLO detects the hand and outputs a bounding box  
2. The detected region is cropped from the input frame  
3. The cropped image is passed into the CNN classifier  
4. The CNN predicts the corresponding ASL letter  

This hybrid approach separates:
- **Detection (YOLO):** locating the hand in real time  
- **Classification (CNN):** recognizing the gesture  

Please see the report for more in depth details

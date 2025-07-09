## pupil-face-recognition
# Pupil Detection in Video Files
A Python script that detects and tracks pupils in video files using OpenCV.
# Requirements
- Python 3.x
- OpenCV library
# Installation
1. Install OpenCV:
```bash
pip install opencv-python
```
# Setup
1. Download or clone the pupil detection script
2. Place your video file in the same directory as the script
3. Open the script and replace the video name with your actual video filename
# What You'll See
- Blue rectangles: Detected faces
- Green rectangles: Detected eyes
- Red circles: Detected pupils
- Yellow dots: Pupil centers
# Troubleshooting
**If no pupils are detected:**
- Make sure faces are clearly visible in the video
- Check that your video file path is correct
**If too many false circles appear:**
- Increase the `param2` value
- Adjust the `minRadius` and `maxRadius` values
# File Structure
project_folder/
│
├── pupil_detection.py
├── your_video.mp4
├── haarcascade_frontalface_default.xml
└── haarcascade_eye.xml
Note: The XML files are automatically included with OpenCV installation and accessed via cv2.data.haarcascades. You don't need to download them separately.
# Notes
- Works best with clear, well-lit videos
- Faces should be facing the camera
- Video resolution affects detection accuracy

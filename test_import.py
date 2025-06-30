#!/usr/bin/env python3
import sys
print("Python version:", sys.version)
print("Python path:", sys.path)

try:
    import face_recognition_models
    print("face_recognition_models imported successfully")
    print("Models path:", face_recognition_models.__file__)
except ImportError as e:
    print("Failed to import face_recognition_models:", e)

try:
    import face_recognition
    print("face_recognition imported successfully")
    
    # Try to use a basic function
    import numpy as np
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    encodings = face_recognition.face_encodings(test_image)
    print("face_recognition functions work, encodings found:", len(encodings))
    
except Exception as e:
    print("Error with face_recognition:", e)
    import traceback
    traceback.print_exc()

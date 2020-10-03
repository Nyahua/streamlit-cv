"""
Check which inputs are a valid camera.
https://github.com/robmarkcole/mqtt-camera-streamer/blob/master/scripts/check-opencv-cameras.py
"""
import streamlit as st

import cv2

def check_webcam():
    webcam_dict = dict()
    for i in range(0, 10):
        cap = cv2.VideoCapture(i)
        is_camera = cap.isOpened()
        if is_camera:
            webcam_dict[f"index[{i}]"] = "VALID"
            cap.release()
        else:
            webcam_dict[f"index[{i}]"] = None
    return webcam_dict

if __name__ == "__main__":
    st.title('WebCam index validation check')
    webcam_dict = check_webcam()
    st.write(webcam_dict)
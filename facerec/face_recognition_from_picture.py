import cv2
import streamlit as st
import os
import numpy as np
import pandas as pd
import face_recognition
import cv2

# CONSTANTS
PATH_DATA = 'data/DB.csv'
COLOR_DARK = (0, 0, 153)
COLOR_WHITE = (255, 255, 255)
COLS_INFO = ['name', 'description']
COLS_ENCODE = [f'v{i}' for i in range(128)]


def init_data(data_path=PATH_DATA):
    if os.path.isfile(data_path):
        return pd.read_csv(data_path)
    else:
        return pd.DataFrame(columns=COLS_INFO + COLS_ENCODE)

# convert image from opened file to np.array


def byte_to_array(image_in_byte):
    return cv2.imdecode(
        np.frombuffer(image_in_byte.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

# convert opencv BRG to regular RGB mode


def BGR_to_RGB(image_in_array):
    return cv2.cvtColor(image_in_array, cv2.COLOR_BGR2RGB)

# convert face distance to similirity likelyhood


def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * np.power((linear_val - 0.5) * 2, 0.2))


if __name__ == "__main__":
    # disable warning signs:
    # https://discuss.streamlit.io/t/version-0-64-0-deprecation-warning-for-st-file-uploader-decoding/4465
    st.set_option("deprecation.showfileUploaderEncoding", False)

    # title area
    st.markdown("""
    # Face Recognition APP
    > Powered by [*ageitgey* face_recognition](https://github.com/ageitgey/face_recognition/) python engine
    """)

    # displays a file uploader widget and return to BytesIO
    image_byte = st.file_uploader(
        label="Select a picture contains faces:", type=['jpg', 'png']
    )
    # detect faces in the loaded image
    max_faces = 0
    rois = []  # region of interests (arrays of face areas)
    if image_byte is not None:
        image_array = byte_to_array(image_byte)
        face_locations = face_recognition.face_locations(image_array)
        for idx, (top, right, bottom, left) in enumerate(face_locations):
            # save face region of interest to list
            rois.append(image_array[top:bottom, left:right].copy())

            # Draw a box around the face and lable it
            cv2.rectangle(image_array, (left, top),
                          (right, bottom), COLOR_DARK, 2)
            cv2.rectangle(
                image_array, (left, bottom + 35),
                (right, bottom), COLOR_DARK, cv2.FILLED
            )
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                image_array, f"#{idx}", (left + 5, bottom + 25),
                font, .55, COLOR_WHITE, 1
            )

        st.image(BGR_to_RGB(image_array), width=720)
        max_faces = len(face_locations)

    if max_faces > 0:
        # select interested face in picture
        face_idx = st.selectbox("Select face#", range(max_faces))
        roi = rois[face_idx]
        st.image(BGR_to_RGB(roi), width=min(roi.shape[0], 300))

        # initial database for known faces
        DB = init_data()
        face_encodings = DB[COLS_ENCODE].values
        dataframe = DB[COLS_INFO]

        # compare roi to known faces, show distances and similarities
        face_to_compare = face_recognition.face_encodings(roi)[0]
        dataframe['distance'] = face_recognition.face_distance(
            face_encodings, face_to_compare
        )
        dataframe['similarity'] = dataframe.distance.apply(
            lambda distance: f"{face_distance_to_conf(distance):0.2%}"
        )
        st.dataframe(
            dataframe.sort_values("distance").iloc[:5]
            .set_index('name')
        )

        # add roi to known database
        if st.checkbox('add it to knonwn faces'):
            face_name = st.text_input('Name:', '')
            face_des = st.text_input('Desciption:', '')
            if st.button('add'):
                encoding = face_to_compare.tolist()
                DB.loc[len(DB)] = [face_name, face_des] + encoding
                DB.to_csv(PATH_DATA, index=False)
    else:
        st.write('No human face detected.')
# Project for Assessment
# Evaluation of face_recognition package via FERET dataset
# ========================================================
"""Help functions."""

import face_recognition as fr

def load_files():
    return

def get_encodings():
    return

def get_scores():
    img_1 = fr.load_image_file("assets/FeretMedium/00001_930831_fa_a.jpg")
    enc_1 = fr.face_encodings(img_1)[0]
    img_2 = fr.load_image_file("assets/FeretMedium/00001_930831_fb_a.jpg")
    enc_2 = fr.face_encodings(img_2)[0]
    result = fr.compare_faces([enc_1], enc_2)
    dist = fr.face_distance([enc_1], enc_2)

    print(result)
    print(dist)

    return

def output_results():
    return


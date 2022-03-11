# Project for Assessment
# Evaluation of face_recognition package via FERET dataset
# ========================================================
"""Help functions."""

import face_recognition as fr
import os

def get_encodings(folder):
    path = os.path.join(os.getcwd(), folder)

    face_encodings = []
    face_names = []
    with os.scandir(path) as files:
        for entry in files:
            if entry.is_file() and entry.name.endswith(".jpg"):
                img_array = fr.load_image_file(entry.path)
                if len(fr.face_encodings(img_array)) < 1:
                    print(f"Unidentified: ", entry.name)
                else:
                    face_encodings.append(fr.face_encodings(img_array)[0])
                    face_names.append(entry.name)     
    
    return face_names, face_encodings

def get_scores(names, encodings):
    probes, subjects, scores, matches = ([] for i in range(4))

    for i in range(len(names)):
        probes.extend([names[i]]*(len(names) - i))
        subjects.extend(names[i:])
        scores.extend(1 - fr.face_distance(encodings[i:], encodings[i]))
        matches.extend(fr.compare_faces(encodings[i:], encodings[i]))
    
    result = [probes, subjects, scores, matches]

    return result

def get_output(result, dir: str=""):
    dir = f"{dir}result.csv"
    try:
        with open(dir, 'w') as f:
            f.write(f"probe, subject, score, match\n")
            for i in range(len(result[0])):
                f.write(f"{result[0][i]}, {result[1][i]}, {result[2][i]}, {result[3][i]}\n")
            print(f"Output saved at /{dir}")
    except Exception as e:
        print(e.args[1])
    return


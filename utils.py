# Project for Assessment
# Evaluation of face_recognition package via FERET dataset
# ========================================================
"""Help functions."""

import face_recognition as fr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_encodings(folder):
    path = os.path.join(os.getcwd(), folder)

    face_encodings = []
    face_names = []
    with os.scandir(path) as files:
        count = 0
        for entry in files:
            if entry.is_file() and entry.name.endswith(".jpg"):
                # count += 1
                # if count > 10:
                #     break
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
        # matches.extend(fr.compare_faces(encodings[i:], encodings[i]))
    
    for i in range(len(probes)):
        if probes[i][:5] == subjects[i][:5]:
            matches.append(True)
        else:
            matches.append(False)
    
    result = [probes, subjects, scores, matches]

    return result

def get_output(result, dir: str=""):
    dir = f"{dir}result.csv"
    output = pd.DataFrame({
        "probe": result[0],
        "subject": result[1],
        "score": result[2],
        "match": result[3]
    })
    output.to_csv(dir)

    return output

def get_distribution(df):
    sns.displot(
        df,
        x="score", bins=100, kde=True, hue="match",
        # log_scale=(False, True)
    )
    # plt.show()
    return

def get_equal_error_rate(df):
    temp = 1
    for th in range(1,99):
        th /= 100
        false_match = sum(df[df["score"] > th]["match"] == False)
        match = sum(df["score"] > th)
        false_non_match = sum(df[df["score"] <= th]["match"] == True)
        non_match = sum(df["score"] <= th)

        if match == 0 or non_match == 0:
            pass
        
        if abs(false_match/match - false_non_match/non_match) < temp:
            temp = abs(false_match/match - false_non_match/non_match)
            result = th

    print(result)
    return result
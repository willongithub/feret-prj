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
    """Calculate face encoding for all images in the folder."""
    
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
    """Calculate distance between faces as matching score."""

    probes, subjects, scores, matches = ([] for _ in range(4))

    for i in range(len(names) - 1):
        probes.extend([names[i]]*(len(names) - 1 - i))
        subjects.extend(names[i + 1:])
        scores.extend(
            (1 - fr.face_distance(encodings[i + 1:], encodings[i]))*100
        )
    
    for i in range(len(probes)):
        if probes[i][:5] == subjects[i][:5]:
            matches.append(True)
        else:
            matches.append(False)
    
    result = [probes, subjects, scores, matches]

    return result

def get_output(result, dir: str=""):
    """Output results to CSV file."""

    dir = f"{dir}result.csv"
    output = pd.DataFrame({
        "probe": result[0],
        "subject": result[1],
        "score": result[2],
        "match": result[3]
    })
    output = output[output["score"] != 100.]
    output.drop_duplicates(subset=["probe", "score"], keep='last')
    output.to_csv(dir)

    print(f"Output save at: {dir}")

    return output

def get_distribution(df):
    """Simple analysis on the results."""

    gen = df[df["match"] == True]
    imp = df[df["match"] == False]

    g = sns.displot(
        gen,
        x="score", bins=30, kde=True,
    ).set(title="Genuine")
    g.refline(x = gen.score.mean())
    # plt.show()

    g = sns.displot(
        imp,
        x="score", bins=30, kde=True,
    ).set(title="Impostor")
    g.refline(x = imp.score.mean())
    plt.show()

    return

def get_equal_error_rate(df):
    """Find equal error rate."""

    temp = 1
    for th in range(1, 99):
        false_match = sum(df[df["score"] <= th]["match"] == True)
        match = sum(df["match"] == True)
        false_non_match = sum(df[df["score"] > th]["match"] == False)
        non_match = sum(df["match"] == False)

        if false_match == 0 or false_non_match == 0:
            continue

        if abs(false_match/match - false_non_match/non_match) < temp:
            temp = abs(false_match/match - false_non_match/non_match)
            score = th
            eer = false_match/match
    
    g = sns.displot(
        df,
        x="score", bins=100, hue="match",
        # kde=True,
        stat="percent",
        common_norm=False,
    ).set(title="Threshold Score for EER")
    g.refline(x = score)
    plt.show()

    return score, eer

# Prompt info for CLI interface.
PROMPT_INIT = """
> Run face match algorithm.     [1]
> Plot score distribution.      [2]
> Find threshold score for EER. [3]
> Run all.                      [0]
> Exit.                         [e]
:"""

PROMPT_INPUT = """
> Enter input folder path (press Enter for default):\t
"""

PROMPT_OUTPUT = """
> Enter output CSV file path (press Enter for default):\t
"""
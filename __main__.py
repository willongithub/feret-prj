# Project for Assessment
# Evaluation of face_recognition package via FERET dataset
# ========================================================
"""Entry point."""

import utils
import time

def main():
    start_time = time.perf_counter()

    filenames, encodings = utils.get_encodings("assets/FeretMedium/")
    result = utils.get_scores(filenames, encodings)
    utils.get_output(result)

    end_time = time.perf_counter()

    print(f"Processing Time: {end_time - start_time:0.6f}" )

if __name__ == '__main__':
    main()
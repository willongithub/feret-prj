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
    output = utils.get_output(result)
    # utils.get_distribution(output)
    rate = utils.get_equal_error_rate(output)

    end_time = time.perf_counter()

    print(f"Processing Time: {end_time - start_time:0.6f}" )

if __name__ == '__main__':
    main()
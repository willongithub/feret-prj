# Project for Assessment
# Evaluation of face_recognition package via FERET dataset
# ========================================================
"""Entry point."""

import utils
import time

def main():
    """CLI interface entry."""

    processed = False

    while True:
        flag = input(utils.PROMPT_INIT)

        if flag in ('1'): processed = False
        
        if processed == False:
            path = input(utils.PROMPT_INPUT)
            if path == '': path = "assets/FeretMedium/"
            start_time = time.perf_counter()
            names, encodings = utils.get_encodings(path)
            result = utils.get_scores(names, encodings)
            end_time = time.perf_counter()
            print(f"Processing Time: {end_time - start_time:0.6f}" )
        
            path = input(utils.PROMPT_OUTPUT)
            output = utils.get_output(result, path)

            processed = True
        
        if flag in ('0', '2'):
            utils.get_distribution(output)
        
        if flag in ('0', '3'):
            value = utils.get_equal_error_rate(output)
            print(f"Equal error rate: {value}")
        
        if flag == 'e': break
    
    print("\nEnd of Process")

if __name__ == '__main__':
    main()
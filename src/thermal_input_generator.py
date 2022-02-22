from helpers import extract_first_number
import config
import numpy as np
import os


def generate_thermal_input():
    '''
    Generates the thermal data for the classifier
    @Returns
        all_data: list of floats of size (n_samples, n_features)
            a list of all of the thermal data of the data samples
        all_labels: list of int of size (n_samples, 1)
            a list of all labels of the participants
    '''
    all_datafiles = [participant + '-' + str(i) for participant in config.PARTICIPANTS for i in range(config.LOWER_FILE_INDEX, config.UPPER_FILE_INDEX + 1)]
    all_data = []
    all_labels = []
    for file in all_datafiles:
        data = np.load(os.path.join(config.THERMAL_PATH, file + '.npy'), allow_pickle=True)
        all_data.append(data)
        all_labels.append(extract_first_number(file))
    return all_data, all_labels

if __name__ == "__main__":
    generate_thermal_input()
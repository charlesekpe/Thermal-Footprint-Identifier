import numpy as np
import config
import cv2
import os
from helpers import extract_first_number
        
def split_in_testing_and_training_data(train_index, test_index, random_state=None):
    '''
    splits the datafile names and labels into testing and training data
    @Parameters
        test_size: float in range(0,1)
            The proportion of the test data to the total data
        random_state: int
            Random seed passed to sklearn function
    @Returns
        X_train: list of string
            the datafile names of the training data
        X_test: list of string
            the datafule names of the testing data
    '''
    all_datafiles = np.array([participant + '-' + str(i) for participant in config.PARTICIPANTS for i in range(config.LOWER_FILE_INDEX, config.UPPER_FILE_INDEX+1)])
    train_files = all_datafiles[train_index]
    test_files = all_datafiles[test_index]
    return train_files, test_files

def generate_descriptors_labels(X_train, X_test):
    '''
    generates the descriptors and labels of the testing and training data
    @Parameters
        X_train: list of string
            the names of the datafiles in the training data
        X_test: list of string
            the names of the datafiles in the testing data
    @Returns
        data_label_dict: dict of array of shape (2,1)
            The dict that contains the descriptors and label to every datafile name
        train_data: array of shape (n_train_samples, n_descriptors)
            An array that contains all descriptors of the training data
    '''
    train_data = []
    data_label_dict = dict()
    for file in np.concatenate((X_train, X_test)):
        label = extract_first_number(file)
        # load the descriptors from the descriptor files
        descriptors = np.load(os.path.join(config.DES_PATH, file + '.npy'), allow_pickle=True)
        data_label_dict[file] = [descriptors, label]
        for des in descriptors:
            if file in X_train:
                if des is None:
                    continue
                train_data.append(des)
    #print("$$$$$$$$$$$$$$$$$$$$$")
    #del train_data[28]
    train_data= [np.float32(data) for data in train_data]
    
    #print("train data--->",np.shape(train_data[0]))
    #print("train data--->",np.shape(train_data[1]))
    #print("train data 28--->",np.shape(train_data[28]))
    #print("train data 100--->",np.shape(train_data[100]))
    #print("train data--->",(train_data[28]))          
    train_data = np.vstack(train_data)
    print(np.shape(train_data))
    return data_label_dict, train_data

def k_mean_clustering(features, k=config.K_MEANS_K):
    '''
    k mean clustering algorithm for finding centres within the features
    @Parameters
        features: 2-dim array
            The features that the centre should be found of
        k: int
            Number of centres to find
    @Rreturns
        centres: array of same shape as features
            The common centres that were found
    '''
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, _, centres = cv2.kmeans(features, k, None, criteria, 10, flags)
    return centres


def bag_of_features(features, centres, k=config.K_MEANS_K):
    '''
    similar algorithm to bag of words in NLU.
    Counts the occurrence of common centres in the features
    @Parameter
        features: 2-dim array
            the feature vector that is passed
        centres: 2-dim array
            the centres that are compared to the features
        k: int
            amount of points that should be compared
    @Returns
        vec: 2-dim array that has the count of feature occurrence in it
    
    Code derived from:
        Title: Image Classification using SIFT
        Author: Akhilesh Sharma
        Date: 27.11.2020
        Availability: https://github.com/Akhilesh64/Image-Classification-using-SIFT/blob/main/main.py
    '''
    # initialize with array of size k with all values being 0
    vec = np.zeros((1, k))
    # vertically stack the features
    #print("Feature $$$$$$$$$ --->",features)
    #features[features == None] = 0
    #features= [np.float32(data) for data in features]
    features = np.vstack(features)
    # for all feature points
    for i in range(features.shape[0]):
        feat = features[i]
        # calculate the difference of the tiled features to the centres
        # tile repeats the feat (k,1) times
        diff = np.tile(feat, (k, 1)) - centres
        # calculate the distances
        dist = pow(((pow(diff, 2)).sum(axis = 1)), 0.5)
        # get sorted index array of the distance array
        idx_dist = dist.argsort()
        # take the index of the smallest distance
        idx = idx_dist[0]
        # increase the vec at this position by 1
        vec[0][idx] += 1
    return vec
        
def generate_cv_input(train_index, test_index, random_state=None):
    '''
    Main function of this module
    Generates the computer vision inputs from the feature files
    @Parameters
        random_state: int
            Random seed passed to split the data
        test_size: float in range (0,1)
            Proportion of the testing size to total size
    @Returns
        train_vec: array of size (n_train_samples, n_cv_features)
            the training input data
        test_vec: array of size (n_test_samples, n_cv_features)
            the testing input data
        train_labels: array of int of size (n_train_samples, 1)
            the training labels
        test_labels: array of int of size (n_test_samples, 1)
            the testing labels
        X_train: array of string of size (n_train_samples, 1)
            names of the training data files ordered
        X_test: array of string of size (n_test_samples, 1)
            names of the testing data files ordered
    '''
    # generates testing and training data names
    X_train, X_test = split_in_testing_and_training_data(train_index, test_index, random_state=random_state)
    # generates [des, label] - dict and training data array
    print("X_TRAIN: ", X_train)
    print("X_TEST: ", X_test)
    dictionary, train_data = generate_descriptors_labels(X_train, X_test)
    # finds centres by using k_mean_clustering among the training data
    #print("train data ------->",train_data)
    #print("train data max------->",(np.max(train_data)))
    #print("train data min------->",(np.min(train_data)))
    #print("train data shape------->",(np.shape(train_data)))
    centres = k_mean_clustering(train_data)
    test_labels = []
    test_vec = []
    train_labels = []
    train_vec = []
    # for all testing files
    for test_file in X_test:
        # for all descriptors from the testing file (one descriptor per frame)
        for des in dictionary[test_file][0]:
            # applies bag of features to current descriptors, comparing with centres
            img_vec = bag_of_features(des, centres)
            # append to testing vector
            test_vec.append(img_vec)
            # append to label list
            test_labels.append(dictionary[test_file][1])
    # do the same for all training data
    for train_file in X_train:
        for des in dictionary[train_file][0]:
            if des is None:
                continue
            img_vec = bag_of_features(des, centres)
            train_vec.append(img_vec)
            train_labels.append(dictionary[train_file][1])
    # vertically stack the vectors
    test_vec = np.vstack(test_vec)
    train_vec = np.vstack(train_vec)
    return train_vec, test_vec, train_labels, test_labels, X_train, X_test


# if executed, do a test run of generate_cv_input
if __name__ == '__main__':
    train_vec, test_vec, _, _, _, _ = generate_cv_input([1,2,3,4,5], [6,7,8])

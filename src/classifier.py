import numpy as np
import config
import time
import sys
import argparse
import pandas as pd
import shap
import warnings
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
from cv_input_generator import generate_cv_input
from helpers import extract_first_number
from thermal_input_generator import generate_thermal_input
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score

# number of iterations
LBFGS_ITERATIONS = 1000
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def main_cv():
    '''
    Classifies the computer vision features only
    '''
    all_datafiles = [participant + '-' + str(i) for participant in config.PARTICIPANTS for i in range(config.LOWER_FILE_INDEX, config.UPPER_FILE_INDEX + 1)]
    all_labels = [extract_first_number(i) for i in all_datafiles]
    stratified_cross_validation(np.zeros(len(all_labels)), all_labels, feature_names=config.OPENCV_FEATURES)
    
def main_thermal():
    '''
    Classifies the thermal features only
    '''
    # calls generate thermal input function
    data, labels = generate_thermal_input()
    # cross validates data and labels
    stratified_cross_validation(data, labels, feature_names=config.THERMAL_FEATURES)
   
def main_combined():
    '''
    Classifies the thermal and the computer vision features
    @Parameters
        random_state: int
            the random state to split the data into testing and training data
    '''
    all_datafiles = [participant + '-' + str(i) for participant in config.PARTICIPANTS for i in range(config.LOWER_FILE_INDEX, config.UPPER_FILE_INDEX + 1)]
    all_labels = [extract_first_number(i) for i in all_datafiles]
    stratified_cross_validation(np.zeros(len(all_labels)), all_labels, feature_names=config.ALL_FEATURES)
    
def stratified_cross_validation(data, labels, feature_names, k=config.CROSS_VALIDATION_K):
    '''
    Stratified k fold cross validation
    @Parameters
        data: array of float of shape (n_samples, n_features)
            the input data to be classified
        labels: array of int of shape (n_samples, )
            the classes of the data
        k: int
            the number of folds
    '''
    # take the start time
    start = time.time()
    X = np.array(data)
    y = np.array(labels)
    # define stratified k fold
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    # initialize measure lists
    accs, precs, f1s, recs, aucs, mats = [], [], [], [], [], []
    list_shap_values = list()
    list_indexes = list()
    # for every fold in the cross validation
    for train_index, test_index in skf.split(X, y):
        # if opencv classification
        if feature_names == config.OPENCV_FEATURES:
            x_train, x_test, y_train, y_test, _, _ = generate_cv_input(train_index, test_index)
            X = np.concatenate((x_train, x_test))
        # if thermal classification
        elif feature_names == config.THERMAL_FEATURES:
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        # if combined classification
        else:
            x_train, x_test, y_train, y_test = generate_combined_features(train_index, test_index)
            X = np.concatenate((x_train, x_test))
        # define logistic regression classifier
        clf = LogisticRegression(multi_class='ovr', max_iter=LBFGS_ITERATIONS, solver='lbfgs')
        # train the classifier
        clf.fit(x_train, y_train)
        # predict labels of x_test
        y_pred = clf.predict(x_test)
        # predict probabilities of labels of x_test (used for AUC)
        y_pred_proba = clf.predict_proba(x_test)
        # calculate measures and add them to lists
        aucs.append(roc_auc_score(y_test, y_pred_proba, multi_class='ovr'))
        accs.append(accuracy_score(y_test, y_pred))
        precs.append(precision_score(y_test, y_pred, labels=np.unique(y_test), average='weighted', zero_division=0))
        f1s.append(f1_score(y_test, y_pred, labels=np.unique(y_test), average='weighted'))
        recs.append(recall_score(y_test, y_pred, labels=np.unique(y_test), average='weighted'))
        mats.append(confusion_matrix(y_test, y_pred, labels=np.unique(y_test), normalize='true'))
        # get shap explainer for this k fold iteration
        explainer = shap.Explainer(clf, x_train) # get shap explainer for this k fold iteration
        # compute shap values on train data
        shap_values = explainer.shap_values(x_train)
        # append the shap values to list
        list_shap_values.append(shap_values)
        list_indexes.append(train_index)
    end = time.time()
    measures = dict()
    measures['mean acc'] = np.mean(accs)
    measures['std acc'] = np.std(accs)
    measures['mean prec '] = np.mean(precs)
    measures['std prec'] = np.std(precs)
    measures['mean f1 '] = np.mean(f1s)
    measures['std f1'] = np.std(f1s)
    measures['mean recall'] = np.mean(recs)
    measures['std recall'] = np.std(recs)
    measures['mean auc'] = np.mean(aucs)
    measures['std auc'] = np.std(aucs)
    measures['computing time'] = end - start
    avg_mat = np.mean(mats, axis=0)
    my_plot_confusion_matrix(avg_mat, np.mean(accs), show=False)
    save_cross_val_feature_importance_graph(list_shap_values, list_indexes, X, feature_names=feature_names)
    print_classifier_report(measures, 'ovr stratified k fold')

def print_classifier_report(measures, classifier_name):
    '''
    Print the results of a classifier
    @Parameters
        measures: dictionary
            all measures with their values that should be reported
        classifier_name: string
            the name of the classifier
    '''
    print(' ########## CLASSIFIER RESULTS ##########')
    print('CLASSIFIER: ', classifier_name.upper())
    print('FEATURE SET: ', config.FEATURE_SET)
    print('LOWER/UPPER FILE INDEX: {}/{}'.format(str(config.LOWER_FILE_INDEX), str(config.UPPER_FILE_INDEX)))
    for key, values in measures.items():
        print('{} : {}'.format(key, str(values)))

def save_cross_val_feature_importance_graph(shap_value_list, index_list, X, feature_names):
    '''
    saves and generates a feature importance graph from multiple SHAP values computed in k fold cross validation
    @Parameters
        shap_value_list: array of shape (k_folds, 20, n_samples, n_features)
            the generated shap values of all samples over all features and all folds
        index_list: array of shape (k_folds, n_samples)
            the list of all indexes that were used for shap feature importance graph
    
    Code derived from:
        Title: Visualizing variable importance using SHAP and cross-validation
        Author: Lucas Ramos
        Date: 26.06.2020
        Availability: https://lucasramos-34338.medium.com/visualizing-variable-importance-using-shap-and-cross-validation-bd5075e9063a
    '''
    #combining results from all iterations
    test_set = index_list[0]
    shap_values = np.array(shap_value_list[0])
    for i in range(1,len(index_list)):
        test_set = np.concatenate((test_set, index_list[i]),axis=0)
        shap_values = np.concatenate((shap_values, np.array(shap_value_list[i])),axis=1)
    pd_frame = pd.DataFrame(X[test_set], columns=feature_names)
    plt.figure()
    shap.summary_plot(shap_values[1], pd_frame, show=False, plot_type='bar', max_display=10)
    plt.savefig(os.path.join(config.CLASSIFIER_PLOT_PATH, 'cross_val_shap_train.png'))
       
def my_plot_confusion_matrix(matrix, accuracy, show=True):
    '''
    Plots a confusion matrix to the command line
    @Parameters
        matrix: Matrix
            The confusion matrix obtained from sklearn
        accuracy: float
            The accuracy score from the classifier
        classifier_name: str
            The name of the classifier used (rbf, ovr, poly)
    @Returns
        None
    '''
    plt.figure(figsize=(9,9))
    sns.heatmap(matrix, annot=True, fmt=".2f", linewidths=.5, square = True, cmap = 'Blues_r', xticklabels=list(range(1,len(config.PARTICIPANTS) + 1)), yticklabels=list(range(1,len(config.PARTICIPANTS) + 1)), annot_kws={"fontsize": 8})
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {}'.format(np.round(accuracy,4))
    plt.title(all_sample_title, size = 15)
    plt.savefig(os.path.join(config.CLASSIFIER_PLOT_PATH, 'confusion_matrix.png'))
    if show:
        plt.show()

def generate_combined_features(train_index, test_index):
    '''
    generates the combined opencv and thermal feature vectors split in testing and training data
    @Parameters
        train_index: ndarray of int
            The training indices in the data
        test_index: ndarray of int
            The testing indices in the data
    @Returns
        X_train_all: training data
        X_test_all: testing data
        y_train: training data labels
        y_test: testing data labels
    '''
    X_train, X_test, y_train, y_test, train_names, test_names = generate_cv_input(train_index, test_index)
    X_train_all = []
    X_test_all = []
    
    # add thermal features to train data
    for index, name in enumerate(train_names):
        # load thermal data from feature files
        data = np.load(os.path.join(config.THERMAL_PATH, name + '.npy'), allow_pickle=True)
        # for alle frames (1 frame = 1 sample)
        for i in range(0,config.FRAMES_PER_PROCESS):
            # make sure that the label is set correctly, else throw Exception
            if y_train[index*config.FRAMES_PER_PROCESS+i] != extract_first_number(name):
                raise Exception("Labels didnt match for ", name, ": y was ", y_train[index*config.FRAMES_PER_PROCESS+i])
            # append cv features of current frame and thermal features to array
            X_train_all.append(np.hstack((X_train[index * config.FRAMES_PER_PROCESS + i], np.array(data))))
            
    # do the same for the testing data
    for index, name in enumerate(test_names):
        data = np.load(os.path.join(config.THERMAL_PATH, name + '.npy'), allow_pickle=True)
        for i in range(0,config.FRAMES_PER_PROCESS):
            if y_test[index*config.FRAMES_PER_PROCESS+i] != extract_first_number(name):
                raise Exception("Labels didnt match for ", name, ": y was ", y_test[index*config.FRAMES_PER_PROCESS+i])
            X_test_all.append(np.hstack((X_test[index*config.FRAMES_PER_PROCESS + i], np.array(data))))
    # vertically stack testing and training data
    X_train_all = np.vstack(X_train_all)
    X_test_all = np.vstack(X_test_all)
    return X_train_all, X_test_all, y_train, y_test


if __name__ == "__main__":
    # argument parser to execute all combinations from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lower', help="lower file index", type=int, default=config.LOWER_FILE_INDEX)
    parser.add_argument('-u', '--upper', help="upper file index", type=int, default=config.UPPER_FILE_INDEX)
    args = parser.parse_args(sys.argv[2:])
    # save lower and upper file index to config
    config.LOWER_FILE_INDEX = args.lower
    config.UPPER_FILE_INDEX = args.upper
    
    # choose classifier, either thermal, computer vision, or combined
    if len(sys.argv) > 1:
        if re.search(r'thermal', sys.argv[1], re.IGNORECASE):
            config.FEATURE_SET = 'THERMAL'
            main_thermal()
        elif re.search(r'cv|vision', sys.argv[1], re.IGNORECASE):
            config.FEATURE_SET = 'OPENCV'
            main_cv()
        elif re.search(r'both|all|combined', sys.argv[1], re.IGNORECASE):
            main_combined()
        else:
            raise NameError('Your argument is not valid. It must be either "thermal", "opencv" or "both"')
    else:
        main_combined()
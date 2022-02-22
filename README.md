# Thermal-Footprint-Identifier-
Using computer vision to identify thermal footprints
#Temporary
Still working in collaboration with other engineers
# Structure of the repo:
data >
    > descriptors: contains generated descriptor files in npy format
    > thermal: contains generated thermal feature files in npy format
    contains .dat files, with thermal information for every acquisition
    does NOT contain the .ravi files, those are not in the repository, because of file size issues
deprecated >
    contains old unused modules, such as the neural network
env >
    automatically generated files from the Anaconda environment
plot_results >
    > classifier: confusion matrix from the classifier and feature importance graph
    > foot images: images from the opencv preprocessing pipeline
    > temperature graphs: plotted temperature graphs, derived from the .dat files
src > (the source code)
    > classifier.py:
        - The file where the classification process is computed
        - To classify the data, this file needs to be executed (see "Run the classifier" above)
    > config.py: 
        - configuration variables
    > cv_input_generator.py: 
        - loads the descriptors and generates the features using the bag-of-features approach
    > datafile.py:
        - file to load and decode the .dat files
    > helpers.py
        - helper function, used across multiple modules
    > image_preprocessor.py:
        - file for descriptor generation and saving those descriptors to files
        - running this file generates the descriptor files (see "generate feature files" above)
    > temperature_preprocessing.py:
        - file for thermal feature generation and saving those features to .npy files
        - running this file generates the thermal feature files (see "generate feature files" above)
    > thermal_input_generator.py:
        - loads thermal features from .npy files and provides them for the classifier
utils:
    > filemover.py:
        - moves captured data into the repository
        - can be ignored, because it only works with local paths

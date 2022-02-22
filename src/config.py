## GENERAL CONFIG
LOWER_FILE_INDEX = 1
UPPER_FILE_INDEX = 10
FEATURE_SET = 'BOTH'
PARTICIPANTS = ['munich1', 'munich2', 'munich3', 'munich4']
CONTROL_PARTICIPANTS = ['munich1', 'munich3', 'munich4', 'munich5']
CONTROL_DATA = [participant + '-' + str(i) for participant in CONTROL_PARTICIPANTS for i in range(LOWER_FILE_INDEX + 10, UPPER_FILE_INDEX + 11)]
AREAS = ['rightFootHeel', 'rightFootMid', 'rightFootToes', 'leftFootHeel', 'leftFootMid', 'leftFootToes']
AREAS_SHORT = ['rHeel', 'rMid', 'rToes', 'lHeel', 'lMid', 'lToes']

# THERMAL CONFIG
STEP_DEGREES = 1.5
FRAMES_TO_CHECK = 7
MIN_THRESHOLD = 20 # degrees Celcius
MAX_THRESHOLD = 35 # degrees Celcius
IGNORE_FIRST_FRAMES = 10
RESCALE_AXIS = -1

# CV CONFIG
K_MEANS_K = 400
SECONDS_AFTER_STEP_IN = 0.5
STEP_OUT_AFTER_MAX = 3
INCLUDE_TRACE_FRAMES = False
CAMERA_FPS = 27
FRAME_INTERVAL = int(CAMERA_FPS/9)
FRAMES_PER_PROCESS = 4

### PATHS
DES_PATH = './data/descriptors/'
SURF_PATH = './data/descriptors_surf'
THERMAL_PATH = './data/thermal/'
PLOT_GRAPH_PATH = './plot_results/temperature_graphs'
FOOT_IMG_PATH = './plot_results/foot_images'
CLASSIFIER_PLOT_PATH = './plot_results/classifier'

### CONFIG FOR CLASSIFIER
CROSS_VALIDATION_K = 5

### FEATURE NAMES FOR SHAP
THERMAL_FEATURES_PER_REGION=['avg', 'std', 'dis', 'max', 'tdi', 'dti', 'mpf']
THERMAL_FEATURES = [i + '-' + j for i in AREAS_SHORT for j in THERMAL_FEATURES_PER_REGION]
OPENCV_FEATURES = ['center' + str(i) for i in range(1,K_MEANS_K + 1)]
ALL_FEATURES = OPENCV_FEATURES + THERMAL_FEATURES
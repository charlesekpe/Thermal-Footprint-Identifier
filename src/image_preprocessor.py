from temperature_preprocessing import TemperaturePreprocessing
import config
import os
import numpy as np
import cv2

class ImagePreprocessor:
    '''
    a class to extract descriptors from the videos with computer vision
    '''
    MEDIAN_KERNEL = 5
    GAUSSIAN_KERNEL = (5,5)
    
    def __init__(self, datafile):
        self.prepro = TemperaturePreprocessing(datafile)
        self.datafile = datafile
        
    def get_frame_index_hybrid_after_step_in(self, mode=max, plot=False):
        '''
        Calculates the frame index hybrid, by using TempPrepro step-in detection
        If no step-in point is detected, the max temperature index is used
        @Parameters
            mode: function
                the mode with which the step in will be detected
            plot: bool
                whether the temp values should be plotted with the index
        @Returns
            index: Int 
                index for the frame extraction       
        '''
        step_in = self.prepro.detect_step_in(mode=mode)
        if step_in:
            index = step_in + round(config.CAMERA_FPS * config.SECONDS_AFTER_STEP_IN)
        else:
            index = self.get_frame_index_as_max()
        if plot:
            self.prepro.plot_datafile(vertical_lines=[index])
        self.in_index = index
        return index
    
    def get_frame_index_hybrid_after_step_out(self, mode=max, plot=False):
        '''
        Calculates the frame index hybrid, by using TempPrepro step-out detection
        If no step-out point is detected, the, step-in point or max temperature index is used
        @Parameters
            mode: function
                the mode with which the step in will be detected
            plot: bool
                whether the temp values should be plotted with the index
        @Returns
            index: Int 
                index for the frame extraction       
        '''
        step_out = self.prepro.detect_step_out(mode=mode)
        if step_out:
            index = step_out + round(config.CAMERA_FPS * config.SECONDS_AFTER_STEP_IN)
        else:
            index = len(self.prepro.times) - config.FRAME_INTERVAL * (config.FRAMES_PER_PROCESS + 1)
        self.out_index = index
        return index

    def get_frame_index_as_max(self, plot=False):
        '''
        Calculates the frame index as the maximum temperature index in the video
        @Parameters
            plot: bool
                whether the temp values should be plotted with the index
        @Returns
            index: Int 
                index for the frame extraction       
        '''
        index = self.prepro.get_max_position()
        if plot:
            self.prepro.plot_datafile(index)
        self.index = index
        return index
    
    def get_frames(self, index, take_before=False, frame_interval=config.FRAME_INTERVAL, number_frames=config.FRAMES_PER_PROCESS ,show=False, save=False):
        '''
        Returns a given amount of frames from the video sequence, from position index.
        @Parameters:
            index: Int
                the position of the first frame in the video
            take_before: Bool
                if true, frames are extracted before position index
            frame_interval: Int
                in which interval the frames should be extracted.
                every 'frame_interal'th frame is extracted
            number_frames: Int
                the number of frames to extract
            show: Bool
                if true, the frame images are displayed
            save: Bool
                if true, the frame images are saved 
        @Returns
            frames: List
                the extracted frames
        '''
        index_array = [i for i in range(index, index + frame_interval * number_frames, frame_interval)]
        frames = []
        if take_before:
            index_array.reverse()
        for i in index_array:
            cap = cv2.VideoCapture('./data/' + self.datafile + '.ravi')
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            cap.release()
            frames.append(frame)
            if show:
                cv2.imshow(self.datafile, frame)
                cv2.waitKey()
            if save:
                cv2.imwrite(os.path.join(config.FOOT_IMG_PATH, 'frame.png'), frame)
        return frames

    def filter_and_greyscale(self, img, show=False, median_filter=True, save=False):
        '''
        filters and greyscales an imag
        @Parameters:
            img:
                an input image
            show: Bool
                if true, image is displayed in the process
            median_filter: Bool
                if true, median filter is applied, if false, gaussian filter is applied
            save: Bool
                if true, images are saved
        @Returns
            grey
                filtered and greyscaled image
        '''
        if median_filter:
            filtered = cv2.medianBlur(img, self.MEDIAN_KERNEL)
        else:
            filtered = cv2.GaussianBlur(img, self.GAUSSIAN_KERNEL, 0)
        grey = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        if show:
            cv2.imshow('greyscaled', grey)
            cv2.waitKey()
        if save:
            cv2.imwrite(os.path.join(config.FOOT_IMG_PATH, 'greyscaled.png'), grey)
            cv2.imwrite(os.path.join(config.FOOT_IMG_PATH, 'median.png'), filtered)
        return grey

    def detect_contours(self, img, median_filter=True, show=False, save=False):
        '''
        thresholds, filters and detects the contours in an image
        @Parameters:
            img:
                input image
            median_filter: Bool
                if true, median filter, if false, gaussian filter
            show: Bool
                if true, image is shown
            save: Bool
                if true, image is saved
        @Return:
            contours: list of int arrays
                the obtained contours of the image
        '''
        # apply binary inverted threshold
        ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        if show:
            cv2.imshow('thresholded', thresh)
            cv2.waitKey()
        if save:
            cv2.imwrite(os.path.join(config.FOOT_IMG_PATH, 'thresholded.png'), thresh)
        # apply filter
        if median_filter:
            filtered = cv2.medianBlur(thresh, self.MEDIAN_KERNEL)
        else:
            filtered = cv2.GaussianBlur(thresh, self.GAUSSIAN_KERNEL, 0)
        # detect contours
        contours, hierarchy = cv2.findContours(filtered, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        self.contours = contours
        return contours
    
    def draw_contours(self, img, contours, only_largest=False, show=False, save=False):
        '''
        draws the contours on to the image.
        @Parameters:
            img:
                the input image
            contours: list of int arrays
                the contours to draw
            only_largest: Bool
                if true, only the largest contour is drawn
            show: Bool
                if true, image is shown
            save: Bool
                if true, image is saved
        @Returns:
            copy:
                image with contours drawn
        '''
        if only_largest:
            largest_contour = max(contours, key=cv2.contourArea)
            contour = [largest_contour]
        else:
            contour = contours
        # copy the input image
        copy = img.copy()
        # draw contours onto image
        cv2.drawContours(image=copy, contours=contour, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        if show:
            cv2.imshow('contoured', copy)
            cv2.waitKey()
        if save:
            cv2.imwrite(os.path.join(config.FOOT_IMG_PATH, 'contoured.png'), copy)
        return copy
    
    def detect_sift(self, img, show=False, save=False):
        '''
        detects keypoints in an image using scale-invariant feature transform
        @Parameters:
            img:
                the input image
            show: Bool
                if true, keypoint image is shown
            save: Bool
                if true, keypoint image is saved
        @Returns:
            kp: List of cv2.Keypoint
                keypoints
            des: array of arrays of float
                descriptors of the keypoints 
        '''
        # creates the SIFT
        #sift = cv2.SIFT_create()
        sift = cv2.SURF_create()
        
        #sift = cv2.ORB_create(nfeatures=128)
        
        # detect and comput the keypoints and their descriptors
        kp, des = sift.detectAndCompute(img, None)
        print(kp)
        print(des)
        print("----")
        print(np.shape(kp))
        print(np.shape(des))
        if show:
            img = cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow('keypoints sift', img)
            cv2.waitKey()
        if save:
            img = cv2.drawKeypoints(img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imwrite(os.path.join(config.FOOT_IMG_PATH, 'keypoints.png'), img)
        self.descriptors = des
        self.keypoints = kp
        return kp, des
    
    def main_multi(self, show=False, save=False):
        '''
        main method for multiple frames. runs the preprocessing pipeline and returns the descriptors.
        @Parameters:
            show: Bool
                if true, all steps in the pipeline display the image they computed
            save: Bool
                if true, all steps in the pipeline save the image they computed 
        @Returns:
            des: array of arrays of float
                descriptors of the keypoints 
        '''
        # calculate the frame index
        index_in = self.get_frame_index_hybrid_after_step_in()
        # extract the frames
        frames = self.get_frames(index_in, show=show, save=save)
        # get number of the datafile
        number = int(self.datafile[-1])
        # if sock image and frames should be included, get trace frames as well
        if (6 <= number <= 9 or number == 0) and config.INCLUDE_TRACE_FRAMES:
            index_trace = self.get_frame_index_hybrid_after_step_out()
            frames_trace = self.get_frames(index_trace, show=show)
            frames.extend(frames_trace)
            print('trace frames added for file ', self.datafile)
        descriptors = []
        # for every frame
        for frame in frames:
            # filter and greyscale
            grey_img = self.filter_and_greyscale(frame, show=show, save=save)
            # detect the contours
            cont = self.detect_contours(grey_img, show=show, save=save)
            copy = grey_img.copy()
            # draw the contours on the copy of the grey image
            contoured_img = self.draw_contours(copy, cont, show=show, save=save)
            # get descriptors of the keypoints
            _, des = self.detect_sift(contoured_img, show=show, save=save)
            
            descriptors.append(des)
        return descriptors
    
    def save_to_file(self):
        '''
        runs the pipeline and saves the generated descriptors in a .npy file
        '''
        des = self.main_multi()
        #np.save(os.path.join(config.DES_PATH, self.datafile), des)
        np.save(os.path.join(config.SURF_PATH, self.datafile), des)

    
# generates all of the descriptors and saves them for all files, if called as main module
if __name__ == '__main__':
    files = ['munich' + str(i) + '-' + str(j) for i in range(1,len(config.PARTICIPANTS) + 1) for j in range(1,11)]# + config.CONTROL_DATA
    for file in files:
        print(file)
        proc = ImagePreprocessor(file)
        proc.save_to_file()
              
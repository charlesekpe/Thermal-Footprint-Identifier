import numpy as np
from matplotlib import pyplot as plt
from datafile import Datafile
from itertools import cycle, chain
import config
import os
from tensorflow.keras.layers.experimental.preprocessing import Normalization

class TemperaturePreprocessing:
    '''
    Class that uses .dat files to preprocess data and to generate the thermal inputs
    '''
    def __init__(self, participant):
        '''
        Initializes instance variables
        Loads data via the Datafile class
        '''
        loaded_ds = Datafile('./data/' + participant + ".dat")
        self.participant = participant
        self.datafile = loaded_ds.get_data_as_numpy_arrays()
        self.times = loaded_ds.get_times()
        self.avg_std = dict()
        self.distribution = dict()
        self.max_values = dict()
        self.temp_diffs = dict()
        self.decay_times = dict()
        self.complete_data = dict()
        self.max_foot_values = dict()
        self.step_in_index = None
        self.step_out_index = None
        df_number = int(self.participant[-1])
        self.is_socks = df_number in (list(range(6,10)) + [0])

    def rescale(self, axis=config.RESCALE_AXIS):
        '''
        rescale the data along an axis
        @Parameters
            axis: int
                the axis along which the data should be rescaled
        '''
        rescaled_dict = dict()
        for key, values in self.complete_data.items():
            normalizer = Normalization(axis=axis)
            normalizer.adapt(values)
            rescaled_dict[key] = normalizer(values)
        self.datafile = rescaled_dict

    def apply_threshold(self, min=config.MIN_THRESHOLD, max=config.MAX_THRESHOLD):
        '''
        applies a certain min and max threshold to the data
        ! Watch out: The da
        @Parameters
            min: float
                min threshold
            max: float
                max threshold    
        '''
        cropped_dict = dict()
        for key, values in self.datafile.items():
            value_array = []
            for value in values:
                if self.is_valid(value, min, max):
                    value_array.append(value)
            cropped_dict[key] = value_array
            if values == []:
                raise ValueError('AREA: ', key, ' Empty thresholded array. Data might be corrupted.')
        self.datafile = cropped_dict
            
    def detect_step_in(self, ftc=config.FRAMES_TO_CHECK, sd=config.STEP_DEGREES, mode=min, step_in_included=True):
        '''
        Detects the step in in the given datafile
        @Parameters
            ftc: int
                frames to check: detects the step in in the next ftc frames
            sd: int
                step degrees: the step in results in at least sd degrees celcius temperature difference
            mode: function
                the mode how the step in is detected, either min, max,mean,avg or median
            step_in_inlcuded: boolean
                should the step in process be included
        @Returns 
            index: int
                the index where the step in happened
        '''
        if not callable(mode):
            raise TypeError('No function provided as Mode')
        index_array = []
        for key, values in self.datafile.items():
            for idx, value in enumerate(values):
                if idx < len(values) - ftc and idx > config.IGNORE_FIRST_FRAMES:
                    if (values[idx + ftc] - value) >= sd and self.is_valid(value):
                        if step_in_included:
                            index_array.append(idx)
                        else:
                            index_array.append(idx+ftc)
                        break
        if len(index_array) > 0:
            index = round(mode(index_array))
            self.step_in_index = index
            return index
        else:
            print(self.participant + ': index array is empty. no step in was detected.')
            self.step_in_index = 0
            return 0
            
    def detect_step_out(self, ftc=config.FRAMES_TO_CHECK, sd=config.STEP_DEGREES, mode=max, step_out_included=True):
        '''
        Detects the step out point in the given data
        @Parameters
            ftc: int
                frames to check: detects the step out in the next ftc frames
            sd: int
                step degrees: the step out results in at least sd degrees celcius temperature difference
            mode: function
                the mode how the step out is detected, either min, max, mean, avg or median
            step_in_inlcuded: boolean
                should the step out process be included
        @Returns 
            index: int
                the index where the step out happened
        '''
        if not callable(mode):
            raise TypeError('No function provided as Mode')
        index_array = []
        if not self.step_in_index:
            self.detect_step_in()
        for key, values in self.datafile.items():
            for idx, value in enumerate(values[int(self.step_in_index or 0):], start=int(self.step_in_index or 0)):
                if idx < (len(values) - ftc):
                    if (value - values[idx + ftc]) >= sd and self.is_valid(value):
                        if step_out_included:
                            index_array.append(idx+ftc)
                        else:
                            index_array.append(idx)
                        break
        if len(index_array) > 0:
            index = round(mode(index_array))
            self.step_out_index = index
            return index
        else:
            print(self.participant + ': index array is empty. no step out was detected.')
            self.step_out_index = 0
            return 0
                    
    @staticmethod
    def is_valid(value, min=config.MIN_THRESHOLD, max=config.MAX_THRESHOLD):
        '''
        checks if a value is withing the threshold range
        @Parameters
            value: number
                the value to check
            min: number
                min threshold
            max: number
                max threshold
        @Returns 
            bool: whether it is valid or not
        '''
        if value > min and value < max:
            return True
        return False
    
    def eliminate_until_step_in(self, mode=min): 
        '''
        eliminates all values per foot region and in times from the step in index onwards
        @Parameters
            mode: function
                the step in detection mode
        '''
        index = self.detect_step_in(mode=mode)
        # eliminate the values from times until step in index
        self.times = self.times[index:]
        adj_length = len(self.times)
        # only take the last n values per foot region, if n = len of new times array
        for key, values in self.datafile.items():
            self.datafile[key] = values[-adj_length:] 
        # set step in index to 0
        self.step_in_index = 0
      
    def eliminate_from_step_out(self, mode=max):
        '''
        eliminates all values per foot region and in times until the step out index
        @Parameters
            mode: function
                the step out detection mode
        '''
        index = self.detect_step_out(mode=mode)
        # if step out not detected compute it 1.5 seconds after the step in index
        if index == 0:
            index = int(self.step_in_index or 0) + round(config.CAMERA_FPS * 1.5)
        # take the elements from times until index
        self.times = self.times[:index]
        adj_length = len(self.times)
        # take the first n elements from every foot region, if n = new len of times
        for key, values in self.datafile.items():
            self.datafile[key] = values[:adj_length]
            
    def eliminate_step_in_step_out(self, mode_in=min, mode_out=max):
        '''
        eliminates until step in and from step out
        @Parameters
            mode_in: function
                mode for step in detection
            mode_out: function
                mode for step out detection
        '''
        if self.step_in_index == None:
            in_index = self.detect_step_in(mode=mode_in)
        else:
            in_index = self.step_in_index
        if self.step_out_index == None:
            out_index = self.detect_step_out(mode=mode_out)
        if self.step_out_index == 0:
            out_index = in_index + round(config.CAMERA_FPS * 2)
        else:
            out_index = self.step_out_index
        self.times = self.times[in_index:out_index+1]
        for key, values in self.datafile.items():
            self.datafile[key] = values[in_index:out_index+1]
        self.step_out_index -= self.step_in_index
        self.step_in_index = 0
    
    def get_avg_std(self):
        '''
        Computes the average and standard deviation of the temp values per foot region
        '''
        for key, value in self.datafile.items():
            if(value == []):
                raise Exception('EMPTY ARRAY: ', key)
            self.avg_std[key] = [np.average(value), np.std(value)]
    
    def get_max_values(self):
        '''
        Computes the max value per foot region
        '''
        for key, value in self.datafile.items():
            max = np.max(list(filter(self.is_valid, value)))
            self.max_values[key] = max
            
    def get_distribution(self):
        '''
        computes the distribution of the avg per region to the total temp sum of all regions
        '''
        sum = 0
        for key, value in self.avg_std.items():
            sum += value[0]
        for key, value in self.avg_std.items():
            self.distribution[key] = (value[0] / sum)
            
    def get_temperature_difference(self):
        '''
        computes the temeperature difference between the max and the min value
        '''
        for key, value in self.datafile.items():
            min = np.min(list(filter(self.is_valid, value)))
            self.temp_diffs[key] = self.max_values[key] - min
        
    def get_decay_times(self):
        '''
        Computes the time (in seconds) it takes for the heat trace to vanish per foot region
        '''
        if self.step_out_index == None:
            out_index = self.detect_step_out(mode=max) + config.CAMERA_FPS
        elif self.step_out_index == 0:
            print('returned max position')
            out_index = self.get_max_position() + config.CAMERA_FPS * config.STEP_OUT_AFTER_MAX
        else:
            out_index = self.step_out_index + config.CAMERA_FPS
        out_index = min(out_index, len(self.times) - 1)
        for key, values in self.datafile.items():
            out_index_temp = values[out_index]
            max_index_temp = list(filter(self.is_valid, values))[-1]
            init_temp = list(filter(self.is_valid, values))[0]
            decay_rate = (out_index_temp - max_index_temp)/((len(values) - 1) - out_index)
            temp_diff = out_index_temp - init_temp
            if decay_rate == 0:
                self.decay_times[key] = 0
            else:
                self.decay_times[key] = (temp_diff / decay_rate) / config.CAMERA_FPS
    
    def get_feet_average(self, right_or_left):
        '''
        computes the maximum values of the averages among the regions per foot
        @Parameters
            right_or_left: string
                specifies whether the right or the left foot should be computed
        '''
        areas = [i for i in config.AREAS if i.startswith(right_or_left)]
        values_list = []
        for area in areas:
            values_list.append(self.datafile[area])
        zipped = list(zip(*values_list))
        mean_list = [sum(x)/len(x) for x in zipped if all(self.is_valid(value) for value in x)]
        print(mean_list)
        self.max_foot_values[right_or_left] = max(mean_list)
        print(self.max_foot_values[right_or_left])
        return self.max_foot_values[right_or_left]
    
    def default_pipeline(self):
        '''
        the default preprocessing and feature computation pipeline
        '''
        self.get_max_values()
        self.get_temperature_difference()
        if self.is_socks:
            self.get_decay_times()
        self.get_feet_average('left')
        self.get_feet_average('right')
        self.eliminate_step_in_step_out()
        self.get_avg_std()
        self.get_distribution()
        self.append_values()
        
    def get_max_position(self, start_frame=config.IGNORE_FIRST_FRAMES):
        '''
        computes the position where the sum of all regions is the largest
        @Parameters
            start_frame: int
                Frame where to start searching for the largest value
        @Returns
            int
                the index of the largest value
        '''
        zipped_dict = zip(*self.datafile.values())
        summed_list = [sum(x) for x in zipped_dict if all(self.is_valid(value) for value in x)]
        max_value = max(summed_list[start_frame:])
        return summed_list.index(max_value)
            
        
    def plot_datafile(self, vertical_lines=[], save=False):
        '''
        plots the datafile
        @Parameters
            vertical_lines: int
                indices where vertical lines should be drawn
        '''
        cycol = cycle('bgrcmy')
        for key, values in self.datafile.items():
            plt.plot(values, color=next(cycol),  marker=',', label=key, linestyle="-")
        if vertical_lines != []:
            if type(vertical_lines) == list:
                for i, line in enumerate(vertical_lines):
                    plt.axvline(line, color=str(i*0.2), linestyle='-')
            else:
                plt.axvline(vertical_lines, color='k', linestyle='-')
        plt.legend()
        plt.grid(axis='y', )
        plt.xlabel('Frames')
        plt.ylabel('° Celcius on avg.')
        plt.title('Datafile: ' + self.participant)
        if save:
            plt.savefig(os.path.join(config.PLOT_GRAPH_PATH, self.participant + '.png')) 
        plt.show()
        plt.close()
        
    def append_values(self):
        '''
        appends the values per region together in the complete_data dict
        '''
        for key, values in self.avg_std.items():
            self.complete_data[key] = list(chain(values, [self.distribution[key]], [self.max_values[key]], [self.temp_diffs[key]]))
            if self.is_socks:
                self.complete_data[key].append(self.decay_times[key])
            else:
                self.complete_data[key].append(0)
            if 'left' in key:
                self.complete_data[key].append(self.max_foot_values['left'])
            if 'right' in key:
                self.complete_data[key].append(self.max_foot_values['right'])
    
    def return_as_single_array(self):
        '''
        returns all preprocessed data as a single one-dim array
        @Returns
            list of floats
                the values stored together in a list
        '''
        all_values = []
        for key, values in self.complete_data.items():
            all_values.extend(values)
        return all_values
    
    def process_and_save_to_npy(self):
        '''
        runs the preprocessing pipeline and stores the data in npy feature files
        '''
        self.default_pipeline()
        data = self.return_as_single_array()
        np.save(os.path.join(config.THERMAL_PATH, self.participant), data)
    
def generate_all_files():
    '''
    generates the thermal features for all .dat files and saves them as npy files
    '''
    part_list = [participant + '-' + str(i) for participant in config.PARTICIPANTS for i in range(1,11)] #+ config.CONTROL_DATA
    for part in part_list:
        prepro = TemperaturePreprocessing(part)
        prepro.process_and_save_to_npy()
    
# when this file is executed, the generate_all_files() method is executed
if __name__ == '__main__':
    generate_all_files()
import numpy as np
from datetime import datetime
from matplotlib.dates import date2num

class Datafile:
    '''
    loads data from .dat file and parses it into arrays of float values and time values
    '''
    def __init__(self, datapath):
        # load data from datafile
        self.all_data = np.genfromtxt(datapath, skip_header=7, skip_footer=2, dtype=None, delimiter=' ', encoding='bytes')
        # initialize dict for temp values
        self.data = {'rightFootHeel': [], 'rightFootMid': [], 'rightFootToes': [], 'leftFootHeel': [], 'leftFootMid': [], 'leftFootToes': []}
        timesDatetime = []
        # for every line
        for single in self.all_data:
            # split by tabs
            byte_array = (single.split(b'\t'))
            # append first byte to times
            timesDatetime.append(datetime.strptime(byte_array[0].decode('utf-8'), '%H:%M:%S,%f'))
            # append following bytes to the data
            for i,key in enumerate(self.data, start=1):
                self.data[key].append(float((byte_array[i].decode('utf-8')).replace(',', '.')))
        self.times = date2num(timesDatetime)

    def get_data_as_numpy_arrays(self):
        '''
        takes the dict and returns it as dict of numpy arrays
        @Returns
            numpy_dict: dict of numpy arrays
        '''
        numpy_dict = dict()
        for key, value in self.data.items():
            numpy_dict[key] = np.array(value)
        return numpy_dict
    
    def get_times(self):
        '''
        returns the times array
        '''
        return self.times
    
# example run for testing purposes
if __name__ == "__main__":
    ds = Datafile('./data/munich1-1.dat')
    ds.get_data_as_numpy_arrays()
    ds.get_times()
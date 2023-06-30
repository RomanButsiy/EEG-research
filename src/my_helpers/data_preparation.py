from loguru import logger
from my_helpers.read_data.read_data_file import ReadDataFile
import numpy as np
import scipy.interpolate as interp

class DataPreparation(ReadDataFile):

    def __init__(self, eeg_config):
        super().__init__(eeg_config)
        self.getData()

        self.mod_sampling_rate = int(self.sampling_rate * self.eeg_config.getMultiplier())

        matrix_passivity_size = self.getNewMatrixSize(self.matrix_passivity[0])
        matrix_activity_size = self.getNewMatrixSize(self.matrix_activity[0])
        matrix_activity_size = matrix_passivity_size = self.mod_sampling_rate

        self.interp_matrix_passivity = [[],[],[]]
        self.interp_matrix_activity = [[],[],[]]

        for channel_number in range(len(self.signals)):
            for i in range(len(self.matrix_passivity[channel_number])):
                arr = np.array(self.matrix_passivity[channel_number][i])
                arr_interp = interp.interp1d(np.arange(arr.size), arr)
                arr_stretch = arr_interp(np.linspace(0, arr.size - 1, matrix_passivity_size))
                self.interp_matrix_passivity[channel_number].append(arr_stretch)

            for i in range(len(self.matrix_activity[channel_number])):
                arr = np.array(self.matrix_activity[channel_number][i])
                arr_interp = interp.interp1d(np.arange(arr.size), arr)
                arr_stretch = arr_interp(np.linspace(0, arr.size - 1, matrix_activity_size))
                self.interp_matrix_activity[channel_number].append(arr_stretch)

    def getNewMatrixSize(self, matrix):
        n = 0
        for i in range(len(matrix)):
            n = n + len(matrix[i])
        n = int((n / len(matrix)) * self.eeg_config.getMultiplier())
        # n = int(len(matrix[0]) * self.ecg_config.getMultiplier())
        return n
    
    def getModSamplingRate(self):
        return self.mod_sampling_rate
    
    def getPreparedData(self):
        return self.interp_matrix_passivity, self.interp_matrix_activity
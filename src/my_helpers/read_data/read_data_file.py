from loguru import logger
import pandas as pd
from pathlib import Path
from my_helpers.read_data.read_openbci_file import ReadOpenBCIFile

class ReadDataFile:

    def __init__(self, eeg_config):
        self.eeg_config = eeg_config
        data_type = eeg_config.getDataType()
        if data_type == 'openbci':
            self.instance = ReadOpenBCIFile(eeg_config)
        else:
            raise ValueError(f"Invalid data type: {data_type}")
        
    def __getattr__(self, name):
        return self.instance.__getattribute__(name)

    def getData(self):
        fr_path = f'{self.eeg_config.getImgPath()}/{self.eeg_config.getConfigBlock()}/function_rhythm.csv'
        if not Path(fr_path).is_file():
            e = 'The rhythm function file %s does not exist' % fr_path
            logger.error(e)
            raise FileNotFoundError(e)
        
        eeg_fr = pd.read_csv(fr_path)
        self.D_c = eeg_fr["D_c"]
        self.D_z = eeg_fr["D_z"]

        self.matrix_passivity = [[],[],[]]
        self.matrix_activity = [[],[],[]]

        for channel_number in range(len(self.signals)):
            for i in range(len(self.D_z) - 1):
                start = int((self.D_c[i] - 1) * self.sampling_rate)
                end = int((self.D_z[i] - 1) * self.sampling_rate)
                self.matrix_passivity[channel_number].append(self.signals[channel_number][start:end])

                start = int((self.D_z[i] - 1) * self.sampling_rate)
                end = int((self.D_c[i + 1] - 1) * self.sampling_rate)
                self.matrix_activity[channel_number].append(self.signals[channel_number][start:end])

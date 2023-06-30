from loguru import logger
import csv
from scipy import signal
import numpy as np

class ReadOpenBCIFile:

    def __init__(self, ecg_config):
        data_path = f'{ecg_config.getDataPath()}/{ecg_config.getFileName()}'
        logger.info("Read OpenBCI file")
        logger.info(data_path)

        input_sample_rate = 250 # !!! 250 !!!
        band = (1,17)

        ADS1299_Vref = 4.5  #reference voltage for ADC in ADS1299.  set by its hardware
        ADS1299_gain = 24.  #assumed gain setting for ADS1299
        scale_fac_uVolts_per_count = ADS1299_Vref / ((float)(pow(2, 23)-1)) / ADS1299_gain * 1000000.
        input_headers = ['id','ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8','accel1','accel2','accel3']
        output_data = []
        output_time_data = []
        output_val_data = []
        time_counter = 1
        time_increment = float(1) / float(input_sample_rate)

        with open(data_path, 'r') as csvfile:
        
            csv_input = csv.DictReader((line.replace('\0','') for line in csvfile), fieldnames=input_headers, dialect='excel')
            row_count = 0

            for row in csv_input:
                row_count = row_count + 1
                if(row_count > 2):
                    output = {}
                    val_data = [0.] * 3
                    time_counter = time_counter + time_increment
                    output['time'] = time_counter
                    output_time_data.append(round(time_counter, 3))

                    for i in range(1, 4):
                        channel_key = 'ch'+str(i)
                        output[channel_key] = self.parseInt24Hex(row[channel_key]) * scale_fac_uVolts_per_count
                        val_data[i - 1] = self.parseInt24Hex(row[channel_key]) * scale_fac_uVolts_per_count
                    output_val_data.append(val_data)
                    output_data.append(output)

        logger.debug('End read data from file')

        y_T = np.transpose(output_val_data)
        self.sampling_rate = input_sample_rate

        notch_channels = []
        for i in range(len(y_T)):
            notch_channels.append(self.notch(50, y_T[i], fs = self.sampling_rate))

        bandpass_notch_channels = []
        for i in range(len(notch_channels)):
            bandpass_notch_channels.append(self.bandpass(band[0],band[1],notch_channels[i], fs = self.sampling_rate))
        
        self.signals = bandpass_notch_channels

        logger.info(f'Sampling rate: {self.sampling_rate}')


    def parseInt24Hex(self, hex):
        if hex is None:
            return 0.
        if (hex[:1] > '7'):
            hex = "FF" + hex
        else:
            hex = "00" + hex
        n = int(hex, 16) & 0xffffffff
        return n | (-(n & 0x80000000))
        
    def bandpass(self, start, stop, data, fs):
        bp_Hz = np.array([start, stop])
        b, a = signal.butter(5, bp_Hz / (fs / 2.0), btype='bandpass')
        return signal.lfilter(b, a, data, axis=0)

    def notch(self, val, data, fs):
        notch_freq_Hz = np.array([float(val)])
        for freq_Hz in np.nditer(notch_freq_Hz):
            bp_stop_Hz = freq_Hz + 3.0 * np.array([-1, 1])
            b, a = signal.butter(3, bp_stop_Hz / (fs / 2.0), 'bandstop')
            fin = data = signal.lfilter(b, a, data)
        return fin
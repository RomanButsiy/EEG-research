from loguru import logger
import pandas as pd
import numpy as np
import neurokit2 as nk
from pathlib import Path
import matplotlib.pyplot as plt
import re
import sys
import time

from classification_metrics.confusion_matrix import ConfusionMatrix


class StatisticsDistance():

    def __init__(self, eeg_config, data):
        self.eeg_config = eeg_config
        logger.debug("Statistics Distance")
        self.sampling_rate = data.getModSamplingRate()
        data_matrix_passivity, data_matrix_activity = data.getPreparedData()

        self.all_matrix_passivity = np.array(data_matrix_passivity)
        self.all_matrix_activity = np.array(data_matrix_activity)

        self.confusion_matrix_names = [
            "True Positive Rate",
            "True Negative Rate", "Positive Predictive Value", "Negative Predictive Value", "False Negative Rate",
            "False Positive Rate", "False Discovery Rate", "False Omission Rate", "Positive Likelihood Ratio",
            "Negative Likelihood Ratio", "Prevalence Threshold", "Threat Score", "Accuracy", "Balanced Accuracy",
            "F1 score", "Matthews Correlation Coefficient", "Fowlkes-Mallows index", "Bookmaker Informedness", 
            "Markedness", "Diagnostic Odds Ratio", "Learning_time", "Testing_time"
        ]

        self.statistics_files = [
            "1 Mathematical Expectation",
            "2 Initial Moments Second Order",
            "3 Initial Moments Third Order",
            "4 Initial Moments Fourth Order"
        ]

    def Calculate(self):
        logger.debug("Calculate")
        num_channels = len(self.all_matrix_passivity)
        path = f'{self.eeg_config.getImgPath()}/{self.eeg_config.getConfigBlock()}/Mathematical Statistics/CSV'

        matrix = []
        target_matrix = []

        # channels_range = [0, 1]

        channels_range = range(num_channels) if 'channels_range' not in locals() and 'channels_range' not in globals() else channels_range

        for channel in range(num_channels):
            passivity_data = self.all_matrix_passivity[channel]
            activity_data = self.all_matrix_activity[channel]

            channel_matrix = [*passivity_data, *activity_data]
            channel_target_matrix = [*[False] * len(passivity_data), *[True] * len(activity_data)]

            matrix.append(channel_matrix)
            target_matrix.append(channel_target_matrix)

        for statistics_file in self.statistics_files:
            res_col = ["Average"]
            res_cm = []
            match = re.match(r"(\d+)\s*(.*)", statistics_file)
            if match:
                exponentiation = int(match.group(1))
            else:
                logger.error(f'Pattern not found: {statistics_file}')
                sys.exit(1)
            lstart = time.time()
            statistic_p, statistic_a = self.read_channel_data(path, statistics_file, num_channels)
            lend = rmsestart = time.time()
            rmse_p = self.rmse(statistic_p, matrix, num_channels, exponentiation)
            rmse_a = self.rmse(statistic_a, matrix, num_channels, exponentiation)
            compare_rmse = rmse_p > rmse_a
            rmseend = time.time()

            y_true_all = np.concatenate([target_matrix[i] for i in channels_range])
            y_pred_all = np.concatenate([compare_rmse[i] for i in channels_range])
            allend = time.time()
            ltime = (lend-lstart)*10**3
            allttime = (allend-rmsestart)*10**3
            confusion_matrix = ConfusionMatrix(y_true_all, y_pred_all, ltime, allttime)
            print(statistics_file)
            print(("%s: %.2f" % ("Accuracy Average",  confusion_matrix.getACC() * 100)))
            res_cm.append(confusion_matrix.getAllVariables())

            for channel in channels_range:
                cstart = time.time()
                y_true = np.array(target_matrix[channel])
                y_pred = np.array(compare_rmse[channel])
                cend = time.time()
                cttime = ((rmseend-rmsestart) + (cend - cstart))*10**3
                confusion_matrix = ConfusionMatrix(y_true, y_pred, ltime, cttime)
                print(("%s %i: %.2f" % ("Accuracy Channel", channel + 1 ,  confusion_matrix.getACC() * 100)))
                res_cm.append(confusion_matrix.getAllVariables())
                res_col.append("%s_%i" % ("Channel", channel + 1))

            path_out = f'{self.eeg_config.getImgPath()}/{self.eeg_config.getConfigBlock()}/Confusion matrix'
            Path(path_out).mkdir(parents=True, exist_ok=True)
            
            df = pd.DataFrame(np.transpose(np.round(res_cm, 2)), index=self.confusion_matrix_names, columns=res_col)
            df.to_csv(f'{path_out}/{statistics_file}.csv')

            
    def rmse(self, statistic, all_matrix, num_channels, exponentiation):
        all_matrix = np.power(all_matrix, exponentiation)
        rmse_results = []
        for channel_idx in range(num_channels):
            channel_expectation = statistic[channel_idx]
            channel_signals = all_matrix[channel_idx]
            channel_rmse = [np.sqrt(np.mean((signal - channel_expectation) ** 2)) for signal in channel_signals]
            rmse_results.append(channel_rmse)
        return np.array(rmse_results)



    def read_channel_data(self, base_path, file_pattern, num_channels):
        passivity_file = pd.read_csv(f'{base_path}/{file_pattern} passivity.csv')
        activity_file = pd.read_csv(f'{base_path}/{file_pattern} activity.csv')

        passivity_channel_data = [passivity_file[f'Data_{channel}'] for channel in range(num_channels)]
        activity_channel_data = [activity_file[f'Data_{channel}'] for channel in range(num_channels)]

        return np.array(passivity_channel_data), np.array(activity_channel_data)

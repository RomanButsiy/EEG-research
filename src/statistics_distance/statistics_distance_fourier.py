from loguru import logger
import pandas as pd
import numpy as np
import neurokit2 as nk
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.integrate import simps
import re
import sys
import time

from classification_metrics.confusion_matrix import ConfusionMatrix
from my_helpers.mathematical_statistics import MathematicalStatistics


class StatisticsDistanceFourier():

    def __init__(self, eeg_config, data, fourier_type, terms):
        self.eeg_config = eeg_config
        self.terms = terms
        self.fourier_type = fourier_type
        logger.debug("Statistics Distance")
        self.sampling_rate = data.getModSamplingRate()
        data_matrix_passivity, data_matrix_activity = data.getPreparedData()

        fstart = time.time()
        self.all_matrix_passivity = [[self.getFourierSeries(m, fourier_type, terms=self.terms) for m in channel] for channel in data_matrix_passivity]
        self.all_matrix_activity = [[self.getFourierSeries(m, fourier_type, terms=self.terms) for m in channel] for channel in data_matrix_activity]
        fend = time.time()
        self.ftime = (fend-fstart)*10**3

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
            # "2 Initial Moments Second Order",
            # "3 Initial Moments Third Order",
            # "4 Initial Moments Fourth Order"
        ]
        self.GetStatistic()

    def Calculate(self):
        logger.debug("Calculate")
        num_channels = len(self.all_matrix_passivity)
        path = f'{self.eeg_config.getImgPath()}/{self.eeg_config.getConfigBlock()}/Mathematical Statistics Fourier/{self.fourier_type}/CSV'

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
            statistic_p, statistic_a = self.read_channel_data(path, statistics_file, num_channels)
            rmsestart = time.time()
            rmse_p = self.rmse(statistic_p, matrix, num_channels, exponentiation)
            rmse_a = self.rmse(statistic_a, matrix, num_channels, exponentiation)
            compare_rmse = rmse_p > rmse_a
            rmseend = time.time()

            y_true_all = np.concatenate([target_matrix[i] for i in channels_range])
            y_pred_all = np.concatenate([compare_rmse[i] for i in channels_range])
            allend = time.time()
            allttime = (allend-rmsestart)*10**3 + self.ftime
            confusion_matrix = ConfusionMatrix(y_true_all, y_pred_all, self.ftime, allttime)
            print(statistics_file)
            print(("%s: %.2f" % ("Accuracy Average",  confusion_matrix.getACC() * 100)))
            res_cm.append(confusion_matrix.getAllVariables())

        #     for channel in channels_range:
        #         cstart = time.time()
        #         y_true = np.array(target_matrix[channel])
        #         y_pred = np.array(compare_rmse[channel])
        #         cend = time.time()
        #         cttime = ((rmseend-rmsestart) + (cend - cstart))*10**3
        #         confusion_matrix = ConfusionMatrix(y_true, y_pred, ltime, cttime)
        #         print(("%s %i: %.2f" % ("Accuracy Channel", channel + 1 ,  confusion_matrix.getACC() * 100)))
        #         res_cm.append(confusion_matrix.getAllVariables())
        #         res_col.append("%s_%i" % ("Channel", channel + 1))

        #     path_out = f'{self.eeg_config.getImgPath()}/{self.eeg_config.getConfigBlock()}/Confusion matrix'
        #     Path(path_out).mkdir(parents=True, exist_ok=True)
            
        #     df = pd.DataFrame(np.transpose(np.round(res_cm, 2)), index=self.confusion_matrix_names, columns=res_col)
        #     df.to_csv(f'{path_out}/{statistics_file}.csv')
        path = f'{self.eeg_config.getImgPath()}/{self.eeg_config.getConfigBlock()}/Confusion matrix/{self.fourier_type}'
        df = pd.read_csv(f'{path}/n-{self.terms}.csv')
        df["SPC"] = np.transpose(np.round(res_cm, 2)) #SPC

        df.to_csv(f'{path}/n-{self.terms}.csv', index=False)

            
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


    def GetStatistic(self):
        logger.debug("Mean Fourier")
        path = f'{self.eeg_config.getImgPath()}/{self.eeg_config.getConfigBlock()}/Mathematical Statistics Fourier/{self.fourier_type}/CSV'

        self.process_matrix(self.all_matrix_passivity, "passivity", path)
        self.process_matrix(self.all_matrix_activity, "activity", path)

    def process_matrix(self, matrices, matrix_type, path):
        statistics_matrix = []
        for matrix in zip(matrices):
            statistics_matrix.append(MathematicalStatistics(matrix).getMathematicalStatistics())
        self.plot_to_csv(path, [statistic.getMathematicalExpectation() for statistic in statistics_matrix], f"1 Mathematical Expectation {matrix_type}")
        self.plot_to_csv(path, [statistic.getInitialMomentsSecondOrder() for statistic in statistics_matrix], f"2 Initial Moments Second Order {matrix_type}")
        self.plot_to_csv(path, [statistic.getInitialMomentsThirdOrder() for statistic in statistics_matrix], f"3 Initial Moments Third Order {matrix_type}")
        self.plot_to_csv(path, [statistic.getInitialMomentsFourthOrder() for statistic in statistics_matrix], f"4 Initial Moments Fourth Order {matrix_type}")

    def plot_to_csv(self, path, plot, name):
        logger.info(f"Save {name}.csv")
        Path(path).mkdir(parents=True, exist_ok=True)
        plot = np.transpose(plot)
        time = np.arange(0, len(plot), 1) / self.sampling_rate
        data = pd.DataFrame({'Time': time, **{f'Data_{i}': plot[:, i] for i in range(plot.shape[1])}})
        data.to_csv(f'{path}/{name}.csv')

    def getFourierSeries(self, y, fourier_type, terms = 40, L = 1):
        x = np.linspace(0, L, self.sampling_rate, endpoint=False)
        a0 = 2./L*simps(y,x)
        an = lambda n:2.0/L*simps(y*np.cos(2.*np.pi*n*x/L),x)
        bn = lambda n:2.0/L*simps(y*np.sin(2.*np.pi*n*x/L),x)
        list_a = np.abs([an(k) for k in range(1, terms + 1)])
        list_b = np.abs([bn(k) for k in range(1, terms + 1)])
        if fourier_type == "an":
            return [a0, *list_a]
        if fourier_type == "bn":
            return [0, *list_b]
        return [a0, *list_a, *list_b]
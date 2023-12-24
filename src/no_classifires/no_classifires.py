from loguru import logger
import pandas as pd
import numpy as np
import neurokit2 as nk
from pathlib import Path
import matplotlib.pyplot as plt
import time

from classification_metrics.confusion_matrix import ConfusionMatrix


class NoClassidire():

    def __init__(self, eeg_config, data):
        self.eeg_config = eeg_config
        logger.debug("No Classifires")
        self.sampling_rate = data.getModSamplingRate()
        data_matrix_passivity, data_matrix_activity = data.getPreparedData()
        self.n_sigma = eeg_config.getSigma()[0]


        self.data_matrix_passivity_1 = np.array(data_matrix_passivity[0]) + 10
        self.data_matrix_activity_1 = np.array(data_matrix_activity[0]) + 10

        # self.data_matrix_passivity_1 = self.CalculateAverage(data_matrix_passivity)
        # self.data_matrix_activity_1 = self.CalculateAverage(data_matrix_activity)

        self.confusion_matrix_names = [
            "True Positive Rate",
            "True Negative Rate", "Positive Predictive Value", "Negative Predictive Value", "False Negative Rate",
            "False Positive Rate", "False Discovery Rate", "False Omission Rate", "Positive Likelihood Ratio",
            "Negative Likelihood Ratio", "Prevalence Threshold", "Threat Score", "Accuracy", "Balanced Accuracy",
            "F1 score", "Matthews Correlation Coefficient", "Fowlkes-Mallows index", "Bookmaker Informedness", 
            "Markedness", "Diagnostic Odds Ratio", "Learning_time", "Testing_time"
        ]
        self.NoAllSigma()

    def CalculateAverage(self, data_matrix):
        assert all(len(channel) == len(data_matrix[0]) for channel in data_matrix)
        sum_array = np.zeros_like(data_matrix[0])
        for channel in data_matrix:
            for i in range(len(channel)):
                sum_array[i] += channel[i]
        return (np.array(sum_array / len(data_matrix)) + 10)

    def NoTest(self):
        logger.debug("No Test Sigma")
        path = f'{self.eeg_config.getImgPath()}/{self.eeg_config.getConfigBlock()}/All Mean'
        p_sigma = pd.read_csv(f'{path}/CSV/Passivity Sigma.csv')["Data"]
        a_sigma = pd.read_csv(f'{path}/CSV/Activity Sigma.csv')["Data"]
        p_mean = pd.read_csv(f'{path}/CSV/Passivity Mathematical Expectation.csv')["Data"]
        a_mean = pd.read_csv(f'{path}/CSV/Activity Mathematical Expectation.csv')["Data"]

        matrix = [*self.data_matrix_passivity_1, *self.data_matrix_activity_1]
        fa_target_matrix = [False] * len(self.data_matrix_passivity_1)
        fa_target_matrix_2 = [True] * len(self.data_matrix_activity_1)
        target_matrix = [*fa_target_matrix, *fa_target_matrix_2]

        c_all = []
        res_col = []
        for exponentiation in [1]: 

            res = []

            n_sigma = self.n_sigma

            lstart = time.time()
            p_upper_bound = np.power(p_mean + (n_sigma * p_sigma), exponentiation)
            p_lower_bound = np.power(p_mean - (n_sigma * p_sigma), exponentiation)
            a_upper_bound = np.power(a_mean + (n_sigma * a_sigma), exponentiation)
            a_lower_bound = np.power(a_mean - (n_sigma * a_sigma), exponentiation)
            lend = time.time()
            ltime = (lend-lstart)*10**3 + self.ltime

            # min_upper = np.minimum(p_upper_bound, a_upper_bound)
            # max_lower = np.maximum(p_lower_bound, a_lower_bound)
            # overlap_length = np.maximum(0, min_upper - max_lower)

            # possible_overlap_length = (a_upper_bound - a_lower_bound) + (p_upper_bound - p_lower_bound)
            # relative_overlap = overlap_length / possible_overlap_length
            # average_relative_overlap = np.mean(relative_overlap)

            # print(average_relative_overlap)

            tstart = time.time()

            for mean in matrix:

                mean = mean ** exponentiation

                p_within_bounds = (mean >= p_lower_bound) & (mean <= p_upper_bound)
                a_within_bounds = (mean >= a_lower_bound) & (mean <= a_upper_bound)

                percent_p_within_bounds = np.mean(p_within_bounds) * 100
                percent_a_within_bounds = np.mean(a_within_bounds) * 100
                res.append(not(percent_p_within_bounds > percent_a_within_bounds))

            tend = time.time()
            ttime = (tend-tstart)*10**3

            y_true = np.array(target_matrix)
            y_pred = np.array(res)
            confusion_matrix = ConfusionMatrix(y_true, y_pred, ltime, ttime)
            c_all.append(confusion_matrix.getAllVariables())
            res_col.append(f'Exponentiation {exponentiation}')

            print(("%s: %.2f" % ("Accuracy",  confusion_matrix.getACC() * 100)))

        path_out = f'{self.eeg_config.getImgPath()}/{self.eeg_config.getConfigBlock()}/Confusion matrix'
        Path(path_out).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(np.transpose(np.round(c_all, 2)), index=self.confusion_matrix_names, columns=res_col)
        df.to_csv(f'{path_out}/NoClassidire.csv')
    
    
            # path = f'{self.eeg_config.getImgPath()}/{self.eeg_config.getConfigBlock()}/Mathematical Statistics'
            # Path(path).mkdir(parents=True, exist_ok=True)
            # plt.clf()
            # plt.rcParams.update({'font.size': 14})
            # f, axis = plt.subplots(1)
            # f.tight_layout()
            # f.set_size_inches(12, 6)
            # axis.grid(True)
            # time = np.arange(0, len(mean), 1) / self.sampling_rate
            # axis.plot(time, mean, linewidth=3)
            # # axis.plot(time, passivity_mean, linewidth=3)
            # axis.plot(time, p_upper_bound, linewidth=3)
            # axis.plot(time, p_lower_bound, linewidth=3)
            # # axis.axis(ymin = -8, ymax = 8)
            # plt.savefig(f'{path}/{t}-Passivity test -> {"Activity" if target else "Passivity"}.png', dpi=300)
    
            # plt.clf()
            # plt.rcParams.update({'font.size': 14})
            # f, axis = plt.subplots(1)
            # f.tight_layout()
            # f.set_size_inches(12, 6)
            # axis.grid(True)
            # time = np.arange(0, len(mean), 1) / self.sampling_rate
            # axis.plot(time, mean, linewidth=3)
            # # axis.plot(time, activity_mean, linewidth=3)
            # axis.plot(time, a_upper_bound, linewidth=3)
            # axis.plot(time, a_lower_bound, linewidth=3)
            # # axis.axis(ymin = -8, ymax = 8)
            # plt.savefig(f'{path}/{t}-Activity test -> {"Activity" if target else "Passivity"}.png', dpi=300)

            # break

    def NoAllSigma(self):
        logger.debug("No All Sigma")
        
        lstart = time.time()
        data_matrix_activity = np.transpose(self.data_matrix_activity_1)
        all_mean_activity_data = [np.mean(i) for i in data_matrix_activity]
        data_matrix_passivity = np.transpose(self.data_matrix_passivity_1)
        all_mean_passivity_data = [np.mean(i) for i in data_matrix_passivity]

        all_sigma_activity_data = np.std(data_matrix_activity, axis=1, ddof=1)
        all_sigma_passivity_data = np.std(data_matrix_passivity, axis=1, ddof=1)
        lend = time.time()
        self.ltime = (lend-lstart)*10**3

        xtext = "$t, s$"
        path = f'{self.eeg_config.getImgPath()}/{self.eeg_config.getConfigBlock()}/All Mean'
        # self.plot_to_png(path, all_mean_activity_data, "Activity Mathematical Expectation", xtext=xtext, ytext=r"$m_{{\xi}} (t), mV$", ylim=(5, 15))
        # self.plot_to_png(path, all_mean_passivity_data, "Passivity Mathematical Expectation", xtext=xtext, ytext=r"$m_{{\xi}} (t), mV$", ylim=(5, 15))
        self.plot_to_csv(all_mean_activity_data, "Activity Mathematical Expectation")
        self.plot_to_csv(all_mean_passivity_data, "Passivity Mathematical Expectation")

        xtext = "$t, s$"
        path = f'{self.eeg_config.getImgPath()}/{self.eeg_config.getConfigBlock()}/All Mean'
        # self.plot_to_png(path, all_sigma_activity_data, "Activity Sigma", xtext=xtext, ytext=r"$m_{{\xi}} (t), mV$")
        # self.plot_to_png(path, all_sigma_passivity_data, "Passivity Sigma", xtext=xtext, ytext=r"$m_{{\xi}} (t), mV$")
        self.plot_to_csv(all_sigma_activity_data, "Activity Sigma")
        self.plot_to_csv(all_sigma_passivity_data, "Passivity Sigma")

    def plot_to_png(self, path, plot, name, xlim = None, ylim = None, ytext="", xtext="", size=(12, 6)):
        logger.info(f"Plot {name}.png")
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.clf()
        plt.rcParams.update({'font.size': 14})
        f, axis = plt.subplots(1)
        f.tight_layout()
        f.set_size_inches(size[0], size[1])
        axis.grid(True)
        time = np.arange(0, len(plot), 1) / self.sampling_rate
        axis.plot(time, plot, linewidth=3)
        axis.set_xlabel(xtext, loc = 'right')
        axis.set_title(ytext, loc = 'left', fontsize=14, position=(-0.06, 0))
        if xlim is not None:
            axis.axis(xmin = xlim[0], xmax = xlim[1])
        if ylim is not None:
            axis.axis(ymin = ylim[0], ymax = ylim[1])
        plt.savefig(f'{path}/{name}.png', dpi=300)

    def plot_to_csv(self, plot, name):
        logger.info(f"Save {name}.csv")
        path = f'{self.eeg_config.getImgPath()}/{self.eeg_config.getConfigBlock()}/All Mean/CSV'
        Path(path).mkdir(parents=True, exist_ok=True)
        time = np.arange(0, len(plot), 1) / self.sampling_rate
        data = pd.DataFrame({"Time" : time, "Data" : plot})
        nk.write_csv(data, f'{path}/{name}.csv')
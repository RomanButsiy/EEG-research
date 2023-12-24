from loguru import logger
import pandas as pd
import numpy as np
import neurokit2 as nk
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.integrate import simps
import time

from classification_metrics.confusion_matrix import ConfusionMatrix


class NoClassidireFourierAllChanels():

    def __init__(self, eeg_config, data, fourier_type, terms):

        self.terms = terms
        # fourier_type = "an"
        # fourier_type = "bn"
        # fourier_type = "an_bn"

        self.eeg_config = eeg_config
        logger.debug("No Classifires Fourier")
        self.sampling_rate = data.getModSamplingRate()
        data_matrix_passivity, data_matrix_activity = data.getPreparedData()
        self.n_sigma = eeg_config.getSigma()[1]

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
        self.fourier_type = fourier_type
        self.ltime = []
        self.NoAllSigma()

    def NoTest(self):
        logger.debug("No Test Sigma")

        num_channels = len(self.all_matrix_passivity)
        path = f'{self.eeg_config.getImgPath()}/{self.eeg_config.getConfigBlock()}/All Mean Fourier/{self.fourier_type}/CSV'

        p_sigmas = self.read_channel_data(path, "Passivity Sigma", num_channels)
        a_sigmas = self.read_channel_data(path, "Activity Sigma", num_channels)
        p_means = self.read_channel_data(path, "Passivity Mathematical Expectation", num_channels)
        a_means = self.read_channel_data(path, "Activity Mathematical Expectation", num_channels)

        # channels_range = [0, 1]

        res_all = []
        matrix = []
        cm_all = []
        cm_col = []
        target_matrix = []
        average_relative_overlaps = []
        channels_range = range(num_channels) if 'channels_range' not in locals() and 'channels_range' not in globals() else channels_range

        for channel in range(num_channels):
            passivity_data = self.all_matrix_passivity[channel]
            activity_data = self.all_matrix_activity[channel]

            channel_matrix = [*passivity_data, *activity_data]
            channel_target_matrix = [*[False] * len(passivity_data), *[True] * len(activity_data)]

            matrix.append(channel_matrix)
            target_matrix.append(channel_target_matrix)

        exponentiation = 1
        n_sigma = self.n_sigma

        lstart = time.time()
        p_upper_bounds = np.power(p_means + (n_sigma * p_sigmas), exponentiation)
        p_lower_bounds = np.power(p_means - (n_sigma * p_sigmas), exponentiation)
        a_upper_bounds = np.power(a_means + (n_sigma * a_sigmas), exponentiation)
        a_lower_bounds = np.power(a_means - (n_sigma * a_sigmas), exponentiation)
        p_lower_bounds[p_lower_bounds < 0] = 0
        a_lower_bounds[a_lower_bounds < 0] = 0
        lend = time.time()
        ltime = np.sum(np.array(self.ltime))  + ((lend-lstart)*10**3) + self.ftime

        p_res_by_channel_all = []
        a_res_by_channel_all = []

        tstart = time.time()
        for channel in range(num_channels):
            p_res_by_channel = []
            a_res_by_channel = []
            res_by_channel = []
            for mean in matrix[channel]:
            
                mean = np.power(mean, exponentiation)

                p_within_bounds = (mean >= p_lower_bounds[channel]) & (mean <= p_upper_bounds[channel])
                a_within_bounds = (mean >= a_lower_bounds[channel]) & (mean <= a_upper_bounds[channel])

                res_by_channel.append(not(np.sum(p_within_bounds) > np.sum(a_within_bounds)))
                p_res_by_channel.append(np.sum(p_within_bounds))
                a_res_by_channel.append(np.sum(a_within_bounds))

            p_res_by_channel_all.append(p_res_by_channel)
            a_res_by_channel_all.append(a_res_by_channel)
            res_all.append(res_by_channel)
        tend = time.time()

        p_res_sum_all = np.sum([p_res_by_channel_all[i] for i in channels_range], axis=0)
        a_res_sum_all = np.sum([a_res_by_channel_all[i] for i in channels_range], axis=0)

        y_pred_all = np.logical_not(p_res_sum_all > a_res_sum_all)
        as_tend = time.time()
        ttime = ((as_tend-tstart)*10**3) + self.ftime

        confusion_matrix = ConfusionMatrix(target_matrix[0], y_pred_all, ltime, ttime)
        cm_all.append(confusion_matrix.getAllVariables())
        cm_col.append('Accuracy Sum')
        print(("%s: %.2f" % ("Accuracy Sum",  confusion_matrix.getACC() * 100)))
        
        # for channel in channels_range:
        #     c_tstart = time.time()
        #     y_true = np.array(target_matrix[channel])
        #     y_pred = np.array(res_all[channel])
        #     c_tend = time.time()
        #     c_ttime = ((tend - tstart) + ((c_tend - c_tstart))*10**3) + self.ftime
        #     confusion_matrix = ConfusionMatrix(y_true, y_pred, ltime, c_ttime)
        #     cm_all.append(confusion_matrix.getAllVariables())
        #     cm_col.append(f'Accuracy Channel {channel + 1}')
        #     print(("%s %i: %.2f" % ("Accuracy Channel", channel + 1 ,  confusion_matrix.getACC() * 100)))

        # c_tstart = time.time()
        # y_true_all = np.concatenate([target_matrix[i] for i in channels_range])
        # y_pred_all = np.concatenate([res_all[i] for i in channels_range])
        # c_tend = time.time()
        # c_ttime = ((tend - tstart) + ((c_tend - c_tstart))*10**3) + self.ftime


        
        # confusion_matrix = ConfusionMatrix(y_true_all, y_pred_all, ltime, c_ttime)
        # cm_all.append(confusion_matrix.getAllVariables())
        # cm_col.append('Accuracy Average')
        # print(("%s: %.2f" % ("Accuracy Average",  confusion_matrix.getACC() * 100)))

        # path_out = f'{self.eeg_config.getImgPath()}/{self.eeg_config.getConfigBlock()}/Confusion matrix'
        # Path(path_out).mkdir(parents=True, exist_ok=True)
        # df = pd.DataFrame(np.transpose(np.round(cm_all, 2)), index=self.confusion_matrix_names, columns=cm_col)
        # df.to_csv(f'{path_out}/NoClassidireFourierAllChanels {self.fourier_type}.csv')

        path = f'{self.eeg_config.getImgPath()}/{self.eeg_config.getConfigBlock()}/Confusion matrix/{self.fourier_type}'
        df = pd.read_csv(f'{path}/n-{self.terms}.csv')
        df["SIC"] = np.transpose(np.round(cm_all, 2)) #SPC

        df.to_csv(f'{path}/n-{self.terms}.csv', index=False)
    

    def NoAllSigma(self):
        logger.debug("No All Mean Fourier")
        xtext = "$n$"
        path = f'{self.eeg_config.getImgPath()}/{self.eeg_config.getConfigBlock()}/All Mean Fourier/{self.fourier_type}'

        self.process_matrix(self.all_matrix_passivity, "Passivity", path, xtext)
        self.process_matrix(self.all_matrix_activity, "Activity", path, xtext)

    def process_matrix(self, matrices, matrix_type, path, xtext):
        for index, matrix in enumerate(matrices):
            lstart = time.time()
            data_matrix = np.transpose(matrix)
            all_mean_data = np.mean(data_matrix, axis=1)
            lend = time.time()
            self.ltime.append((lend-lstart)*10**3)
            title = f"{matrix_type} Mathematical Expectation {index}"
            # self.plot_to_png(path, all_mean_data, title, xtext=xtext, ytext=self.fourier_type)
            self.plot_to_csv(path, all_mean_data, title)
            all_sigma_data = np.std(data_matrix, axis=1, ddof=1)
            title = f"{matrix_type} Sigma {index}"
            # self.plot_to_png(path, all_sigma_data, title, xtext=xtext, ytext=self.fourier_type)
            self.plot_to_csv(path, all_sigma_data, title)

    def read_channel_data(self, base_path, file_pattern, num_channels):
        channel_data = []
        for channel in range(num_channels):
            file_name = f'{file_pattern} {channel}.csv'
            full_path = f'{base_path}/{file_name}'
            channel_data.append(pd.read_csv(full_path)["Data"])
        return np.array(channel_data)

    def plot_to_png(self, path, plot, name, xlim = None, ylim = None, ytext="", xtext="", size=(12, 6)):
        logger.info(f"Plot {name}.png")
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.clf()
        plt.rcParams.update({'font.size': 14})
        f, axis = plt.subplots(1)
        f.tight_layout()
        f.set_size_inches(size[0], size[1])
        axis.grid(True)
        time = np.arange(0, len(plot), 1)
        # axis.plot(time, plot, 'o-', markersize=10)
        _, stemlines, _ = axis.stem(time, plot)
        axis.set_xlabel(xtext, loc = 'right')
        axis.set_title(ytext, loc = 'left', fontsize=14, position=(-0.06, 0))
        if xlim is not None:
            axis.axis(xmin = xlim[0], xmax = xlim[1])
        if ylim is not None:
            axis.axis(ymin = ylim[0], ymax = ylim[1])
        plt.savefig(f'{path}/{name}.png', dpi=300)

    def plot_to_csv(self, path, plot, name):
        logger.info(f"Save {name}.csv")
        path = f'{path}/CSV'
        Path(path).mkdir(parents=True, exist_ok=True)
        time = np.arange(0, len(plot), 1) / self.sampling_rate
        data = pd.DataFrame({"Time" : time, "Data" : plot})
        nk.write_csv(data, f'{path}/{name}.csv')
    
    # def getFourierSeries(self, y, fourier_type, terms = 40, L = 1):
    #     x = np.linspace(0, L, self.sampling_rate, endpoint=False)
    #     a0 = 2./L*simps(y,x)
    #     an = lambda n:2.0/L*simps(y*np.cos(2.*np.pi*n*x/L),x)
    #     bn = lambda n:2.0/L*simps(y*np.sin(2.*np.pi*n*x/L),x)
    #     list_a = np.abs([an(k) for k in range(1, terms + 1)])
    #     list_b = np.abs([bn(k) for k in range(1, terms + 1)])
    #     if fourier_type == "an":
    #         return [0, *list_a]
    #     if fourier_type == "bn":
    #         return [0, *list_b]
    #     return [0, *list_a, *list_b]
    def getFourierSeries(self, y, fourier_type, terms=40, L=1):
        x = np.linspace(0, L, self.sampling_rate, endpoint=False)
        # a0 = 2./L*simps(y, x)
        n_values = np.arange(1, terms + 1)
        cos_vals = np.cos(2. * np.pi * n_values[:, None] * x[None, :] / L)
        sin_vals = np.sin(2. * np.pi * n_values[:, None] * x[None, :] / L)

        list_a = 2.0 / L * np.abs(np.array([simps(y * cos_n, x) for cos_n in cos_vals]))
        list_b = 2.0 / L * np.abs(np.array([simps(y * sin_n, x) for sin_n in sin_vals]))

        if fourier_type == "an":
            return [0, *list_a]
        if fourier_type == "bn":
            return [0, *list_b]
        return [0, *list_a, *list_b]
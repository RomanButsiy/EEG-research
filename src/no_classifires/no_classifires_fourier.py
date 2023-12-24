from loguru import logger
import pandas as pd
import numpy as np
import neurokit2 as nk
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.integrate import simps
import time

from classification_metrics.confusion_matrix import ConfusionMatrix


class NoClassidireFourier():

    def __init__(self, eeg_config, data, fourier_type):

        self.terms = 40
        # fourier_type = "an"
        # fourier_type = "bn"
        # # fourier_type = "an_bn"

        self.eeg_config = eeg_config
        logger.debug("No Classifires Fourier")
        self.sampling_rate = data.getModSamplingRate()
        data_matrix_passivity, data_matrix_activity = data.getPreparedData()

        self.n_sigma = eeg_config.getSigma()[1]
        
        fstart = time.time()
        f_passivity_matrix = [self.getFourierSeries(m, fourier_type, terms=self.terms) for m in data_matrix_passivity[0]]
        f_activity_matrix = [self.getFourierSeries(m, fourier_type, terms=self.terms) for m in data_matrix_activity[0]]
        fend = time.time()
        self.ftime = (fend-fstart)*10**3

        self.data_matrix_passivity_1 = f_passivity_matrix
        self.data_matrix_activity_1 = f_activity_matrix
        self.confusion_matrix_names = [
            "True Positive Rate",
            "True Negative Rate", "Positive Predictive Value", "Negative Predictive Value", "False Negative Rate",
            "False Positive Rate", "False Discovery Rate", "False Omission Rate", "Positive Likelihood Ratio",
            "Negative Likelihood Ratio", "Prevalence Threshold", "Threat Score", "Accuracy", "Balanced Accuracy",
            "F1 score", "Matthews Correlation Coefficient", "Fowlkes-Mallows index", "Bookmaker Informedness", 
            "Markedness", "Diagnostic Odds Ratio", "Learning_time", "Testing_time"
        ]
        self.fourier_type = fourier_type

        self.NoAllSigma()

    def NoTest(self):
        logger.debug("No Test Sigma")
        path = f'{self.eeg_config.getImgPath()}/{self.eeg_config.getConfigBlock()}/All Mean Fourier/{self.fourier_type}'
        passivity_sigma = pd.read_csv(f'{path}/CSV/Passivity Sigma.csv')["Data"]
        activity_sigma = pd.read_csv(f'{path}/CSV/Activity Sigma.csv')["Data"]
        passivity_mean = pd.read_csv(f'{path}/CSV/Passivity Mathematical Expectation.csv')["Data"]
        activity_mean = pd.read_csv(f'{path}/CSV/Activity Mathematical Expectation.csv')["Data"]

        matrix = [*self.data_matrix_passivity_1, *self.data_matrix_activity_1]
        fa_target_matrix = [False] * len(self.data_matrix_passivity_1)
        fa_target_matrix_2 = [True] * len(self.data_matrix_activity_1)
        target_matrix = [*fa_target_matrix, *fa_target_matrix_2]

        t = 0

        res = []

        n_sigma = self.n_sigma

        lstart = time.time()
        p_upper_bound = passivity_mean + (n_sigma * passivity_sigma)
        p_lower_bound = passivity_mean - (n_sigma * passivity_sigma)
        p_lower_bound[p_lower_bound < 0] = 0

        a_upper_bound = activity_mean + (n_sigma * activity_sigma)
        a_lower_bound = activity_mean - (n_sigma * activity_sigma)
        a_lower_bound[a_lower_bound < 0] = 0
        lend = time.time()

        ltime = ((lend-lstart)*10**3) + self.ftime + self.ltime

        tstart = time.time()
        for mean, target in zip(matrix, target_matrix):

            t = t + 1 

            p_within_bounds = (mean >= p_lower_bound) & (mean <= p_upper_bound)
            a_within_bounds = (mean >= a_lower_bound) & (mean <= a_upper_bound)

            percent_p_within_bounds = np.mean(p_within_bounds) * 100
            percent_a_within_bounds = np.mean(a_within_bounds) * 100

            res.append(not(percent_p_within_bounds > percent_a_within_bounds))

        y_true = np.array(target_matrix)
        y_pred = np.array(res)
        tend = time.time()
        ttime = (tend - tstart)*10**3 + self.ftime
        confusion_matrix = ConfusionMatrix(y_true, y_pred, ltime, ttime)
        cm_all = confusion_matrix.getAllVariables()

        print(("%s: %.2f" % ("Accuracy",  confusion_matrix.getACC() * 100)))

        path_out = f'{self.eeg_config.getImgPath()}/{self.eeg_config.getConfigBlock()}/Confusion matrix'
        Path(path_out).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(np.transpose(np.round(cm_all, 2)), index=self.confusion_matrix_names, columns=[f'Fourier {self.fourier_type}'])
        df.to_csv(f'{path_out}/NoClassidireFourier {self.fourier_type}.csv')

    def NoAllSigma(self):
        logger.debug("No All Mean Fourier")

        lstart = time.time()
        data_matrix_activity = np.transpose(self.data_matrix_activity_1)
        all_mean_activity_data = np.mean(data_matrix_activity, axis=1)
        data_matrix_passivity = np.transpose(self.data_matrix_passivity_1)
        all_mean_passivity_data = np.mean(data_matrix_passivity, axis=1)

        all_sigma_activity_data = np.std(data_matrix_activity, axis=1, ddof=1)
        all_sigma_passivity_data = np.std(data_matrix_passivity, axis=1, ddof=1)
        lend = time.time()
        self.ltime = (lend-lstart)*10**3

        xtext = "$n$"
        path = f'{self.eeg_config.getImgPath()}/{self.eeg_config.getConfigBlock()}/All Mean Fourier/{self.fourier_type}'
        # self.plot_to_png(path, all_mean_activity_data, "Activity Mathematical Expectation", xtext=xtext, ytext=self.fourier_type)
        # self.plot_to_png(path, all_mean_passivity_data, "Passivity Mathematical Expectation", xtext=xtext, ytext=self.fourier_type)
        self.plot_to_csv(all_mean_activity_data, "Activity Mathematical Expectation")
        self.plot_to_csv(all_mean_passivity_data, "Passivity Mathematical Expectation")

        # self.plot_to_png(path, all_sigma_activity_data, "Activity Sigma", xtext=xtext, ytext=self.fourier_type)
        # self.plot_to_png(path, all_sigma_passivity_data, "Passivity Sigma", xtext=xtext, ytext=self.fourier_type)
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

    def plot_to_csv(self, plot, name):
        logger.info(f"Save {name}.csv")
        path = f'{self.eeg_config.getImgPath()}/{self.eeg_config.getConfigBlock()}/All Mean Fourier/{self.fourier_type}/CSV'
        Path(path).mkdir(parents=True, exist_ok=True)
        time = np.arange(0, len(plot), 1)
        data = pd.DataFrame({"Time" : time, "Data" : plot})
        nk.write_csv(data, f'{path}/{name}.csv')
    
    def getFourierSeries(self, y, fourier_type, terms = 40, L = 1):
        x = np.linspace(0, L, self.sampling_rate, endpoint=False)
        a0 = 2./L*simps(y,x)
        an = lambda n:2.0/L*simps(y*np.cos(2.*np.pi*n*x/L),x)
        bn = lambda n:2.0/L*simps(y*np.sin(2.*np.pi*n*x/L),x)
        list_a = np.abs([an(k) for k in range(1, terms + 1)])
        list_b = np.abs([bn(k) for k in range(1, terms + 1)])
        if fourier_type == "an":
            return [0, *list_a]
        if fourier_type == "bn":
            return [0, *list_b]
        return [0, *list_a, *list_b]
from my_helpers.mathematical_statistics_data import MathematicalStatisticsData
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
import neurokit2 as nk
import pandas as pd

class PlotStatistics():
    def __init__(self, statistics, sampling_rate, ecg_config, zone_type):
        self.sampling_rate = sampling_rate
        self.ecg_config = ecg_config
        self.statistics = statistics
        self.zone_type = zone_type
        self.plot_path = f'{self.ecg_config.getImgPath()}/{self.ecg_config.getConfigBlock()}'

    def plotAllStatistics(self):
        logger.info("Plot Mathematical Statistics")
        mathematical_statistics = [statistic.getMathematicalStatistics() for statistic in self.statistics]
        xtext = "$t, s$"
        self.plot_to_png([statistic.getMathematicalExpectation() for statistic in mathematical_statistics], f"1 Mathematical Expectation {self.zone_type}", xtext=xtext, ytext=r"$m_{{\xi {}}} (t), \mu V$", ylim=(-4.5, 4.5))
        self.plot_to_png([statistic.getInitialMomentsSecondOrder() for statistic in mathematical_statistics], f"2 Initial Moments Second Order {self.zone_type}", xtext=xtext, ytext=r"$d_{{\xi {}}} (t), \mu V^2$")
        self.plot_to_png([statistic.getInitialMomentsThirdOrder() for statistic in mathematical_statistics], f"3 Initial Moments Third Order {self.zone_type}", xtext=xtext, ytext=r"$d_{{\xi {}}} (t), \mu V^3$")
        self.plot_to_png([statistic.getInitialMomentsFourthOrder() for statistic in mathematical_statistics], f"4 Initial Moments Fourth Order {self.zone_type}", xtext=xtext, ytext=r"$d_{{\xi {}}} (t), \mu V^4$")
        self.plot_to_png([statistic.getVariance() for statistic in mathematical_statistics], f"5 Variance {self.zone_type}", xtext=xtext, ytext=r"$d_{{\xi {}}} (t), \mu V^2$")
        self.plot_to_png([statistic.getCentralMomentFunctionsFourthOrder() for statistic in mathematical_statistics], f"6 Central Moment Functions Fourth Order {self.zone_type}",  xtext=xtext, ytext=r"$d_{{\xi {}}} (t), \mu V^4$")
 
        self.plot_to_csv([statistic.getMathematicalExpectation() for statistic in mathematical_statistics], f"1 Mathematical Expectation {self.zone_type}")
        self.plot_to_csv([statistic.getInitialMomentsSecondOrder() for statistic in mathematical_statistics], f"2 Initial Moments Second Order {self.zone_type}")
        self.plot_to_csv([statistic.getInitialMomentsThirdOrder() for statistic in mathematical_statistics], f"3 Initial Moments Third Order {self.zone_type}")
        self.plot_to_csv([statistic.getInitialMomentsFourthOrder() for statistic in mathematical_statistics], f"4 Initial Moments Fourth Order {self.zone_type}")
        # self.plot_to_csv(mathematical_statistics.getVariance(), "5 Variance")
        # self.plot_to_csv(mathematical_statistics.getCentralMomentFunctionsFourthOrder(), "6 Central Moment Functions Fourth Order")

    def plotAllFourierStatistics(self):
        logger.info("Plot Mathematical Statistics Fourier")
        self.statistics = [statistic.setSamplingRate(self.sampling_rate) or statistic for statistic in self.statistics]
        mathematical_statistics = [statistic.getMathematicalStatisticsFourierSeries() for statistic in self.statistics]
        
        xtext = "$n$"
        self.fs_plot_to_png([statistic.getMathematicalExpectation() for statistic in mathematical_statistics], f"1 Mathematical Expectation {self.zone_type}", xtext=xtext, ytext=(r"$a_n, \mu V$", r"$b_n, \mu V$"))
        self.fs_plot_to_png([statistic.getInitialMomentsSecondOrder() for statistic in mathematical_statistics], f"2 Initial Moments Second Order {self.zone_type}", xtext=xtext, ytext=(r"$a_n, \mu V^2$", r"$b_n, \mu V^2$"))
        self.fs_plot_to_png([statistic.getInitialMomentsThirdOrder() for statistic in mathematical_statistics], f"3 Initial Moments Third Order {self.zone_type}", xtext=xtext, ytext=(r"$a_n, \mu V^3$", r"$b_n, \mu V^3$"))
        self.fs_plot_to_png([statistic.getInitialMomentsFourthOrder() for statistic in mathematical_statistics], f"4 Initial Moments Fourth Order {self.zone_type}", xtext=xtext, ytext=(r"$a_n, \mu V^4$", r"$b_n, \mu V^4$"))
        self.fs_plot_to_png([statistic.getVariance() for statistic in mathematical_statistics], f"5 Variance {self.zone_type}", xtext=xtext, ytext=(r"$a_n, \mu V^2$", r"$b_n, \mu V^2$"))
        self.fs_plot_to_png([statistic.getCentralMomentFunctionsFourthOrder() for statistic in mathematical_statistics], f"6 Central Moment Functions Fourth Order {self.zone_type}",  xtext=xtext, ytext=(r"$a_n, \mu V^4$", r"$b_n, \mu V^4$"))
 
        # self.fs_plot_to_csv(mathematical_statistics.getMathematicalExpectation(), "1 Mathematical Expectation")
        # self.fs_plot_to_csv(mathematical_statistics.getInitialMomentsSecondOrder(), "2 Initial Moments Second Order")
        # self.fs_plot_to_csv(mathematical_statistics.getInitialMomentsThirdOrder(), "3 Initial Moments Third Order")
        # self.fs_plot_to_csv(mathematical_statistics.getInitialMomentsFourthOrder(), "4 Initial Moments Fourth Order")
        # self.fs_plot_to_csv(mathematical_statistics.getVariance(), "5 Variance")
        # self.fs_plot_to_csv(mathematical_statistics.getCentralMomentFunctionsFourthOrder(), "6 Central Moment Functions Fourth Order")

    def fs_plot_to_png(self, plot2, name, xlim = None, ylim = None, ytext=(r"$a_n, \mu V$", r"$b_n, \mu V$"), xtext="", size=(9, 9)):
        path = f'{self.plot_path}/Mathematical Statistics Fourier'
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Plot {name} an.png")
        plt.clf()
        plt.rcParams.update({'font.size': 14})
        f, axis = plt.subplots(len(plot2))
        f.tight_layout()
        f.set_size_inches(size)
        for i in range(len(plot2)):
            an, _ = plot2[i]
            axis[i].set_xlabel(xtext, loc = 'right')
            axis[i].set_title(ytext[0], loc = 'left', fontsize=15, position=(-0.05, 0))
            axis[i].grid(True)
            _, stemlines, _ = axis[i].stem([0, *an[1:]])
            plt.setp(stemlines, 'linewidth', 2)
        plt.savefig(f'{path}/{name} an.png', dpi=300)

        logger.info(f"Plot {name} bn.png")
        plt.clf()
        plt.rcParams.update({'font.size': 14})
        f, axis = plt.subplots(len(plot2))
        f.tight_layout()
        f.set_size_inches(size)
        for i in range(len(plot2)):
            _, bn = plot2[i]
            axis[i].set_xlabel(xtext, loc = 'right')
            axis[i].set_title(ytext[1], loc = 'left', fontsize=15, position=(-0.05, 0))
            axis[i].grid(True)
            _, stemlines, _ = axis[i].stem([0, *bn])
            plt.setp(stemlines, 'linewidth', 2)
        plt.savefig(f'{path}/{name} bn.png', dpi=300)

    def plot_to_png(self, plot, name, xlim = None, ylim = None, ytext="", xtext="", size=(7, 9)):
        logger.info(f"Plot {name}.png")
        path = f'{self.plot_path}/Mathematical Statistics'
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.clf()
        # plt.rcParams.update({'font.size': 14})
        f, axis = plt.subplots(len(plot))
        f.tight_layout()
        f.set_size_inches(size[0], size[1])
        time = np.arange(0, len(plot[0]), 1) / self.sampling_rate
        for i in range(len(plot)):
            axis[i].grid(True)
            axis[i].plot(time, plot[i], linewidth=2)
            axis[i].set_xlabel(xtext, loc = 'right')
            axis[i].set_title(ytext.format(i+1), loc = 'left', fontsize=10, position=(-0.07, 0))
            if xlim is not None:
                axis[i].axis(xmin = xlim[0], xmax = xlim[1])
            if ylim is not None:
                axis[i].axis(ymin = ylim[0], ymax = ylim[1])
        plt.savefig(f'{path}/{name}.png', dpi=300)

    def plot_to_csv(self, plot, name):
        logger.info(f"Save {name}.csv")
        path = f'{self.plot_path}/Mathematical Statistics/CSV'
        Path(path).mkdir(parents=True, exist_ok=True)
        plot = np.transpose(plot)
        time = np.arange(0, len(plot), 1) / self.sampling_rate
        data = pd.DataFrame({'Time': time, **{f'Data_{i}': plot[:, i] for i in range(plot.shape[1])}})
        nk.write_csv(data, f'{path}/{name}.csv')

    # def fs_plot_to_csv(self, plot2, name):
    #     an, bn = plot2
    #     logger.info(f"Save {name}.csv")
    #     path = f'{self.plot_path}/Mathematical Statistics Fourier/CSV'
    #     Path(path).mkdir(parents=True, exist_ok=True)
    #     time = np.arange(0, len(an), 1)
    #     data = pd.DataFrame({"n" : time, "an" : an, "bn" : [0, *bn]})
    #     nk.write_csv(data, f'{path}/{name}.csv')

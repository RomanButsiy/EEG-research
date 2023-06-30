from get_config.eeg_config import EEGConfig
from loguru import logger
import argparse
from my_helpers.classifiers import Classifiers
from my_helpers.data_preparation import DataPreparation
from my_helpers.mathematical_statistics import MathematicalStatistics
from my_helpers.plot_statistics import PlotStatistics

from my_helpers.rhythm_function import RhythmFunction

if __name__ == '__main__':
    logger.info('START of execution')
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('plot-fr', 'plot-eeg', 'plot-statistics', 'plot-fourier-statistics', 'train-classifier'))
    parser.add_argument('-c', type=str, required=True)
    parser.add_argument('-d', type=str)
    parser.add_argument('-s', type=str)
    a = parser.parse_args()
    config_block = a.c
    logger.debug("Read config file")
    eeg_config = EEGConfig(config_block)
    logger.debug(eeg_config)
    if a.action == "plot-fr":
        RhythmFunction(eeg_config).plotFr()
    if a.action == "plot-eeg":
        RhythmFunction(eeg_config).plotEEG()
    if a.action == "plot-statistics":
        data = DataPreparation(eeg_config)
        statistics_passivity, statistics_activity = [], []
        for passivity, activity in zip(*data.getPreparedData()):
            statistics_passivity.append(MathematicalStatistics(passivity))
            statistics_activity.append(MathematicalStatistics(activity))
        PlotStatistics(statistics_passivity, data.getModSamplingRate(), eeg_config, "passivity").plotAllStatistics()
        PlotStatistics(statistics_activity, data.getModSamplingRate(), eeg_config, "activity").plotAllStatistics()

    if a.action == "plot-fourier-statistics":
        data = DataPreparation(eeg_config)
        statistics_passivity, statistics_activity = [], []
        for passivity, activity in zip(*data.getPreparedData()):
            statistics_passivity.append(MathematicalStatistics(passivity))
            statistics_activity.append(MathematicalStatistics(activity))
        PlotStatistics(statistics_passivity, data.getModSamplingRate(), eeg_config, "passivity").plotAllFourierStatistics()
        PlotStatistics(statistics_activity, data.getModSamplingRate(), eeg_config, "activity").plotAllFourierStatistics()

    if a.action == "train-classifier":
        data = DataPreparation(eeg_config)
        Classifiers(eeg_config, data)

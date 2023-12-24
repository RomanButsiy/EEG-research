from diff.diff_confusion_matrics import DiffConfusionMatrix
from get_config.eeg_config import EEGConfig
from loguru import logger
import argparse
from my_helpers.classifiers import Classifiers
from my_helpers.data_preparation import DataPreparation
from my_helpers.mathematical_statistics import MathematicalStatistics
from my_helpers.plot_classifiers import PlotClassifiers
from my_helpers.plot_statistics import PlotStatistics

from my_helpers.rhythm_function import RhythmFunction
from no_classifires.no_classifires import NoClassidire
from no_classifires.no_classifires_all_chanels import NoClassidireAllChanels
from no_classifires.no_classifires_fourier import NoClassidireFourier
from no_classifires.no_classifires_fourier_all_chanels import NoClassidireFourierAllChanels
from statistics_distance.statistics_distance import StatisticsDistance
from statistics_distance.statistics_distance_fourier import StatisticsDistanceFourier

if __name__ == '__main__':
    logger.info('START of execution')
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('diff-confusion-matrix', 'statistics-distance-f', 'statistics-distance', 'no-all-test-fd', 'no-all-sigma-fd', 'no-all-test-d', 'no-all-sigma-d', 'no-all-test-f', 'no-all-sigma-f', 'no-all-mean-f', 'no-all-test', 'no-all-sigma', 'plot-fr', 'plot-eeg', 'plot-statistics', 'plot-fourier-statistics', 'train-classifier', 'plot-classifier'))
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
        for fourier_type in ["an", "bn", "an_bn"]:
            Classifiers(eeg_config, data, fourier_type)
    if a.action == "plot-classifier":
        # data = DataPreparation(eeg_config)
        PlotClassifiers(eeg_config, None)
    if a.action == "no-all-test":
        data = DataPreparation(eeg_config)
        NoClassidire(eeg_config, data).NoTest()
    if a.action == "no-all-sigma":
        data = DataPreparation(eeg_config)
        NoClassidire(eeg_config, data).NoAllSigma()
    if a.action == "no-all-test-f":
        data = DataPreparation(eeg_config)
        for fourier_type in ["an", "bn", "an_bn"]:
            NoClassidireFourier(eeg_config, data, fourier_type).NoTest()
    if a.action == "no-all-sigma-f":
        data = DataPreparation(eeg_config)
        NoClassidireFourier(eeg_config, data).NoAllSigma()
    if a.action == "no-all-test-d":
        data = DataPreparation(eeg_config)
        NoClassidireAllChanels(eeg_config, data).NoTest()
    if a.action == "no-all-sigma-d":
        data = DataPreparation(eeg_config)
        NoClassidireAllChanels(eeg_config, data).NoAllSigma()

    if a.action == "no-all-test-fd":
        data = DataPreparation(eeg_config)
        # fourier_type = "an"
        # fourier_type = "bn"
        # fourier_type = "an_bn"
        for fourier_type in ["an", "bn", "an_bn"]:
            for terms in [3, 4, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 200, 250]:
                NoClassidireFourierAllChanels(eeg_config, data, fourier_type, terms).NoTest()
    if a.action == "no-all-sigma-fd":
        data = DataPreparation(eeg_config)
        NoClassidireFourierAllChanels(eeg_config, data).NoAllSigma()

    if a.action == "statistics-distance":
        data = DataPreparation(eeg_config)
        StatisticsDistance(eeg_config, data).Calculate()
    if a.action == "statistics-distance-f":
        data = DataPreparation(eeg_config)
        for fourier_type in ["an", "bn", "an_bn"]:
            for terms in [3, 4, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 200, 250]:
                StatisticsDistanceFourier(eeg_config, data, fourier_type, terms).Calculate()
    if a.action == "diff-confusion-matrix":
        for fourier_type in ["an", "bn", "an_bn"]:
            for terms in [3, 4, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 200, 250]:
                DiffConfusionMatrix(eeg_config, None, fourier_type, terms).DiffConfusionMatrix()
    

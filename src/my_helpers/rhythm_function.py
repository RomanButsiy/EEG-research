from loguru import logger
from my_helpers.read_data.read_data_file import ReadDataFile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class RhythmFunction(ReadDataFile):

    def __init__(self, ecg_config):
        super().__init__(ecg_config)
        self.plot_path = f'{self.eeg_config.getImgPath()}/{self.eeg_config.getConfigBlock()}'
        Path(self.plot_path).mkdir(parents=True, exist_ok=True)

    def plotFr(self):
        logger.info("Plot a rhythm function")

        self.getData()

        T1_D_c = []
        T1_D_z = []
        T1_X = []
        T1_Y = []

        for i in range(len(self.D_c) - 1):
            T1_D_c.append(round(self.D_c[i+1] - self.D_c[i], 2))

        for i in range(len(self.D_z) - 1):
            T1_D_z.append(round(self.D_z[i+1] - self.D_z[i], 2))

        for i in range(len(self.D_c) - 1):
            T1_X.append(self.D_c[i])
            T1_X.append(self.D_z[i])

        for i in range(len(T1_D_c)):
            T1_Y.append(T1_D_c[i])
            T1_Y.append(T1_D_z[i])

        plt.clf()
        plt.rcParams.update({'font.size': 14})
        f, axis = plt.subplots(1)
        f.tight_layout()
        f.set_size_inches(10, 6)
        axis.grid(True)
        axis.plot(T1_X, T1_Y, linewidth=2)
        axis.set_xlabel("$t, s$", loc = 'right')
        axis.legend(['$T(t, 1), s$'])
        axis.axis(ymin = 0, ymax = 5)
        axis.axis(xmin = 0)
        plt.savefig(f"{self.plot_path}/function_rhythm.png", dpi=300)

    def plotEEG(self):
        logger.info("Plot a EEG")

        self.getData()

        f, axis = plt.subplots(len(self.signals))
        f.tight_layout()
        f.set_size_inches(17, 9)
        time = np.arange(0, len(self.signals[0]), 1) / 250

        xtext = "$t, s$"
        ytext=r"$\xi_{{\omega {}}} (t), \mu V$"

        for i in range(len(self.signals)):
            axis[i].grid(True)
            axis[i].plot(time, self.signals[i])
            axis[i].set_xlabel(xtext, loc = 'right')
            axis[i].set_title(ytext.format(i+1), loc = 'left', fontsize=10, position=(-0.07, 0))

        for i in range(len(self.signals)):
            axis[i].axis(xmin = 7.2, xmax = 20)
            axis[i].axis(ymin = -10, ymax = 10)
        plt.savefig(f"{self.plot_path}/Графік-1-17.png", dpi=300)



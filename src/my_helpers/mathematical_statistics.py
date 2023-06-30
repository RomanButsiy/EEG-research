from my_helpers.mathematical_statistics_data import MathematicalStatisticsData
import numpy as np
from scipy.integrate import simps

class MathematicalStatistics(MathematicalStatisticsData):
    def __init__(self, data):
        data = np.transpose(data)
        #Mathematical expectation
        self.m_ = [np.mean(i) for i in data]
        #Initial moments of the second order
        self.m_2_ = [np.sum(np.array(i)**2) / len(i) for i in data]
        #Initial moments of the third order
        self.m_3_ = [np.sum(np.array(i)**3) / len(i) for i in data]
        #Initial moments of the fourth order
        self.m_4_ = [np.sum(np.array(i)**4) / len(i) for i in data]
        #Variance
        self.m__2 = [sum((data[i] - self.m_[i])**2) / len(data[i]) for i in range(len(self.m_))]
        #Central moment functions of the fourth order
        self.m__4 = [sum((data[i] - self.m_[i])**4) / len(data[i]) for i in range(len(self.m_))]

    def getMathematicalStatistics(self):
        return MathematicalStatisticsData(self.m_, self.m_2_, self.m_3_, self.m_4_, self.m__2, self.m__4)
    
    def getMathematicalStatisticsFourierSeries(self):
        m_f = self.getFourierSeries(self.m_)
        m_2_f = self.getFourierSeries(self.m_2_)
        m_3_f = self.getFourierSeries(self.m_3_)
        m_4_f = self.getFourierSeries(self.m_4_)
        m__2_f = self.getFourierSeries(self.m__2)
        m__4_f = self.getFourierSeries(self.m__4)
        return MathematicalStatisticsData(m_f, m_2_f, m_3_f, m_4_f, m__2_f, m__4_f)

    def getFourierSeries(self, y, terms = 50, L = 1):
        x = np.linspace(0, L, self.sampling_rate, endpoint=False)
        a0 = 2./L*simps(y,x)
        an = lambda n:2.0/L*simps(y*np.cos(2.*np.pi*n*x/L),x)
        bn = lambda n:2.0/L*simps(y*np.sin(2.*np.pi*n*x/L),x)
        list_a = np.abs([a0, *[an(k) for k in range(1, terms + 1)]])
        list_b = np.abs([bn(k) for k in range(1, terms + 1)])
        return list_a, list_b
    
    def setSamplingRate(self, sampling_rate):
        self.sampling_rate = sampling_rate
        
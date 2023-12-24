import numpy as np
import time
from scipy.integrate import simps
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from pathlib import Path

from classification_metrics.confusion_matrix import ConfusionMatrix

class Classifiers():
        
    def __init__(self, eeg_config, data, fourier_type):
        print("Train Classifiers")

        # fourier_type = "an"
        # fourier_type = "bn"
        # fourier_type = "an_bn"

        names = [
            "Nearest Neighbors",
            "Linear SVM",
            "Decision Tree",
            "Random Forest",
            "Neural Net (MLP)",
            "AdaBoost",
            "Naive Bayes"
        ]

        confusion_matrix_names = [
            "True Positive Rate",
            "True Negative Rate", "Positive Predictive Value", "Negative Predictive Value", "False Negative Rate",
            "False Positive Rate", "False Discovery Rate", "False Omission Rate", "Positive Likelihood Ratio",
            "Negative Likelihood Ratio", "Prevalence Threshold", "Threat Score", "Accuracy", "Balanced Accuracy",
            "F1 score", "Matthews Correlation Coefficient", "Fowlkes-Mallows index", "Bookmaker Informedness", 
            "Markedness", "Diagnostic Odds Ratio", "Learning_time", "Testing_time"
        ]

        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1, max_iter=1000),
            AdaBoostClassifier(),
            GaussianNB()
        ]

        self.sampling_rate = data.getModSamplingRate()

        data_matrix_passivity, data_matrix_activity = data.getPreparedData()
        data_matrix_passivity_1 = data_matrix_passivity[0]
        data_matrix_activity_1 = data_matrix_activity[0]

        # for i in [999]:

        for i in [3, 4, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 200, 250, 999]:

            fa_target_matrix = [0] * len(data_matrix_passivity_1)
            fa_target_matrix_2 = [1] * len(data_matrix_activity_1)
            if i == 999:
                matrix = [*data_matrix_passivity_1, *data_matrix_activity_1]
            else:    
                fa_matrix = [self.getFourierSeries(m, fourier_type, terms=i) for m in data_matrix_passivity_1]
                fa_matrix_2 = [self.getFourierSeries(m, fourier_type, terms=i) for m in data_matrix_activity_1]
                matrix = [*fa_matrix, *fa_matrix_2]
            
    
            target_matrix = [*fa_target_matrix, *fa_target_matrix_2]

            data_train, data_test, target_values_train, target_values_test = train_test_split(matrix, target_matrix, test_size=0.3, random_state=42)

            _cm = []

            for name, clf in zip(names, classifiers):
                clf = make_pipeline(StandardScaler(), clf)
                lstart = time.time()
                clf.fit(data_train, target_values_train)
                lend = tstart = time.time()
                y_true = np.array(target_values_test)
                y_pred = clf.predict(data_test)
                tend = time.time()
                ltime = (lend-lstart)*10**3
                ttime = (tend-tstart)*10**3
    
                confusion_matrix = ConfusionMatrix(y_true, y_pred, ltime, ttime)
                res = confusion_matrix.getAllVariables()

                _cm.append(res)

                print(("%s: %.2f" % (name,  confusion_matrix.getACC() * 100)))

            path = f'{eeg_config.getImgPath()}/{eeg_config.getConfigBlock()}/Confusion matrix/{fourier_type}'
            Path(path).mkdir(parents=True, exist_ok=True)
            
            df = pd.DataFrame(np.transpose(np.round(_cm, 2)), index=confusion_matrix_names, columns=names)
            df.to_csv(f'{path}/n-{i}.csv')
        


    def getFourierSeries(self, y, fourier_type, terms=40, L=1):
        x = np.linspace(0, L, self.sampling_rate, endpoint=False)
        a0 = 2./L*simps(y, x)
        n_values = np.arange(1, terms + 1)
        cos_vals = np.cos(2. * np.pi * n_values[:, None] * x[None, :] / L)
        sin_vals = np.sin(2. * np.pi * n_values[:, None] * x[None, :] / L)

        list_a = 2.0 / L * np.abs(np.array([simps(y * cos_n, x) for cos_n in cos_vals]))
        list_b = 2.0 / L * np.abs(np.array([simps(y * sin_n, x) for sin_n in sin_vals]))

        if fourier_type == "an":
            return [a0, *list_a]
        if fourier_type == "bn":
            return [0, *list_b]
        return [0, *list_a, *list_b]
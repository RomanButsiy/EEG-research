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
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd
from pathlib import Path

from classification_metrics.confusion_matrix import ConfusionMatrix

class Classifiers():
        
    def __init__(self, ecg_config, data):
        print("Train Classifiers")

        names = [
            "Nearest Neighbors",
            "Linear SVM",
            "Gaussian Process",
            "Decision Tree",
            "Random Forest",
            "Neural Net (MLP)",
            "AdaBoost",
            "Naive Bayes",
            "QDA"
        ]

        confusion_matrix_names = [
            "True Positives", "True Negative", "False Positive", "False Negative", "True Positive Rate",
            "True Negative Rate", "Positive Predictive Value", "Negative Predictive Value", "False Negative Rate",
            "False Positive Rate", "False Discovery Rate", "False Omission Rate", "Positive Likelihood Ratio",
            "Negative Likelihood Ratio", "Prevalence Threshold", "Threat Score", "Accuracy", "Balanced Accuracy",
            "F1 score", "Matthews Correlation Coefficient", "Fowlkes-Mallows index", "Bookmaker Informedness", 
            "Markedness", "Diagnostic Odds Ratio"
        ]

        classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1, max_iter=1000),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()
        ]
        
        l1 = []
        t1 = []
        t2 = []

        self.sampling_rate = data.getModSamplingRate()

        data_matrix_passivity, data_matrix_activity = data.getPreparedData()
        data_matrix_passivity_1 = data_matrix_passivity[0]
        data_matrix_activity_1 = data_matrix_activity[0]

        # noise = (np.random.rand(len(data_matrix), len(data_matrix[0])) - 0.5 ) / 10
        # noise_2 = (np.random.rand(len(data_matrix_2), len(data_matrix_2[0])) - 0.5 ) / 10

        # data_matrix = data_matrix + noise
        # data_matrix_2 = data_matrix_2 + noise_2

        # # 15, 20, 25, 30, 35, 40, 45, 50, 55, 60
        # # 3, 4, 5, 7, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60

        for i in [30]:

            fa_matrix = [self.getFourierSeriesA(m, terms=i) for m in data_matrix_passivity_1]
            fa_target_matrix = [0] * len(fa_matrix)
            fa_matrix_2 = [self.getFourierSeriesA(m, terms=i) for m in data_matrix_activity_1]
            fa_target_matrix_2 = [1] * len(fa_matrix_2)
    
            matrix = [*fa_matrix, *fa_matrix_2]
            target_matrix = [*fa_target_matrix, *fa_target_matrix_2]
    
            # matrix = [*data_matrix_passivity_1, *data_matrix_activity_1]       
    
            data_train, data_test, target_values_train, target_values_test = train_test_split(matrix, target_matrix, test_size=0.3, random_state=42)
            _l1 = []
            _t1 = []
            _t2 = []
            _cm = []

            for name, clf in zip(names, classifiers):
                clf = make_pipeline(StandardScaler(), clf)
                lstart = time.time()
                clf.fit(data_train, target_values_train)

                y_true = np.array(target_values_test)
                y_pred = clf.predict(data_test)

                confusion_matrix = ConfusionMatrix(y_true, y_pred)

                res = confusion_matrix.getAllVariables()

                _cm.append(res)

                lend = tstart = time.time()
                score = clf.score(data_test, target_values_test)
                
                
                tend = time.time()
    
                _l1.append(score * 100)
                _t1.append((lend-lstart)*10**3)
                _t2.append((tend-tstart)*10**3)
                print(("%s: %.2f" % (name, score * 100)))
                # print(f'Learning time: {(lend-lstart)*10**3:.03f} ms | Testing time: {(tend-tstart)*10**3:.03f} ms')

            # print(_cm)

            path = f'{ecg_config.getImgPath()}/{ecg_config.getConfigBlock()}/Confusion matrix'
            Path(path).mkdir(parents=True, exist_ok=True)
            
            df = pd.DataFrame(np.transpose(np.round(_cm, 4)), index=confusion_matrix_names, columns=names)
            df.to_csv(f'{path}/Test.csv')

            l1.append(_l1)
            t1.append(_t1)
            t2.append(_t2)

        # plot_path = f'{ecg_config.getImgPath()}/{data.getSigNameDir()}'
        # Path(f'{plot_path}/Classifiers').mkdir(parents=True, exist_ok=True)

        # l = pd.DataFrame({"l1" : l1, "t1" : t1, "t2" : t2})
        # nk.write_csv(l, f'{plot_path}/Classifiers/L1.csv')

        # plt.clf()
        # plt.rcParams.update({'font.size': 14})
        # f, axis = plt.subplots(1)
        # f.tight_layout()
        # f.set_size_inches(19, 6)
        # axis.grid(True)
        # axis.set_xlabel("$t, s$", loc = 'right')
        # for i in np.transpose(l1):
        #     axis.plot([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60], i, linewidth=2)
        # axis.legend(loc='best',prop={'size':10})
        # Path(f'{plot_path}/Classifiers').mkdir(parents=True, exist_ok=True)
        # plt.savefig(f'{plot_path}/Classifiers/L1.png', dpi=300)
        


    def getFourierSeriesA(self, y, terms = 30, L = 1):
        x = np.linspace(0, L, self.sampling_rate, endpoint=False)
        a0 = 2./L*simps(y,x)
        an = lambda n:2.0/L*simps(y*np.cos(2.*np.pi*n*x/L),x)
        list_a = np.abs([*[an(k) for k in range(1, terms + 1)]])
        return list_a
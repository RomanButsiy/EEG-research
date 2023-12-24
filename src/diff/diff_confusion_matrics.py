from loguru import logger
import pandas as pd
import numpy as np
from pathlib import Path
import diff.operators as used_operators

class DiffConfusionMatrix():

    def __init__(self, eeg_config, data, fourier_type, terms):
        self.eeg_config = eeg_config
        self.terms = terms
        self.fourier_type = fourier_type
        logger.debug("Diff Confusion Matrix")

        self.names = [
            "Nearest Neighbors",
            "Linear SVM",
            "Decision Tree",
            "Random Forest",
            "Neural Net (MLP)",
            "AdaBoost",
            "Gaussian Naive Bayes",
            "SIC",
            "SPC"
        ]

        self.confusion_matrix_names = [
            "True Positive Rate",
            "True Negative Rate", "Positive Predictive Value", "Negative Predictive Value", "False Negative Rate",
            "False Positive Rate", "False Discovery Rate", "False Omission Rate", "Positive Likelihood Ratio",
            "Negative Likelihood Ratio", "Prevalence Threshold", "Threat Score", "Accuracy", "Balanced Accuracy",
            "F1 score", "Matthews Correlation Coefficient", "Fowlkes-Mallows index", "Bookmaker Informedness", 
            "Markedness", "Diagnostic Odds Ratio", "Learning_time", "Testing_time"
        ]

    def DiffConfusionMatrix(self):
        logger.debug("DiffConfusionMatrix")
        selected_array = used_operators.arrays.get(self.eeg_config.getConfigBlock(), [])


        all_data = []
        for operator in selected_array:
            read_path = f'{self.eeg_config.getImgPath()}/{operator}/Confusion matrix/{self.fourier_type}'
            df = pd.read_csv(f'{read_path}/n-{self.terms}.csv')
            if 'Unnamed: 0' in df.columns:
                df = df.drop('Unnamed: 0', axis=1)
            all_data.append(df)
        res = np.round(np.mean(all_data, axis=0), 2)
        df = pd.DataFrame(res, index=self.confusion_matrix_names, columns=self.names)
        path = f'{self.eeg_config.getImgPath()}/{self.eeg_config.getConfigBlock()}/Confusion matrix/{self.fourier_type}'
        Path(path).mkdir(parents=True, exist_ok=True)
        df.to_csv(f'{path}/n-{self.terms}.csv')
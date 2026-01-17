import numpy as np
import joblib

"""
Filename: get_species.py
Author: Mili Ostojic
Date: 09-01-2026
Version: 1.0
Description: Predict the species from the features of a flight pattern
"""


def random_forest(features):
    """Predict species from the features of a flight pattern"""

    # Load Model 3
    rf = joblib.load("model3_best.joblib")

    # Predict species
    predicted_label = rf.predict(np.array([features]))

    return predicted_label

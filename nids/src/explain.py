"""
explain.py
Wraps SHAP's TreeExplainer for the trained RandomForest so the dashboard
can show "why" a specific connection was flagged, not just the label.
"""

import shap
import numpy as np
import pandas as pd


class Explainer:
    def __init__(self, model, feature_columns):
        self.model = model
        self.feature_columns = feature_columns
        self.explainer = shap.TreeExplainer(model)

    def explain_row(self, row: pd.DataFrame, predicted_class_idx: int, top_n: int = 8):
        """
        Returns the top_n features (name, shap_value) driving the prediction
        for a single row, for the specific predicted class.
        """
        shap_values = self.explainer.shap_values(row, check_additivity=False)

        # shap_values shape handling: list-per-class (older SHAP) vs 3D array (newer SHAP)
        if isinstance(shap_values, list):
            class_values = shap_values[predicted_class_idx][0]
        else:
            arr = np.array(shap_values)
            if arr.ndim == 3:
                class_values = arr[0, :, predicted_class_idx]
            else:
                class_values = arr[0]

        pairs = list(zip(self.feature_columns, class_values))
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)
        return pairs[:top_n]

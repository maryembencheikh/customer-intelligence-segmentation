import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class CustomerFeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.frequency_map = {
            "Weekly": 52,
            "Bi-Weekly": 26,
            "Fortnightly": 26,
            "Monthly": 12,
            "Quarterly": 4,
            "Every 3 Months": 4,
            "Annually": 1
        }

        self.final_features = [
            "Age",
            "Purchase Amount (USD)",
            "Review Rating",
            "Previous Purchases",
            "Purchase Frequency",
            "Is_Subscribed",
            "Log_Annual_Spend",
            "Log_Purchase_Engagement"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        features = pd.DataFrame({
            "Age": X["Age"],
            "Purchase Amount (USD)": X["Purchase Amount (USD)"],
            "Review Rating": X["Review Rating"],
            "Previous Purchases": X["Previous Purchases"],

            "Purchase Frequency": X["Frequency of Purchases"].map(
                self.frequency_map
            ),

            "Is_Subscribed": X["Subscription Status"].map({
                "Yes": 1,
                "No": 0
            })
        })

        features["Estimated_Annual_Spend"] = (
            features["Purchase Amount (USD)"]
            * features["Purchase Frequency"]
        )

        features["Purchase_Engagement"] = (
            features["Previous Purchases"]
            * features["Purchase Frequency"]
        )

        features["Log_Annual_Spend"] = np.log1p(
            features["Estimated_Annual_Spend"]
        )

        features["Log_Purchase_Engagement"] = np.log1p(
            features["Purchase_Engagement"]
        )
        return features[self.final_features]
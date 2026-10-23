import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import os

MODEL_PATH = "src/model.pkl"


class EnergyForecaster:

    def __init__(self):
        self.model = None

    def train_model(self, df):

        df = df.copy()

        if len(df) < 5:
            return None

        df["Index"] = np.arange(len(df))

        X = df[["Index"]]
        y = df["Power"]

        model = LinearRegression()
        model.fit(X, y)

        joblib.dump(model, MODEL_PATH)

        self.model = model

        return model

    def load_model(self):

        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            return self.model
        return None

    def predict_future(self, df, steps=10):

        if self.model is None:
            self.load_model()

        if self.model is None:
            return []

        last_index = len(df)

        future_X = np.array(
            [[i] for i in range(last_index, last_index + steps)]
        )

        predictions = self.model.predict(future_X)

        return predictions
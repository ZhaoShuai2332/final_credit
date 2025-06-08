import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

class Linear:
    def __init__(self, name: str,
                 X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray,  y_test: np.ndarray):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.params_path = os.path.join(base_dir, "params", f"{name}_linear_params.npz")
        self.scaler_path = os.path.join(base_dir, "params", f"{name}_min_max_scaler_params.npz")

        # Raw data
        self.X_train, self.y_train = X_train, y_train
        self.X_test,  self.y_test  = X_test,  y_test

        self.pipeline = Pipeline([
            ("scaler", MinMaxScaler()),  # Only scale X
            ("regressor", LinearRegression())
        ])

        self.loaded = False
        if os.path.exists(self.params_path) and os.path.exists(self.scaler_path):
            print(f"Loading model params from {self.params_path}")
            params = np.load(self.params_path)
            reg = self.pipeline.named_steps["regressor"]
            reg.coef_ = params["coef"]
            reg.intercept_ = params["intercept"]

            print(f"Loading scaler params from {self.scaler_path}")
            scaler_params = np.load(self.scaler_path)
            scaler = self.pipeline.named_steps["scaler"]
            scaler.min_ = scaler_params["min"]
            scaler.scale_ = scaler_params["scale"]
            scaler.data_min_ = scaler_params["data_min"]
            scaler.data_max_ = scaler_params["data_max"]
            scaler.data_range_ = scaler.data_max_ - scaler.data_min_
            scaler.n_samples_seen_ = self.X_train.shape[0]

            self.loaded = True
        else:
            print(f"No pre-saved params found for X; will fit from scratch.")

    def fit(self):
        if self.loaded:
            print("Parameters already loaded; skipping fit.")
            return

        self.pipeline.fit(self.X_train, self.y_train)

        reg = self.pipeline.named_steps["regressor"]
        np.savez(self.params_path,
                 coef=reg.coef_,
                 intercept=reg.intercept_)

        scaler = self.pipeline.named_steps["scaler"]
        np.savez(self.scaler_path,
                 min=scaler.min_,
                 scale=scaler.scale_,
                 data_min=scaler.data_min_,
                 data_max=scaler.data_max_)

    def predict(self) -> np.ndarray:
        return self.pipeline.predict(self.X_test)

    def get_params(self):
        reg = self.pipeline.named_steps["regressor"]
        return reg.coef_, reg.intercept_

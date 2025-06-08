import os
import numpy as np
from sklearn.linear_model import LinearRegression


class Linear:
    def __init__(self, name: str,
                 X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray,  y_test: np.ndarray):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.params_path = os.path.join(base_dir, "params", f"{name}_linear_params.npz")

        self.X_train, self.y_train = X_train, y_train
        self.X_test,  self.y_test  = X_test,  y_test

        self.model = LinearRegression()

        self.loaded = False
        if os.path.exists(self.params_path):
            print(f"Loading model params from {self.params_path}")
            params = np.load(self.params_path)
            self.model.coef_ = params["coef"]
            self.model.intercept_ = params["intercept"]
            self.loaded = True

    def fit(self):
        if self.loaded:
            print("Parameters already loaded; skipping fit.")
            return

        self.model.fit(self.X_train, self.y_train)

        np.savez(self.params_path,
                 coef=self.model.coef_,
                 intercept=self.model.intercept_)

    def predict(self) -> np.ndarray:
        return self.model.predict(self.X_test)
    
    def get_target(self):
        return self.y_test

    def get_params(self):
        return self.model.coef_, self.model.intercept_

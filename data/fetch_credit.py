import pandas as pd
import os, sys
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

root_path = os.path.dirname(os.path.dirname(__file__))

class CreditFetcher:
    def __init__(self):
        self.credit_path = os.path.join(root_path, "data", "credit")

    def preprocess_features(
        self,
        features: pd.DataFrame,
        test_features: pd.DataFrame = None,
        encoding: str = 'ohe',
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
        sampling: str = 'smote',
        sampling_strategy: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess for Elastic Net pipeline:
        - Drop ID/TARGET, encode features
        - Split into train/test DataFrames
        - Handle class imbalance
        - Standardize features
        - Select features via ElasticNetCV
        """
        # Extract and drop ID/label
        y = features['TARGET'].values
        X = features.drop(columns=['SK_ID_CURR', 'TARGET'])

        # Prepare test features for alignment
        if test_features is not None:
            X_test = test_features.drop(
                columns=[c for c in ['SK_ID_CURR', 'TARGET'] if c in test_features.columns]
            )
        else:
            X_test = None

        # Encoding
        if encoding == 'ohe':
            X = pd.get_dummies(X)
            if X_test is not None:
                X_test = pd.get_dummies(X_test)
                X, X_test = X.align(X_test, join='inner', axis=1)
        elif encoding == 'le':
            le = LabelEncoder()
            for col in X.columns:
                if X[col].dtype == 'object':
                    X[col] = le.fit_transform(X[col].astype(str))
                    if X_test is not None and col in X_test.columns:
                        X_test[col] = le.transform(X_test[col].astype(str))
        else:
            raise ValueError("encoding must be 'ohe' or 'le'")

        # Split into train/test DataFrames
        stratify_arr = y if stratify else None
        x_train_df, x_test_df, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arr
        )

        # Imbalance sampling on train
        if sampling:
            if sampling == 'oversample':
                sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=random_state)
            elif sampling == 'undersample':
                sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=random_state)
            elif sampling == 'smote':
                sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
            else:
                raise ValueError("sampling must be None, 'oversample', 'undersample', or 'smote'")

            x_res, y_res = sampler.fit_resample(x_train_df.values, y_train)
            x_train_df = pd.DataFrame(x_res, columns=x_train_df.columns)
            y_train = y_res

        # Feature scaling
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_df)
        x_test_scaled = scaler.transform(x_test_df)

        # ElasticNet feature selection
        enet = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], cv=5, random_state=random_state)
        enet.fit(x_train_scaled, y_train)
        coef = enet.coef_
        mask = coef != 0

        x_train_final = x_train_scaled[:, mask]
        x_test_final = x_test_scaled[:, mask]

        return x_train_final, x_test_final, y_train, y_test

    def load_data(self, encoding: str = 'ohe', sampling: str = 'smote', sampling_strategy: float = 1.0):
        features = pd.read_csv(os.path.join(self.credit_path, "train_credit.csv"))
        test_features = pd.read_csv(os.path.join(self.credit_path, "test_credit.csv"))
        features = features.fillna(0)
        test_features = test_features.fillna(0)

        return self.preprocess_features(
            features,
            test_features,
            encoding=encoding,
            sampling=sampling,
            sampling_strategy=sampling_strategy
        )

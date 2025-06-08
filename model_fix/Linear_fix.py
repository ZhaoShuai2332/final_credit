
import os, sys
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

import numpy as np
from scripts.FixPoint import FixedPoint

class Linear_fix:
    def __init__(self, 
               X_train : np.ndarray, 
               y_train : np.ndarray, 
               X_test : np.ndarray, 
               y_test : np.ndarray, 
               weight : np.ndarray, 
               bias : np.ndarray):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.weight = weight
        self.bias = bias

    def predict(self):
        # Ensure bias is a FixedPoint, not an array
        bias_fp = self.bias.item() if isinstance(self.bias, np.ndarray) else self.bias
        res_list = []
        for x in self.X_test:
            res = np.dot(x, self.weight) + bias_fp
            res_list.append(res)
        return np.array(res_list).flatten()
    
    def evl(self):
        y_pred = self.predict()
        


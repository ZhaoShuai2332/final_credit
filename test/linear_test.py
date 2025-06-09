import os, sys
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from scripts.FixPoint import FixedPoint, parse_float, parse_fix
from scripts.activation_interface import *
import numpy as np
from sklearn.metrics import accuracy_score
from model_fix.Linear_fix import LinearModel
from model_fix.Min_Max_Scaler import Min_Max_Scaler
from sklearnex import patch_sklearn, unpatch_sklearn
from scripts.run_proms import parse_args, dataset_selecter
from scripts.FixPoint import parse_float_data, parse_params_name
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

patch_sklearn()

if __name__ == "__main__":
    dataset_name = parse_args().name
    grid_search = parse_args().grid_search
    # load t list
    if grid_search == 1:
        t_list = np.linspace(0.01, 1.00, 100)
    else:
        np.random.seed(42)
        t_list = np.random.rand(100)
    
    # linear model with float data
    print("Linear model with float data")
    train_X, test_X, train_y, test_y = parse_float_data(dataset_name ,is_fix = False)
    params = parse_params_name(dataset_name, "linear", ["coef", "intercept"], is_fix = False)
    linear_model= LinearModel(train_X, train_y, test_X, test_y, params["coef"], params["intercept"])
    pred_float = linear_model.predict()
    res_float = np.array([f_logr_pred(pred_float, t) for t in t_list])
    sum_float = np.array([np.sum(single_arr == 1) for single_arr in res_float])

    # linear model with fixed data
    print("Linear model with fixed data")
    train_X, test_X, train_y, test_y = parse_float_data(dataset_name)
    params = parse_params_name(dataset_name, "linear", ["coef", "intercept"])
    linear_fix = LinearModel(train_X, train_y, test_X, test_y, params["coef"], params["intercept"])
    pred_fix = linear_fix.predict()
    pred_fix_float = parse_fix(pred_fix)
    res_fix_float = np.array([f_logr_pred(pred_fix_float, t) for t in t_list])
    sum_fix_float = np.array([np.sum(single_arr == 1) for single_arr in res_fix_float])

    # compare errors between fix_float and float
    print("error between fix_float and float: ", np.abs(pred_fix_float - pred_float))
    print("Total same: ", np.sum(sum_fix_float == sum_float))

    
    

    





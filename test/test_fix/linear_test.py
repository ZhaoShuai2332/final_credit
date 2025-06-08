import os, sys
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parent_dir)

from scripts.FixPoint import FixedPoint, parse_float, parse_fix
import numpy as np
from sklearn.metrics import accuracy_score
from model_fix.Linear_fix import Linear_fix
from model_fix.Min_Max_Scaler import Min_Max_Scaler
from sklearnex import patch_sklearn, unpatch_sklearn
from scripts.run_proms import parse_args, dataset_selecter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix

input_dir = os.path.join(parent_dir, "params")
# output_dir = os.path.join(parent_dir, "models_fix_point", "fix_params")
# os.makedirs(output_dir, exist_ok=True)

# dataset_names = ["mnist", "uci", "credit"]
# model_names = ["linear", "svr", 'min_max_scaler', 'scaler']
# params_keys = {
#     "linear": ["coef", "intercept"], 
#     "svr" : ["coef", "intercept", "suppost_vertors", "support", "n_support"], 
#     "min_max_scaler": ["min", "scale", "data_min", "data_max"],
#     "scaler": ["min", "scale", "mean", "n_samples"]
#     }

def parse_params_name(dataset_name : str, model_name : str, params_keys : list):
    params = np.load(os.path.join(input_dir, f"{dataset_name}_{model_name}_params.npz"))
    new_params = {}
    for k in params.files:
        arr = params[k]
        if k in params_keys:
            fixed_arr = parse_float(arr, frac_bits=32)
            # fixed_arr = arr
            new_params[k] = fixed_arr
            # print(f"Converted {k}: {fixed_arr}")
        else:
            new_params[k] = arr
    return new_params

def parse_float_data(name):
    train_X, test_X, train_y, test_y = dataset_selecter(name)
    scaler = Min_Max_Scaler(name)
    train_X = scaler.scaler_x(train_X)
    test_X = scaler.scaler_x(test_X)
    
    datasets = [train_X, test_X, train_y, test_y]
    datasets = [parse_float(data.astype(np.float32)) for data in datasets]

    # return parsed datasets as FixedPoint arrays
    return datasets[0], datasets[1], datasets[2], datasets[3]
                
dataset_name = parse_args().name
train_X, test_X, train_y, test_y = parse_float_data(dataset_name)
params = parse_params_name(dataset_name, "linear", ["coef", "intercept"])
linear_model = Linear_fix(train_X, train_y, test_X, test_y, params["coef"], params["intercept"])
print(parse_fix(linear_model.predict()))        




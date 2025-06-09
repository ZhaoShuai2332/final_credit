import math
import numpy as np
from tqdm import tqdm


class FixedPoint:
    def __init__(self, value, frac_bits=32, from_float=False):
        """
        Initialize a fixed-point number
        
        Parameters:
        - value: either a float (if from_float=True) or raw integer value
        - frac_bits: number of fractional bits
        - from_float: whether the value is a float to be converted
        """
        self.frac_bits = frac_bits
        
        if from_float:
            scale = float(1 << frac_bits)
            self.value = int(round(value * scale))
        else:
            self.value = value
    
    def to_float(self, target_frac_bits=None):
        """Convert fixed-point number to float"""
        if target_frac_bits is None:
            target_frac_bits = self.frac_bits
        
        if target_frac_bits != self.frac_bits:
            # Handle both cases where target_frac_bits is larger or smaller
            shift = self.frac_bits - target_frac_bits
            if shift > 0:
                res = self.value >> shift
            else:
                res = self.value << (-shift)
            return float(res) / (1 << target_frac_bits)
        else:
            scale = float(1 << self.frac_bits)
            return float(self.value) / scale
    
    def get_raw_value(self):
        """Get the raw integer value"""
        return self.value
    
    def print_binary(self):
        """Print binary representation"""
        if self.value >= 0:
            bits = bin(self.value)[2:]  # remove '0b' prefix
        else:
            # For negative numbers, use two's complement representation
            bits = bin(self.value & ((1 << (64 if isinstance(self.value, int) else 32)) - 1)[2:])
        
        total_bits = 64 if isinstance(self.value, int) else 32
        bits = bits.zfill(total_bits)
        
        print(f"{bits} (Integer:{total_bits-self.frac_bits-1}bits, Fraction:{self.frac_bits}bits)")
    
    def __mul__(self, other):
        if not isinstance(other, FixedPoint):
            raise TypeError("Operands must be FixedPoint")
            
        # Perform multiplication with proper scaling
        temp = self.value * other.value
        # The product has 2*frac_bits fractional bits, so we need to scale back
        res = temp >> self.frac_bits
        # print(f"temp: {temp}, res: {res}")
        return FixedPoint(res, self.frac_bits)
    
    def __add__(self, other):
        if not isinstance(other, FixedPoint):
            raise TypeError("Operands must be FixedPoint")
            
        return FixedPoint(self.value + other.value, self.frac_bits)
    
    def __sub__(self, other):
        if not isinstance(other, FixedPoint):
            raise TypeError("Operands must be FixedPoint")
            
        return FixedPoint(self.value + -1*other.value, self.frac_bits)
    
    def __repr__(self):
        return f"FixedPoint(value={self.value}, frac_bits={self.frac_bits})"
    
    

def parse_float(float_array: np.ndarray, frac_bits=32):
    shape = float_array.shape
    flat = float_array.flatten()
    fixed_list = [FixedPoint(v, frac_bits, from_float=True) for v in flat]
    return np.array(fixed_list, dtype=object).reshape(shape)

def parse_fix(fix_array: np.ndarray, frac_bits=32):
    shape = fix_array.shape
    flat = fix_array.flatten()
    float_list = [v.to_float(frac_bits) for v in flat]
    return np.array(float_list, dtype=np.float32).reshape(shape)

def multiply_fixed_array(a: np.ndarray, b: np.ndarray):
    return a * b

import os, sys
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

input_dir = os.path.join(parent_dir, "params")
from model_fix.Min_Max_Scaler import Min_Max_Scaler
from scripts.run_proms import dataset_selecter

def parse_params_name(dataset_name : str, model_name : str, params_keys : list, is_fix = True):
    params = np.load(os.path.join(input_dir, f"{dataset_name}_{model_name}_params.npz"))
    if not is_fix:
        return dict(params)
    new_params = {}
    for k in params.files:
        arr = params[k]
        if k in params_keys:
            fixed_arr = parse_float(arr, frac_bits=32)
            new_params[k] = fixed_arr
        else:
            new_params[k] = arr
    return new_params

def parse_float_data(name, is_fix = True):
    save_dir = os.path.join(parent_dir, "data", "fixed_data", name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir, f"x_train_fixed.npz")):
        train_X, test_X, train_y, test_y = dataset_selecter(name)
        scaler = Min_Max_Scaler(name)
        train_X = scaler.scaler_x(train_X)
        test_X = scaler.scaler_x(test_X)
        np.savez(os.path.join(save_dir, f"x_train_fixed.npz"), train_X)
        np.savez(os.path.join(save_dir, f"x_test_fixed.npz"), test_X)
        np.savez(os.path.join(save_dir, f"y_train_fixed.npz"), train_y)
        np.savez(os.path.join(save_dir, f"y_test_fixed.npz"), test_y)
    else:
        print("load fixed data...")
        # load fixed data from .npz archives and extract arrays
        with np.load(os.path.join(save_dir, f"x_train_fixed.npz")) as data:
            train_X = data[data.files[0]]
        with np.load(os.path.join(save_dir, f"x_test_fixed.npz")) as data:
            test_X = data[data.files[0]]
        with np.load(os.path.join(save_dir, f"y_train_fixed.npz")) as data:
            train_y = data[data.files[0]]
        with np.load(os.path.join(save_dir, f"y_test_fixed.npz")) as data:
            test_y = data[data.files[0]]

    datasets = [train_X, test_X, train_y, test_y]
    dataset_names = ['train_X', 'test_X', 'train_y', 'test_y']
    
    print(f"Dataset: {name}")
    print("=" * 50)
    for i, (data, name_label) in enumerate(zip(datasets, dataset_names)):
        print(f"{name_label}:")
        print(f"  Shape: {data.shape}")
        print(f"  Data type: {data.dtype}")
        print(f"  Memory usage: {data.nbytes / 1024 / 1024:.2f} MB")
        
        # For label datasets (train_y, test_y), count positive and negative examples
        if 'y' in name_label:
            unique, counts = np.unique(data, return_counts=True)
            print(f"  Class distribution:")
            for class_val, count in zip(unique, counts):
                print(f"    Class {class_val}: {count} samples ({count/len(data)*100:.1f}%)")
        
        print("-" * 30)

    if not is_fix:
        return datasets[0], datasets[1], datasets[2], datasets[3]
    total_elements = sum(data.size for data in datasets)
    processed = 0
    datasets_fixed = []
    with tqdm(total=total_elements, desc="Converting to fixed point", unit="elements") as pbar:
        for data in datasets:
            data_float32 = data.astype(np.float32)
            datasets_fixed.append(parse_float(data_float32))
            processed += data.size
            pbar.update(data.size)

    # return parsed datasets as FixedPoint arrays
    return datasets_fixed[0], datasets_fixed[1], datasets_fixed[2], datasets_fixed[3]
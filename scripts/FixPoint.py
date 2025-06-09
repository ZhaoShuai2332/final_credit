import math
import numpy as np

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




# def main():
#     a = -3.14
#     b = 2.71
#     frac_bits = 16  # Q48.16 format
    
#     print(f"原始浮点数: a = {a}, b = {b}")
    
#     # 1. Convert float to fixed-point
#     a_fixed = FixedPoint(a, frac_bits, from_float=True)
#     b_fixed = FixedPoint(b, frac_bits, from_float=True)
    
#     print(f"定点数表示: a_fixed = {a_fixed.get_raw_value()}, b_fixed = {b_fixed.get_raw_value()}")
    
#     # 2. Perform fixed-point multiplication
#     res = a_fixed * b_fixed
    
#     print(f"乘法结果(定点数): {res.get_raw_value()}")
    
#     # 3. Convert back to float
#     ans = res.to_float()
    
#     print(f"乘法结果(浮点数): {ans}")
#     print(f"精确结果: {a * b}")
#     print(f"误差: {(a * b) - ans}")


# if __name__ == "__main__":
#     main()
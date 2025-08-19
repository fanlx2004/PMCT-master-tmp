import numpy as np
import math
from scipy.stats import norm

'''
    These functions perform experiments on obtaining the relative error upper bound of the collision risk.
'''

def construct_original_matrix(K, i):
    np.random.seed(i)
    matrix = np.zeros((K, K))
    for row_idx in range(K-1):
        matrix[row_idx] = np.random.dirichlet(np.ones(K))
    matrix[-1, -1] = 1.0
    return matrix

def adjust_row(row, betastar, Nstar, j):
    K = row.size
    np.random.seed(j)
    delta = norm.ppf(1-betastar/2) * np.sqrt(row * (1 - row) / Nstar)
    high = np.minimum(row + delta, 1.0)
    low = np.maximum(row - delta, 0.0)
    
    new_row = row.copy()
    for _ in range(10):
        direction = np.random.rand(K) < 0.5
        delta_plus = high - new_row
        delta_minus = new_row - low
        
        S_plus = np.sum(delta_plus[direction])
        S_minus = np.sum(delta_minus[~direction])
        
        if S_plus + S_minus < 1e-10:
            continue
            
        adjust_ratio = 0.9
        T_plus = S_plus * adjust_ratio
        T_minus = S_minus * adjust_ratio
        
        adjustment = np.zeros(K)
        if S_plus > 0:
            adjustment[direction] = T_plus * delta_plus[direction] / S_plus
        if S_minus > 0:
            adjustment[~direction] = -T_minus * delta_minus[~direction] / S_minus
            
        temp_row = new_row + adjustment
        temp_row = np.clip(temp_row, low, high)
        
        correction = 1.0 - np.sum(temp_row)
        if np.abs(correction) < 1e-9:
            new_row = temp_row
            break
        
        available_plus = high - temp_row
        available_minus = temp_row - low
        total_available = np.sum(available_plus) + np.sum(available_minus)
        
        if total_available < 1e-10:
            continue
            
        ratio_plus = available_plus / total_available
        ratio_minus = available_minus / total_available
        final_adjust = ratio_plus * correction - ratio_minus * correction
        new_row = temp_row + final_adjust
        new_row = np.clip(new_row, low, high)
        
        if np.isclose(np.sum(new_row), 1.0, atol=1e-9):
            break
            
    new_row = np.clip(new_row, low, high)
    new_row /= np.sum(new_row)
    return new_row

def process_i(i, K, betastar, Nstar, threshold):
    original_matrix = construct_original_matrix(K, i)
    power_original = np.linalg.matrix_power(original_matrix, K-2)
    total_count = 0
    
    for j in range(100):
        new_matrix = original_matrix.copy()
        for r in range(K-1):
            new_matrix[r] = adjust_row(original_matrix[r], betastar, Nstar, j + i * 100)
        
        power_new = np.linalg.matrix_power(new_matrix, K-2)
        difference = power_new - power_original
        last_col_diff = difference[:-1, -1]
        last_col_original = power_original[:-1, -1]
        
        relative_diff = np.abs(last_col_diff) / np.abs(last_col_original)
        count = np.sum(relative_diff > threshold)
        total_count += count
    return total_count

def relative_error_ub_test(K = 10, Nstar = 200, threshold = 0.93, betastar = 0.01, epsilonstar = 0.01):
    
    N = int(np.log(epsilonstar) / np.log(1 - betastar))
    
    total_count = 0
    for i in range(math.ceil(N / 100)):
        total_count += process_i(i, K+2, betastar, Nstar, threshold)
    
    if total_count == 0:
        print("All adjusted matrices passed the test.")
        return True
    else:
        print(f"{total_count} adjusted matrices failed the test.")
        return False
    

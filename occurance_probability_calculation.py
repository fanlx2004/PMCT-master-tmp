import torch
import numpy as np
import multiprocessing
import json
from network_structure.Q_net import DRLAgent
from utility_functions.deduction_function import deduction
from functools import partial
from conf import conf
from collections import Counter

'''
    This script calculates the initial state occurance probability in each subset, 
    and calculates the state distribution D used in sampling initial state that follows NDE.
'''
## Calculate the initial state occurance probability in each subset

def calculate_interval_probabilities(data, lb, ub, delta, atol=1e-8):
    if not data:
        return []

    n_bins = int(np.floor((ub - lb - atol) / delta)) + 1
    edges = np.linspace(lb, lb + n_bins * delta, n_bins + 1)
    edges[-1] = ub
    counts, _ = np.histogram(data, bins=edges)
    in_bins = (data >= lb) & (data < ub) 
    equals_ub = np.isclose(data, ub, atol=atol)  
    to_add = equals_ub & ~in_bins  
    
    counts[-1] += np.sum(to_add)
    
    total = len(data)
    probabilities = counts / total
    probabilities = probabilities / np.sum(probabilities) 
    return probabilities.tolist()

def compute_probability(lst):
    counts = Counter(lst)
    total = len(lst)
    return [counts.get(i, 0) / total for i in range(3)] 

def worker(_):
    return partial_deduction()

source = "./result/occurance_probability_data/"

agent = DRLAgent()
agent.q_net.load_state_dict(torch.load(conf.final_net_params_path))

generate_data = True  # If you want to generate new data, set this to True

A = np.zeros((conf.K+1), dtype=int)
P = np.zeros((conf.K+1), dtype=float)

partial_deduction = partial(deduction, all_state_flag=True)


    
data_dir = source + "nde_state_data.json"
coodinate_list = []

if generate_data:
    output = []
    with multiprocessing.Pool() as pool:
        output = pool.map(worker, range(5000))
    
    for j in range(len(output)):
        if len(output[j]) < 300:
            continue
        coodinate_list.extend(output[j][300:])

    with open(data_dir, "w") as f:
            json.dump(coodinate_list, f, indent=4)

    R_list = agent.get_q_value(coodinate_list)
    counts = np.bincount(R_list, minlength=conf.K+2)[1:conf.K+2]
    A += counts
    np.savetxt(source + "occurance_time.txt", A, fmt='%d')
    P = A / np.sum(A)
    np.savetxt(source + "occurance_probability.txt", P, fmt='%.4f')   

## Calculate the state distribution D

    transposed = [list(row) for row in zip(*coodinate_list)]

    probability = []

    probability.append(compute_probability(transposed[0]))
    lb = np.array([0, 20, 6, 20, -118, 20, 6, 20, -118, 20, 6, 20, -118, 20], dtype=float)
    ub = np.array([0, 40, 118, 40, -6, 40, 118, 40, -6, 40, 118, 40, -6, 40], dtype=float)
    delta = np.array([0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1], dtype=float)

    for i in range(1,14):
        lb0 = lb[i]
        ub0 = ub[i]
        delta0 = delta[i]
        old_list = transposed[i]
        if i % 4 == 3 or i % 4 == 0:
            new_list = [lb[i]+0.000001 if x is None else x for x in old_list]
        else:
            new_list = [ub[i]-0.000001 if x is None else x for x in old_list]
        probability.append(calculate_interval_probabilities(new_list, lb0, ub0, delta0))
        
    with open(source + "nde_init.json", 'w') as f:
        json.dump(probability, f, indent=4)
    
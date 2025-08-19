import numpy as np
import random
import json

'''
    These functions are used in sampling initial states from the state set.
    Attention: These functions are only available under the three-lane autonomous driving system settings.
'''

def sample_in_nde(num, mode):
    # Sample a list of coordinates based on the NDE model.
    source = "./result/occurance_probability_data/" + mode + "_init.json"
    assert mode == 'nde', "Mode only supports 'nde' for now."
    with open(source, 'r') as f:
        init = json.load(f)
    coodinate_list = []
    lb0 = np.array([0, 20, 6, 20, -118, 20, 6, 20, -118, 20, 6, 20, -118, 20], dtype=float)
    ub0 = np.array([0, 40, 118, 40, -6, 40, 118, 40, -6, 40, 118, 40, -6, 40], dtype=float)
    delta0 = np.array([0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1], dtype=float)
    for i in range(num):
        coodinate = []
        coodinate.append(random.choices([0, 1, 2], weights=init[0], k=1)[0])
        for j in range(1,14):
            length = len(init[j])
            index = random.choices(list(range(length)), weights=init[j], k=1)[0]
            value = random.uniform(lb0[j]+index*delta0[j], lb0[j]+(index+1)*delta0[j])
            value = min(max(value, lb0[j]), ub0[j])
            coodinate.append(value)
        if coodinate[0] == 0:
            coodinate[10] = None
            coodinate[11] = None
            coodinate[12] = None
            coodinate[13] = None
        elif coodinate[0] == 2:
            coodinate[6] = None
            coodinate[7] = None
            coodinate[8] = None
            coodinate[9] = None
        coodinate_list.append(coodinate)
        
    return coodinate_list

def adjust_sampling(num):
    # Sample a list of coordinates randomly.
    coodinate_0 = None
    r = None
    coodinate_list = []
    while len(coodinate_list) != num:
        coodinate = random_sample(coodinate_0, r)
        coodinate_list.append(coodinate)
    return coodinate_list
   

def random_sample(coodinate_0=None,r=None):
    '''
    Sample a coordinate randomly or based on the previous coordinate and risk level.
    Args:
        coodinate_0: Previous coordinate, if None, the new coordinate will be sampled randomly
        r: Previous risk level, if None, a new risk level will be sampled randomly
    '''
    dc_range = [0, 1, 2]
    lb = np.array([20, 6, 20, -115, 20, 6, 20, -115, 20, 6, 20, -115, 20], dtype=float)
    ub = np.array([40, 115, 40, -6, 40, 115, 40, -6, 40, 115, 40, -6, 40], dtype=float)
    delta = np.array([1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1], dtype=float)
    extreme_value_b = [[16,105],[23,37],[-105,-16]]
    if coodinate_0 is None and r is None:
        # randomly sample a coordinate
        coodinate = []
        coodinate.append(random.choice(dc_range))
        for i in range(len(lb)):
            coodinate.append(random.uniform(lb[i], ub[i]))
        if coodinate[0] == 0:
            coodinate[10] = None 
            coodinate[11] = None
            coodinate[12] = None
            coodinate[13] = None
        elif coodinate[0] == 2:
            coodinate[6] = None 
            coodinate[7] = None
            coodinate[8] = None
            coodinate[9] = None
        for i in range(1, 7):
            if coodinate[2*i] is not None and coodinate[2*i+1] is not None and random.random() < 0.05:
                coodinate[2*i] = None
                coodinate[2*i+1] = None
        
    else:
        # sample a coordinate based on the previous coordinate and risk level
        coodinate = []
        coodinate.append(coodinate_0[0])
        for i in range(1,len(coodinate_0)):
            if coodinate_0[i] is None:
                coodinate.append(None)
            else:
                c = random.uniform(delta[i-1]/3, delta[i-1]/2)
                if coodinate_0[i] + c > ub[i-1]:
                    coodinate.append(coodinate_0[i] - c)
                elif coodinate_0[i] - c < lb[i-1]:
                    coodinate.append(coodinate_0[i] + c)
                else:
                    coodinate.append(random.choice([coodinate_0[i] + c, coodinate_0[i] - c]))
                
    for i in range(1, 7):
            if coodinate[2*i] is None:
                coodinate[2*i+1] = None   
            if coodinate[2*i+1] is None:
                coodinate[2*i] = None     
    
    if coodinate[0] == None or coodinate[1] == None:
        raise ValueError("Invalid coordinate: first two elements cannot be None")
    return coodinate
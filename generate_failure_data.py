import torch
from utility_functions.deduction_function import deduction
import os
from utility_functions.sample_function import random_sample
from network_structure.Q_net import DRLAgent
import json
from conf import conf

'''
    This script is used to generate data that fails the verification of the Q-net.  
    The data will be used in the follow-up verification of the Q-net as supplementary data,  
    if the verification can't end within the predefined sample number.
'''

agent = DRLAgent() 
if os.path.exists(conf.training_net_params_path):
        agent.q_net.load_state_dict(torch.load(conf.training_net_params_path))

n = 0
coo_tem = None
r_tem = None
file_index = 0
data_dict = {} 
for i in range(1,11):
    data_dict[str(i)] = []
    
for s in range(10000000):
    
    print("$$$$$$$$$$$$$$$$$$$")
    print("sampling times:",s)
    print("$$$$$$$$$$$$$$$$$$$")
    coodinate = random_sample(coo_tem, r_tem)
    point_list = deduction(coodinate)
    r = len(point_list)
    print("****************")
    if r == 11:
        print("r = 11")
        coo_tem = None
        r_tem = None
    else:
      flag = True
      for t in range(len(point_list)):
        coodinate = point_list[t]
        print("point ", t)
        print(f"actual risk step: {r}")
        Q = agent.get_q_value(coodinate)
        print(f"net Q value: {Q}")
        if flag :
            coo_tem = coodinate
            r_tem = r
        if Q <= r :
            print("Success!")
            r -= 1
        else:
            print("Unsuccess! Save.")
            flag = False
            st = coodinate
            if t == len(point_list) - 1:
                st_1 = None
            else:
                st_1 = point_list[t+1]
            data_dict[str(r)].append([st, st_1])
            n += 1
            print("Now n:", n)
            r -= 1
    print("****************")
    if s % 200 == 0 and s > 0:
        print(data_dict)
        with open(f'./result/failure_data/failure_data_{file_index}.json', 'w') as file:
            json.dump(data_dict, file, indent=4)
        file_index += 1
        data_dict = {} 
        for i in range(1,11):
            data_dict[str(i)] = []

        
        
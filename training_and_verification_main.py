import torch
import numpy as np
import random
from utility_functions.deduction_function import deduction
import os
import multiprocessing
from pathlib import Path
import json
from utility_functions.sample_function import random_sample
from network_structure.Q_net import DRLAgent
from conf import conf

'''
    This script trains and verifies the DRL-based Q-network to solve the set partition problem.
'''


if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    available_cores = multiprocessing.cpu_count()
    
    
    agent = DRLAgent()
    if os.path.exists(conf.training_net_params_path):
        agent.q_net.load_state_dict(torch.load(conf.training_net_params_path))
        agent.q_net.to(device)
    
    step = 1000000
    N = np.zeros(conf.K+1)
    for i in range(conf.K+1):
        N[i] = np.log(conf.beta1) / np.log(1 - conf.epsilons[i])
    verification = []
    
    loss_total = 0
    loss_min = 100000
    for s in range(step):
        print("*****************************")
        print(s)
        print("*****************************")
        n = 0
        k = 0
        coo_tem = None
        r_tem = None
        json_files = list(Path(conf.failure_data_path).glob("*.json"))

########   add data to buffer   #########
        while n < conf.sample_per_adding:
            print("n = ", n)
            k += 1
            
            ppp = np.random.randint(int(available_cores*2/3), available_cores)
            coo_tem_list = [coo_tem] * ppp
            coo_tem_list.extend([None] * (available_cores-ppp))
            r_tem_list = [r_tem] * ppp
            r_tem_list.extend([None] * (available_cores-ppp))
            
            coodinate_list = []
            for i in range(available_cores):
                coodinate_list.append(random_sample(coo_tem_list[i], r_tem_list[i]))
            
            
            with multiprocessing.Pool() as pool:
                point_list_list = pool.map(deduction, coodinate_list)

            random.shuffle(point_list_list)
            for j in range(len(point_list_list)):
                point_list = point_list_list[j]
                r = len(point_list)
                if r == conf.K+1:
                    # safe sample, only add the first state
                    s_t = point_list[0]
                    s_t1 = point_list[1]
                    for i in range(1, len(s_t)):
                        if s_t[i] is not None:
                            s_t[i] = max(conf.lb[i-1], min(s_t[i], conf.ub[i-1]))
                    for i in range(1, len(s_t1)):
                        if s_t1[i] is not None:
                            s_t1[i] = max(conf.lb[i-1], min(s_t1[i], conf.ub[i-1]))
                    agent.buffer.push(s_t, s_t1, r)
                    n += 1
                else:
                    # unsafe sample, add all states
                    op = np.random.randint(0,r)
                    # let the following data be randomly sampled near the dangerous state
                    coo_tem = point_list[op]
                    r_tem = r - op
                    n += r
                    for i in range(0, len(point_list)):
                        s_t = point_list[i]
                        if r == 1:
                            s_t1 = [None] * conf.STATE_DIM
                        else:
                            s_t1 = point_list[i+1]
                        for l in range(1, len(s_t)):
                            if s_t[l] is not None:
                                s_t[l] = max(conf.lb[l-1], min(s_t[l], conf.ub[l-1]))
                        for l in range(1, len(s_t1)):
                            if s_t1[l] is not None:
                                s_t1[l] = max(conf.lb[l-1], min(s_t1[l], conf.ub[l-1]))
                        agent.buffer.push(s_t, s_t1, r)
                        r -= 1
                        
########   Training   #########    
                   
        for i in range(conf.update_times_per_epoch):        
            loss = agent.update()
            if loss is not None:
                print("*****************************")
                print(f"Step {s:5d} | Loss: {loss:.4f}")
                print("*****************************")
                loss_total += loss    
            
            
##############  Verification  #############
        if s % conf.verification_interval == 0 and s > 0:
            print("*****************************")
            print("Verification")
            print("*****************************")
            N_q = [0]*conf.K+1
            coo_tem = None
            r_tem = None
            flag = True
            while not all(N_q[i] >= N[i] for i in range(0,conf.K+1)):
                yyy = sum(N_q)
                if yyy >= sum(N)*10:
                    # use the pre-collected failure data if the sample number is large enough but the verification hasn't ended
                    print("Using pre-collected failure data.")
                    key_r = conf.K+1
                    for o in range(conf.K):
                        if N_q[o] < N[o]:
                            key_r = o + 1
                            break
                    
                    
                    s_t_list = [None] * available_cores

                    for uuu in range(available_cores):
                      while s_t_list[uuu] == None:
                        random_json_file = random.choice(json_files)
                        with open(random_json_file, "r", encoding="utf-8") as f:
                            data = json.load(f)  
                            if data[str(key_r)] == []:
                                json_files.remove(random_json_file)
                                continue
                            else:
                                random_element = random.choice(data[str(key_r)])
                                s_t_list[uuu] = random_element[0]

                    
                    # To avoid overfitting these data, only the initial states are used, and the step number to collision is resampled. 
                    with multiprocessing.Pool() as pool:
                        point_list_list = pool.map(deduction, s_t_list)
                    
                    for ooo in range(available_cores):
                        r = len(point_list_list[ooo])

                        Q = agent.get_q_value(s_t_list[ooo])
                        print(f"actual risk step: {r}")
                        print(f"net Q value: {Q}")
                        if Q > r:
                            print(f"Unsuccess! In sample {ooo}.")
                            coodinate_list_fail = [s_t_list[ooo]]*5
                            
                            # add the failed sample and its neighbour to the buffer, so that the Q-net can learn from it
                            for i in range(45):
                                coodinate_list_fail.append(random_sample(s_t_list[ooo], r))
                            
                            with multiprocessing.Pool() as pool:
                                point_list_list = pool.map(deduction, coodinate_list_fail)
                            
                            for point_list in point_list_list:
                                r = len(point_list)
                                if r == conf.K+1:
                                    agent.buffer.push(point_list[0],point_list[1], r)
                                else:
                                    for i in range(0, len(point_list)):
                                        s_t = point_list[i]
                                        if i == len(point_list) - 1:
                                            s_t1 = [None] * conf.STATE_DIM
                                        else:
                                            s_t1 = point_list[i+1]
                                        agent.buffer.push(s_t, s_t1, r)
                                        r -= 1
                            flag = False
                            break
                        else:
                            N_q[Q-1] += 1
                            print(f"Success in sample {ooo}.")
                    if not flag:
                        break
                    else:
                        print("$$$$$$$$$$$$$$$$$$$$$$")
                        print(N_q)
                        print("$$$$$$$$$$$$$$$$$$$$$$")
                        continue
                    
                else:
                    # randomly sample to verify
                    break_flag = False
                    
                    op = np.random.randint(int(available_cores/2),available_cores)
                    coo_tem_list = [coo_tem] * op
                    r_tem_list = [r_tem] * op
                    coo_tem_list.extend([None] * (available_cores-op))
                    r_tem_list.extend([None] * (available_cores-op))
                    
                    coodinate_list = []
                    for i in range(available_cores):
                        coodinate_list.append(random_sample(coo_tem_list[i], r_tem_list[i]))

                    with multiprocessing.Pool() as pool:
                        point_list_list = pool.map(deduction, coodinate_list)
                    
                    random.shuffle(point_list_list)
                    for aa in range(len(point_list_list)):
                        point_list = point_list_list[aa]
                        
                        r = len(point_list)
                        if r == conf.K+1:
                            coo_tem = None
                            r_tem = None
                        else:
                            list_Q_tem = []
                            for lll in range(conf.K+1):
                                if N_q[lll] < N[lll]:
                                    list_Q_tem.append(lll)
                                if len(list_Q_tem) == 0:
                                    break
            
                            else:
                                random.shuffle(list_Q_tem)
                                coo_tem = point_list[0]
                                r_tem = r
                                for kkk in list_Q_tem :
                                    if kkk + 1 <= r :
                                        r_tem = kkk + 1
                                        coo_tem = point_list[r-kkk-1]
                                        break
                            

                            for i in range(0, len(point_list)):
                                s_t = point_list[i]
                                if r == 1:
                                    s_t1 = [None] * conf.STATE_DIM
                                else:
                                    s_t1 = point_list[i+1]  
                                print(f"actual risk step: {r}")    
                                Q = agent.get_q_value(s_t) 
                                print(f"net Q value: {Q}")                 
                                if Q > r:
                                    print(f"Unsuccess!")
                                    # add the failed sample to the buffer, so that the Q-net can learn from it
                                    for xxx in range(20):
                                        agent.buffer.push(s_t, s_t1, r)
                                    break_flag = True
                                    break
                                else:
                                    print(f"Success.")
                                    N_q[Q-1] += 1
                                r -= 1
                        if break_flag:
                            break    
                    if break_flag:
                        break

                    print("$$$$$$$$$$$$$$$$$$$$$$")
                    print(N_q)
                    print("$$$$$$$$$$$$$$$$$$$$$$")
                
                    
            if all(N_q[i] >= N[i] for i in range(0,conf.K+1)):
                print("*****************************")
                print("Verification success!")
                print("*****************************")
                torch.save(agent.q_net.state_dict(), conf.training_net_params_path)
                break
            

                
####  Save loss  ####        
        if s % conf.loss_save_interval == conf.loss_save_interval - 1:
            loss_tem = loss_total / conf.loss_save_interval * conf.update_times_per_epoch
            print("*****************************")
            print("Average loss:", loss_tem)
            print("*****************************")
            loss_total = 0
            if loss_tem < loss_min:
                loss_min = loss_tem
                print("*****************************")
                print("New minimum loss")
                print("*****************************")
            
                torch.save(agent.q_net.state_dict(), conf.training_net_params_path)
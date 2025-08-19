import torch
import numpy as np
from utility_functions.deduction_function import deduction
from utility_functions.sample_function import sample_in_nde
from network_structure.Q_net import DRLAgent
from multiprocessing import Pool
from conf import conf
from utility_functions.relative_error_ub_test import relative_error_ub_test

'''
This script is used to calculate the expectations of collision risk for different sets of states and the total collision risk upper bound in the NDE environment.
'''

source = "./result/collision_risk_result/"
occurance_source = "./result/occurance_probability_data/"

agent = DRLAgent() 
agent.q_net.load_state_dict(torch.load(conf.final_net_params_path))

# If you want to calculate the transition matrix, set this to True
calculate_transition_matrix = False
# If you want to generate real collision risk data, set this to True
generate_real_collision_risk_data = False

K = conf.K
Nstar = 200 # Minimum number of samples for each set X_u(i+1)
A = np.zeros((K+1, K+2), dtype=int) # A[i][j] means the number of times that set X_u(i+1) transits to set X_u(j+1)
p = np.zeros((K+2, K+2),dtype=float) # p[i][j] means the probability of set X_u(i+1) transiting to set X_u(j+1)
E = np.zeros(K+1, dtype=float)  # E[i] means the collision risk expectation of set X_u(i+1)
P = np.zeros(K+1, dtype=float)  # P[i] means the initial state occurrence probabilities of set X_u(i+1)
hat_theta = 1 # collision risk estimation
relative_error_ub = 0.85 # relative error upper bound for the collision risk estimation
theta = 1 # collision risk upper bound

betastar = 0.005 # significance level
epsilonstar = 0.01 # failure rate 

A_file = source + "transition_time.txt"
p_file = source + "transition_probability.txt"
E_file = source + "subset_collision_risk_expectation.txt"
collision_risk_estimation_file = source + "collision_risk_estimation.txt"
collision_risk_upper_bound_file = source + "collision_risk_upper_bound.txt"
relative_error_ub_file = source + "relative_error_ub.txt"
P_file = occurance_source + "occurance_probability.txt"
real_collision_risk_file = source + "real_collision_risk.txt"


if not calculate_transition_matrix:
    # Load existing data of transition matrix and expectations
    
    A = np.loadtxt(A_file, dtype=int, delimiter=',')
    p = np.loadtxt(p_file, dtype=float)
    E = np.loadtxt(E_file, dtype=float)
    
else:
    # Calculate the transition matrix and expectations
    
    p[K+1][K+1] = 1
    coo_tem = None
    r_tem = None

    while not all(np.sum(A[j]) >= Nstar for j in range(0,K+1)):
        s_t_list = sample_in_nde(Nstar, 'nde')
        
        with Pool() as pool:
            point_list_list = pool.map(deduction, s_t_list)
        
        for jjj in range(len(point_list_list)):
            point_list =point_list_list[jjj]
            r = len(point_list)

                
            for l in range(0, r):
                Qa = agent.get_q_value(point_list[l])
                if l == r - 1:
                    if r == K+1:
                        continue
                    else:
                        Qb = 0
                else : 
                    Qb = agent.get_q_value(point_list[l+1])
                if Qb == 0:
                    A[Qa - 1][K+1] += 1
                else :
                    A[Qa - 1][Qb - 1] += 1    
                    
            np.savetxt(A_file, A, fmt='%d', delimiter=',')

    for i in range(K+1):
        for j in range(K+2):
            p[i][j] = A[i][j] / np.sum(A[i])
    np.savetxt(p_file, p, fmt='%.4f')

    Q = np.linalg.matrix_power(p, K)

    for i in range(K+1):
        E[i] = Q[i][-1]            
        print(f"set X_u{i+1}'s expectation is {E[i]}\n")
    np.savetxt(E_file, E, fmt='%.4f')

# Load the initial state occurrence probabilities
P = np.loadtxt(P_file, dtype=float)

# Calculate the collision risk estimation
hat_theta = np.sum(E * P)
print(f"Collision risk estimation is {hat_theta}\n")
with open(collision_risk_estimation_file, 'w') as f:
    f.write(f"{hat_theta}\n")

# Verify the relative error upper bound
while not relative_error_ub_test(K, Nstar, relative_error_ub, betastar, epsilonstar):
    relative_error_ub += 0.005
with open(relative_error_ub_file, 'w') as f:
    f.write(f"{relative_error_ub}\n")
print(f"Relative error upper bound is {relative_error_ub}\n")

# Calculate the collision risk upper bound
theta = hat_theta * (1 + relative_error_ub)
with open(collision_risk_upper_bound_file, 'w') as f:
    f.write(f"{theta}\n")
print(f"Collision risk upper bound is {theta}\n")

if not generate_real_collision_risk_data:
    # Load existing real collision risk data 
    
    collision_risk_real = np.loadtxt(real_collision_risk_file, dtype=float)
    
else:
    # Generate real collision risk data and calculate the real collision risk 
    
    collision_risk_real = np.zeros(100, dtype=float)
    for i in range(0,100):
        n = 0
        m = 0
        s_list = sample_in_nde(2000, 'nde')
        with Pool() as pool:
            point_list_list = pool.map(deduction,s_list)
        for j in range(2000):
            point_list = point_list_list[j]
            r = len(point_list)
            n += 1
            if r < 11:
                m += 1
        p_real = m / n
        collision_risk_real[i] = p_real
        with open(real_collision_risk_file, 'a') as f:
            f.write(f'{p_real}\n')

collision_risk_real_mean = np.mean(collision_risk_real)
collision_risk_real_std = np.std(collision_risk_real)
collision_risk_real_max = np.max(collision_risk_real)
print(f"Real collision risk mean is {collision_risk_real_mean}\n")
print(f"Real collision risk std is {collision_risk_real_std}\n")
print(f"Real collision risk max is {collision_risk_real_max}\n")

if collision_risk_real_max <= theta:
    print("The real collision risk is within the upper bound.")
else:
    print("The real collision risk exceeds the upper bound.")

import numpy as np
import os
import json
import matplotlib.pyplot as plt
from utility_functions.deduction_function import deduction_for_MP
from multiprocessing import Pool
from safety_metric_tool.MPrISM_solver import MPrISM_solver
from safety_metric_tool.PET_solver import PET_solver
from safety_metric_tool.PMCT_solver import PMCT_solver
from utility_functions.sample_function import adjust_sampling
import time
import json

source = "./result/safety_metric_experiment_data/"
figure_dir = "./result/experiment_figure/ROC_PR/"

'''
    This script operates scenario experiments on safety metrics and save the comparison results,
    and draws the ROC and PR curves.
'''

## Operate scenario experiments on safety metrics and save the comparison results

generate_state_list = False # If false, load the state list of scenarios to be tested; otherwise, create a new one if you need to sample again 
calculate_pet = False # If true, calculate the PET again
calculate_pmct = False # If true, calculate the PMCT again
calculate_mprism = False # If true, calculate the MPrISM again

if not generate_state_list:
    with open(os.path.join(source, 'state_list.json'), 'r') as f:
        state_list = json.load(f)
else:
    sample_num = 5150
    coodinate_list = adjust_sampling(sample_num) 
    state_list = []
    with Pool() as pool:
        state_list = pool.map(deduction_for_MP, coodinate_list)     
    with open(os.path.join(source, 'state_list.json'), 'w') as f:
        json.dump(state_list, f, indent=4)        

# save the true value of the collision time from the initial states
true_list = []
for i in range(len(state_list)):
    true_list.append(state_list[i][0])

# calculate the PET for each initial state

pet_cal_time = []

if calculate_pet:
    pet_list = []
    for i in range(len(state_list)):
        
        print(f"PET : Processing state {i+1}/{len(state_list)}")
        start_time = time.time()
        pet = PET_solver(state_list[i][1][0], state_list[i][1][1])
        end_time = time.time()
        pet_list.append(pet)
        if i == 0:
            continue
        pet_cal_time.append(end_time - start_time)
        
    TandP_list = np.array([true_list, pet_list]).T

    np.savetxt(os.path.join(source, 'TandP_list.txt'), TandP_list, fmt='%.4f')
    
    with open(source + 'PET_computation_time.txt', 'w') as file:
        for pt in pet_cal_time: 
            file.write(f"{pt:.15f}\n")
    
# calculate the PMCT for each initial state

pmct_cal_time = []

if calculate_pmct:
    net_list = []

    for i in range(len(state_list)):
        AV_state = state_list[i][1][0]
        BV_list = state_list[i][1][1]
        state_for_net = [
            int((AV_state[1] - 41.999999) / 4.0), # AV lane id
            AV_state[2], # AV vx
            None if BV_list[0] == None else np.hypot(BV_list[0][0] - AV_state[0], BV_list[0][1] - AV_state[1]).item(), #BV_Lead distance
            None if BV_list[0] == None else BV_list[0][2], # BV_Lead vx
            None if BV_list[1] == None else -np.hypot(BV_list[1][0] - AV_state[0], BV_list[1][1] - AV_state[1]).item(), # BV_Foll distance
            None if BV_list[1] == None else BV_list[1][2], # BV_Foll vx
            None if BV_list[2] == None else np.hypot(BV_list[2][0] - AV_state[0], BV_list[2][1] - AV_state[1]).item(), # BV_LeftLead distance
            None if BV_list[2] == None else BV_list[2][2], # BV_LeftLead vx
            None if BV_list[3] == None else -np.hypot(BV_list[3][0] - AV_state[0], BV_list[3][1] - AV_state[1]).item(), # BV_LeftFoll distance
            None if BV_list[3] == None else BV_list[3][2], # BV_LeftFoll vx
            None if BV_list[4] == None else np.hypot(BV_list[4][0] - AV_state[0], BV_list[4][1] - AV_state[1]).item(), # BV_RightLead distance
            None if BV_list[4] == None else BV_list[4][2], # BV_RightLead vx
            None if BV_list[5] == None else -np.hypot(BV_list[5][0] - AV_state[0], BV_list[5][1] - AV_state[1]).item(), # BV_RightFoll distance
            None if BV_list[5] == None else BV_list[5][2] # BV_RightFoll vx
        ]
        start_time = time.time()
        net_value = PMCT_solver(state_for_net)
        end_time = time.time()
        net_list.append(net_value)
        if i == 0:
            continue
        pmct_cal_time.append(end_time - start_time)        

    with open(source + 'PMCT_computation_time.json', 'w') as file:
        json.dump(pmct_cal_time, file, indent=4)
    
    TandR_list = np.array([true_list, net_list]).T
    np.savetxt(os.path.join(source, 'TandR_list.txt'), TandR_list, fmt='%.4f')

    print("PMCT calculation completed.")

# calculate the MPrISM for each initial state
# Attention: This part may take a long time ! We suggest using the data we have collected previously.

mprism_cal_time = []

if calculate_mprism:
    mprism_list = []
    for i in range(len(state_list)):
        
        mprism = MPrISM_solver(state_list[i][1][0], state_list[i][1][1])
        print(f"MPrISM : Processing state {i+1}/{len(state_list)} completed.")
        start_time = time.time()
        mprism_list.append(mprism)
        end_time = time.time()
        if i == 0:
            continue
        mprism_cal_time.append(end_time - start_time)   
        
    with open(source + 'MPrISM_computation_time.json', 'w') as file:
        json.dump(mprism_cal_time, file, indent=4)
    
    TandM_list = np.array([true_list, mprism_list]).T

    np.savetxt(os.path.join(source, f'TandM_list.txt'), TandM_list, fmt='%.4f')


## Draw the ROC and PR curves

metrics = [
    {"name": "MPrISM", "data_path": source + "TandM_list.txt", 
     "color": "green", "marker": "s", "linestyle": "--"},
    {"name": "PET", "data_path": source + "TandP_list.txt", 
     "color": "blue", "marker": "o", "linestyle": "-"},
    {"name": "PMCT", "data_path": source + "TandR_list.txt", 
     "color": "red", "marker": "^", "linestyle": "-."}
]

all_results = []

for metric in metrics:

    data = np.loadtxt(metric["data_path"])
    true_values = data[:, 0]
    pred_values = data[:, 1]
    thresholds = np.arange(0.0, 5.2, 0.1)
    results = []

    for threshold in thresholds:
        TP = 0  
        FP = 0  
        FN = 0  
        TN = 0  
        
        for true, pred in zip(true_values, pred_values):
            if true < 5.0:
                if pred < threshold:
                    TP += 1
                else:
                    FN += 1
            else:
                if pred < threshold:
                    FP += 1
                else:
                    TN += 1

        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 1
  
        results.append({
            'threshold': threshold,
            'FPR': FPR,
            'Recall': TPR,
            'Precision': precision
        })
    
    auc = 0
    for i in range(1, len(results)):
        x1, x2 = results[i-1]['FPR'], results[i]['FPR']
        y1, y2 = results[i-1]['Recall'], results[i]['Recall']
        auc += (x2 - x1) * (y1 + y2) / 2

    metric['results'] = results
    metric['auc'] = auc
    all_results.append(metric)


plt.figure(figsize=(10, 8))
for metric in all_results:
    fpr = [res['FPR'] for res in metric['results']]
    recall = [res['Recall'] for res in metric['results']]
    plt.plot(fpr, recall, 
             color=metric['color'], 
             marker=metric['marker'], 
             markersize=5,
             linestyle=metric['linestyle'],
             label=f"{metric['name']} (AUC = {metric['auc']:.4f})")

plt.xlabel('False Positive Rate (FPR)', fontsize=24)
plt.ylabel('True Positive Rate (Recall)', fontsize=24)
plt.grid(False)
plt.legend(fontsize=22, loc='lower right')
plt.xticks(np.arange(0, 1.1, 0.1),fontsize=22)
plt.yticks(np.arange(0, 1.1, 0.1),fontsize=22)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.tight_layout()
plt.savefig(figure_dir + 'ROC_Curves.png')
plt.show()

plt.figure(figsize=(10, 8))
for metric in all_results:
    recall = [res['Recall'] for res in metric['results']]
    precision = [res['Precision'] for res in metric['results']]
    plt.plot(recall, precision, 
             color=metric['color'], 
             marker=metric['marker'], 
             markersize=5,
             linestyle=metric['linestyle'],
             label=f"{metric['name']}")


plt.xlabel('True Positive Rate (Recall)', fontsize=24)
plt.ylabel('Positive Predictive Value (Precision)', fontsize=24)
plt.grid(False)

plt.xticks(np.arange(0, 1.1, 0.1),fontsize=20)
plt.yticks(np.arange(0, 1.1, 0.1),fontsize=20)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.tight_layout()
plt.savefig(figure_dir + 'PR_Curves.png')
plt.show()

print("\nAUC Values:")
for metric in all_results:
    print(f"{metric['name']}: {metric['auc']:.4f}")

plt.figure(figsize=(10, 7))
for metric in all_results:
    recall = [res['Recall'] for res in metric['results']]
    precision = [res['Precision'] for res in metric['results']]

    filtered_recall = []
    filtered_precision = []
    for r, p in zip(recall, precision):
        if r >= 0.85:
            filtered_recall.append(r)
            filtered_precision.append(p)

    
    plt.plot(filtered_recall, filtered_precision, 
             color=metric['color'], 
             marker=metric['marker'], 
             markersize=8,  
             linestyle=metric['linestyle'],
             linewidth=2.5,  
             label=f"{metric['name']}")


plt.legend(fontsize=24, loc='lower left')  
plt.xticks(np.arange(0.86, 1.01, 0.02), fontsize=24)  
plt.yticks(fontsize=24)
plt.xlim([0.845, 1.005])  
plt.ylim([-0.02, 0.65])  
plt.tight_layout()
plt.savefig(figure_dir + 'PR_Curves_Zoomed.png')
plt.show()

# Analyze the per-sample computation time of the safety metrics

with open(source + 'MPrISM_computation_time.json', 'r') as file:
        mprism_time_list = np.array(json.load(file))
        
with open(source + 'PET_computation_time.txt', 'r') as file:
        pet_time_list = np.loadtxt(file)
        
with open(source + 'PMCT_computation_time.json', 'r') as file:
        pmct_time_list = np.array(json.load(file))
        
print("the mean per-sample computation time of MPrISM is: ")
print(np.mean(mprism_time_list))

print("the std per-sample computation time of MPrISM is: ")
print(np.std(mprism_time_list))

print("the mean per-sample computation time of PET is: ")
print(np.mean(pet_time_list))

print("the std per-sample computation time of PET is: ")
print(np.std(pet_time_list))

print("the mean per-sample computation time of PMCT is: ")
print(np.mean(pmct_time_list))

print("the std per-sample computation time of PMCT is: ")
print(np.std(pmct_time_list))
        

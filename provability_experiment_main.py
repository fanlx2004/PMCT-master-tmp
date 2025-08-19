import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from utility_functions.sample_function import adjust_sampling
from safety_metric_tool.PMCT_solver import PMCT_solver
from utility_functions.deduction_function import deduction
from multiprocessing import Pool
from conf import conf
import math

'''
    This script performs the verification of the probability of the PMCT.
'''

# If you want to generate data again, set this to True
generate_data = False  

true_values = []
pred_values = []
epsilon = max(conf.epsilons)
beta = conf.beta1
N = math.ceil(np.log(beta) / np.log(1-epsilon)) #135

figure_dir = "./result/experiment_figure/provability_experiment/"
data_dir = "./result/provability_experiment_data/"

if generate_data:
    # Generate the prediction using the DRL agent, and generate the true values using the simulation environment
    partial_deduction = partial(deduction, get_time=True)
    for i in range(1,N+1):
        coodinate_list = adjust_sampling(1770)
        pred_values_0 = PMCT_solver(coodinate_list)
        pred_values.extend(pred_values_0)
        with Pool() as pool:
            true_values_0 = pool.map(partial_deduction, coodinate_list)
        true_values.extend(true_values_0)
        with open(data_dir + f'result_{i}/' + 'true_and_predicted_values_list.txt', 'w') as file:
            for true_val, pred_val in zip(true_values_0, pred_values_0):
                if true_val is not None and pred_val is not None:
                    file.write(f"{true_val} {pred_val}\n")
                    
    with open(data_dir + 'final_true_and_predicted_values_list.txt', 'w') as file:
        for true_val, pred_val in zip(true_values, pred_values):
                if true_val is not None and pred_val is not None:
                    file.write(f"{true_val} {pred_val}\n")
    
else:
    # Load the true and predicted values from the file
    
    with open(data_dir + 'final_true_and_predicted_values_list.txt', 'r') as file:
        for line in file:
            if line.strip() == '':
                continue
            try:
                true_val, pred_val = map(float, line.split())
                true_values.append(true_val)
                pred_values.append(pred_val)
            except:
                continue

# Draw the histogram of the true and predicted values

bins = np.arange(0, 5.1, 0.5)
bin_labels = [f'{x:.1f}' for x in bins]
positive_counts = np.zeros(len(bins))
negative_counts = np.zeros(len(bins))
for true, pred in zip(true_values, pred_values):
    bin_idx = int(round(pred / 0.5))
    bin_idx = min(bin_idx, len(bins)-1)
    
    if true >= pred:
        positive_counts[bin_idx] += 1
    else:
        negative_counts[bin_idx] += 1

plt.figure(figsize=(12, 9))

scale_factor = 10
bar_width = 0.25
label_offset = 500 

bars_positive = plt.bar(bins, positive_counts, width=bar_width, color="#3e9b10", alpha=0.7,
                       label='True Value ≥ Predicted Value')
bars_negative = plt.bar(bins, negative_counts * scale_factor, width=bar_width, bottom=positive_counts, 
                       color="#290645", alpha=0.7, label='True Value < Predicted Value')

threshold_lines = []
for i, (pos, neg) in enumerate(zip(positive_counts, negative_counts)):
    total = pos + neg * scale_factor
    if total > 0:
        threshold = total * (1-min(conf.epsilons))
        threshold_lines.append(plt.hlines(threshold, bins[i]-bar_width/2, bins[i]+bar_width/2, 
                                        colors="#8B3316", linestyles='dashed', linewidth=1.5))

for i, (pos, neg) in enumerate(zip(positive_counts, negative_counts)):
    total = pos + neg
    plt.text(bins[i], pos + neg * scale_factor + max(positive_counts)*0.01,  
            f'{int(total)}', 
            ha='center', va='bottom', fontsize=19)

for i, neg in enumerate(negative_counts):

    line_start_x = bins[i] + bar_width/2
    line_start_y = positive_counts[i] + neg * scale_factor * 0.5

    label_y = line_start_y - label_offset
    label_x = line_start_x + 0.04
    
    plt.text(label_x, label_y, f'{int(neg)}', ha='left', va='top', color='black', fontsize=19)
    
    plt.plot([line_start_x, label_x], [line_start_y, label_y], 
             color='black', linewidth=1)

threshold_percent = round(min(conf.epsilons) * 100)

if threshold_lines:
    threshold_lines[0].set_label(f'Failure Rate Threshold ({threshold_percent}%)')

max_height = max([p + n * scale_factor for p, n in zip(positive_counts, negative_counts)])
plt.ylim(0, max_height * 1.1)
plt.yticks(fontsize=20)
plt.xticks(bins, bin_labels, fontsize=20)
plt.xlabel('Predicted Value of Collision Time (s)', fontsize=22)
plt.ylabel('Number', fontsize=22)
plt.legend(fontsize=23)
plt.tight_layout()
plt.savefig(figure_dir + 'true_and_predicted_values', dpi=300)

# Draw the line chart of the failure rate by predicted time

plt.figure(figsize=(11, 9))

failure_rates = []
for i in range(len(bins)):
    total = positive_counts[i] + negative_counts[i]
    if total > 0:
        failure_rate = (negative_counts[i] / total) * 100
    else:
        failure_rate = 0
    failure_rates.append(failure_rate)

plt.plot(bins, failure_rates, 'bo-', linewidth=2.5, markersize=9, label='Failure Rate')

plt.hlines(y=threshold_percent, xmin=0.0, xmax=5.0, 
           colors='r', linestyles='--', linewidth=2.5, 
           label=f'Failure Rate Threshold ({threshold_percent}%)')

for i, rate in enumerate(failure_rates):
    plt.text(bins[i], rate + 0.1, f'{rate:.2f}%', 
             ha='center', va='bottom', fontsize=20)

plt.xticks(bins, bin_labels, fontsize=22)
plt.yticks(fontsize=22)
plt.xlabel('Predicted Value of Collision Time (s)', fontsize=22)
plt.ylabel('Failure Rate (%)', fontsize=22)
plt.legend(loc='center right', fontsize=23)
plt.ylim(-0.2, 5.2)
plt.tight_layout()

plt.savefig(figure_dir + 'failure_rate_by_predicted_time.png', dpi=300)


# calculate the probability of true collision time ≥ predicted value to verify the provability

successful_rate_list = []

for i in range(1,N+1):
    with open(data_dir + f'result_{i}/' + 'true_and_predicted_values_list.txt') as file:
        tandp_data = np.loadtxt(file, delimiter=' ', dtype=float)
    n = tandp_data.shape[0]
    m = 0
    for j in range(n):
        if tandp_data[j][1] <= tandp_data[j][0]:
            m += 1
    successful_rate = m / n
    successful_rate_list.append(successful_rate)

with open(data_dir + 'successful_rate_list.txt', 'w') as file:
    for item in successful_rate_list:
        file.write(f"{item}\n")
        
successful_rate_array = np.array(successful_rate_list)
mean_rate = np.mean(successful_rate_array)
std_rate = np.std(successful_rate_array)
max_rate = np.max(successful_rate_array)
min_rate = np.min(successful_rate_array)
print("Mean rate:", mean_rate)
print("Std rate:", std_rate)
print("Max rate:", max_rate)
print("Min rate:", min_rate)

if min_rate >= epsilon:
    print("The provability of PMCT is verified successfully.")
else:
    print("The verification fails.")
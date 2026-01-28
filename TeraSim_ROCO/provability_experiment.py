import numpy as np
import matplotlib.pyplot as plt
import os
import random
import json
import torch
from collections import Counter, defaultdict
from tools.PMCTNetwork_attention import PMCTNetwork_attention

accident_mode = 'road' # 'road' or 'cross'

do_calculate = False  
cache_dir = 'data/provability_data/'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
data_cache_file = os.path.join(cache_dir, f'pred_results_{accident_mode}.json')



def _parse_sample(item):
    if isinstance(item, dict) and 'st' in item and 'st1' in item and 'rt' in item:
        st, st1, rt = item['st'], item['st1'], item['rt']
    elif isinstance(item, list) and len(item) == 3:
        st, st1, rt = item
    else:
        return None
    
    if (isinstance(st, list) and len(st) == 33 and 
        isinstance(st1, list) and len(st1) == 33 and 
        isinstance(rt, (int, float)) and 1 <= rt <= 11):
        return np.array(st, dtype=np.float32), np.array(st1, dtype=np.float32), int(rt)
    return None

epsilon = 0.05
N = 135
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if accident_mode == 'road':
        figure_dir = 'figures/TeraSim_road_figure/'
        data_test_dir = '/home/lingxiang/terasim_data/test_road'
        checkpoint = torch.load('model_params/saved_pmct_models_onlyroad/best_model_attention.pth', map_location=device)
        num_sample = 1000
        
elif accident_mode == 'cross':
        figure_dir = 'figures/TeraSim_cross_figure/'
        data_test_dir = '/home/lingxiang/terasim_data/test_cross'
        checkpoint = torch.load('model_params/saved_pmct_models_cross/best_model_attention.pth', map_location=device)
        num_sample = 1500
        
if do_calculate:
    data_file_list = os.listdir(data_test_dir)
    model = PMCTNetwork_attention().to(device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else :
        model.load_state_dict(checkpoint)


    sampled_data = []
    sampled_label = []
    for j in range(num_sample):
        file_path = random.choice(data_file_list)

        
        data = []
        attempts = 0
        max_attempts = 10 
            
        while data == [] and attempts < max_attempts:
            try:
                with open(os.path.join(data_test_dir, file_path), 'r') as f:
                    data = json.load(f)
                break
            except Exception as e:
                attempts += 1
                file_path = random.choice(data_file_list)
        
        if data == []:
            continue

        samples_by_label = defaultdict(list)
        valid_samples = []
        total_items = 0
        valid_count = 0
        
        
        for idx, item in enumerate(data):
            total_items += 1
            sample = _parse_sample(item)
            if sample:
                st, st1, rt = sample
                samples_by_label[rt].append((st, st1, rt))
                valid_samples.append(((st, st1, rt), rt))
                valid_count += 1
        
    
        if samples_by_label:
            
            min_count = float('inf')
            valid_labels = []
            
            for label, samples in samples_by_label.items():
                if len(samples) > 0:
                    min_count = min(min_count, len(samples))
                    valid_labels.append(label)
            
            
            if min_count == 0:
                continue
            elif min_count == float('inf'):
                continue
            

            for label in valid_labels:
                samples = samples_by_label[label]
                selected = random.sample(samples, min_count)
                for sample in selected:
                    sampled_data.append(sample[0]) 
                    sampled_label.append(label)




    sampled_data = np.array(sampled_data)
    print(f"sampled_data shape: {sampled_data.shape}")
    st_tensor = torch.from_numpy(sampled_data).float().to(device)
    model.eval() 
    with torch.no_grad():
        q_st, msc_pred = model(st_tensor)


    q_st, msc_pred = model(st_tensor)
    msc_pred = (msc_pred - 1) * 0.5
    pred_values = msc_pred.cpu().tolist()
    true_values = (np.array(sampled_label) - 1) * 0.5
    true_values = true_values.tolist()
    
    cache_data = {
        'true_values': true_values,
        'pred_values': pred_values
    }
    with open(data_cache_file, 'w') as f:
        json.dump(cache_data, f)
    print(f"Results saved to {data_cache_file}")


else:
    if os.path.exists(data_cache_file):
        with open(data_cache_file, 'r') as f:
            cache_data = json.load(f)
        true_values = cache_data['true_values']
        pred_values = cache_data['pred_values']
        print(f"Results loaded from {data_cache_file}. Skipping model inference.")
    else:
        raise FileNotFoundError(f"No cache file found at {data_cache_file}. Please set do_calculate=True first.")


bins = np.arange(0, 5.1, 0.5)
bin_labels = [f'{x:.1f}' for x in bins]


positive_counts = np.zeros(len(bins))  
negative_counts = np.zeros(len(bins))  

for true, pred in zip(true_values, pred_values):

    bin_idx = int(round(true / 0.5))
    bin_idx = min(bin_idx, len(bins)-1)

    if true >= pred:
        positive_counts[bin_idx] += 1
    else:
        negative_counts[bin_idx] += 1

plt.figure(figsize=(16, 10))


font_large = 26
font_med = 22
font_small = 18
scale_factor = 1
bar_width = 0.25


bars_positive = plt.bar(bins, positive_counts, width=bar_width, color="#3e9b10", alpha=0.7,
                       label='True Value ≥ Predicted Value')
bars_negative = plt.bar(bins, negative_counts * scale_factor, width=bar_width, bottom=positive_counts, 
                       color="#290645", alpha=0.7, label='True Value < Predicted Value')


threshold_lines = []
for i, (pos, neg) in enumerate(zip(positive_counts, negative_counts)):
    total = pos + neg * scale_factor
    if total > 0:
        threshold = total * (1-epsilon)
        threshold_lines.append(plt.hlines(threshold, bins[i]-bar_width/2, bins[i]+bar_width/2, 
                                        colors="#8B3316", linestyles='dashed', linewidth=1.5))


totals = positive_counts + negative_counts
total_max = totals.max() if totals.size > 0 else 1
for i, (pos, neg) in enumerate(zip(positive_counts, negative_counts)):
    total = pos + neg
    plt.text(bins[i], pos + neg * scale_factor + total_max * 0.01,
             f'{int(total)}',
             ha='center', va='bottom', fontsize=font_small)

label_offset = max(5.0, total_max * 0.06)
for i, neg in enumerate(negative_counts):
    line_start_x = bins[i] + bar_width/2
    line_start_y = positive_counts[i] + neg * scale_factor * 0.5

    label_y = line_start_y - label_offset
    label_x = line_start_x + 0.027

    plt.text(label_x, label_y, f'{int(neg)}', ha='left', va='top', color='black', fontsize=font_small)

    plt.plot([line_start_x, label_x], [line_start_y, label_y],
             color='black', linewidth=1)

threshold_percent = epsilon * 100

if threshold_lines:
    threshold_lines[0].set_label(f'Misclassification Rate Threshold ({threshold_percent}%)')

max_height = max([p + n * scale_factor for p, n in zip(positive_counts, negative_counts)])
plt.ylim(0, max_height * 1.4)
plt.yticks(fontsize=font_med)
plt.xticks(bins, bin_labels, fontsize=font_med)
plt.xlabel('True Value of Collision Time (s)', fontsize=font_large)  
plt.ylabel('Number of Samples', fontsize=font_large)  

plt.legend(fontsize=font_med)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(figure_dir + 'true_and_predicted_values_swapped.png', dpi=200, bbox_inches='tight')


plt.figure(figsize=(13,9))

misclassification_rates = []
for i in range(len(bins)):
    total = positive_counts[i] + negative_counts[i]
    if total > 0:
        misclassification_rate = (negative_counts[i] / total) * 100  
    else:
        misclassification_rate = 0
    misclassification_rates.append(misclassification_rate)

plt.plot(bins, misclassification_rates, 'bo-', linewidth=2.0, markersize=8, label='Misclassification Rate')

plt.hlines(y=threshold_percent, xmin=0.0, xmax=5.0,
           colors='r', linestyles='--', linewidth=2.0,
           label=f'Misclassification Rate Threshold ({threshold_percent}%)')

for i, rate in enumerate(misclassification_rates):
    plt.text(bins[i], rate + 0.2, f'{rate:.2f}%', ha='center', va='bottom', fontsize=font_small)

plt.xticks(bins, bin_labels, fontsize=font_med)
plt.yticks(fontsize=font_med)
plt.xlabel('True Value of Collision Time (s)', fontsize=font_large)  
plt.ylabel('Misclassification Rate (%)', fontsize=font_large)
plt.legend(loc='best', fontsize=font_med)
plt.ylim(-0.2, 6)
plt.tight_layout(rect=[0, 0, 0.85, 1])

plt.savefig(figure_dir + 'misclassification_rate_by_true_value.png', dpi=200, bbox_inches='tight')


for i in range(len(bins)):
    if i == len(bins)-1:
        continue
    lower = bins[i]
    upper = bins[i+1]
    total = positive_counts[i] + negative_counts[i]

total_samples = len(true_values)
correct_samples = sum(1 for t, p in zip(true_values, pred_values) if t >= p)
error_samples = total_samples - correct_samples
overall_error_rate = error_samples / total_samples * 100


if overall_error_rate <= epsilon * 100:
    print(f"\n✓ misclassification rate ({overall_error_rate:.2f}%) ≤ threshold ({epsilon*100:.1f}%)")
else:
    print(f"\n✗ misclassification rate ({overall_error_rate:.2f}%) > threshold ({epsilon*100:.1f}%)")

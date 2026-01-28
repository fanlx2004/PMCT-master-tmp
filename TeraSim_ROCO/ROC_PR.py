import numpy as np
import matplotlib.pyplot as plt
import os
import random
import json
import torch
import time
from collections import Counter, defaultdict
from tools.PMCTNetwork_attention import PMCTNetwork_attention
from tools.metric_solver import PET_solver, TTC_solver, ACT_solver, TA_solver, TwoD_TTC_solver, MPrISM_solver

plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16
})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accident_mode = 'road'  # 'road' or 'cross'
do_calculate = False

base_direct = 'data/ROC_PR_data/'

TTC_dir = base_direct + "TandTTC_list_" + accident_mode + ".txt"
PET_dir = base_direct + "TandP_list_" + accident_mode + ".txt"
PMCT_dir = base_direct + "TandR_list_" + accident_mode + ".txt"
mprism_dir = base_direct + "TandM_list_" + accident_mode + ".txt"
ACT_dir = base_direct + "TandA_list_" + accident_mode + ".txt"
TA_dir = base_direct + "TandTA_list_" + accident_mode + ".txt"
TwoDTTC_dir = base_direct + "Tand2DTTC_list_" + accident_mode + ".txt"


if accident_mode == 'road':
    figure_dir = 'figures/TeraSim_road_figure/'
elif accident_mode == 'cross':
    figure_dir = 'figures/TeraSim_cross_figure/'


if do_calculate:

    with open(TTC_dir, 'w') as f:
        pass
    with open(PET_dir, 'w') as f:
        pass
    with open(PMCT_dir, 'w') as f:
        pass
    with open(ACT_dir, 'w') as f:
        pass
    with open(TA_dir, 'w') as f:
        pass
    with open(TwoDTTC_dir, 'w') as f:
        pass
    
    #with open(mprism_dir, 'w') as f:
    #    pass 
    
    def _parse_sample(item):
        if isinstance(item, dict) and 'st' in item and 'st1' in item and 'rt' in item:
            st, st1, rt = item['st'], item['st1'], item['rt']
        elif isinstance(item, list) and len(item) == 3:
            st, st1, rt = item
        else:
            return None
        
        if (isinstance(st, list) and len(st) == 40 and 
            isinstance(st1, list) and len(st1) == 40 and 
            isinstance(rt, (int, float)) and 1 <= rt <= 11):
            return np.array(st, dtype=np.float32), np.array(st1, dtype=np.float32), int(rt)
        return None

    epsilon = 0.05
    N = 135
    
    if accident_mode == 'road':
        data_test_dir = 'data/terasim_data/test_road'
        model = PMCTNetwork_attention().to(device)
        sample_num = 500
        checkpoint = torch.load('model_params/saved_pmct_models_onlyroad/best_model_attention.pth', map_location=device)
    else:
        data_test_dir = 'data/terasim_data/test_cross'
        model = PMCTNetwork_attention().to(device)
        sample_num = 750
        checkpoint = torch.load('model_params/saved_pmct_models_cross/best_model_attention.pth', map_location=device)

    data_file_list = os.listdir(data_test_dir)


    print("=" * 50)

    sampled_data = []
    sampled_label = []
    for j in range(sample_num):
        
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

            for label in sorted(samples_by_label.keys()):
                count = len(samples_by_label[label])

            
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
                if label == 11:
                    try:
                        selected = random.sample(samples, min_count*3)
                    except:
                        selected = random.sample(samples, min_count)
                for sample in selected:
                    sampled_data.append(sample[0]) 
                    sampled_label.append(label)
    

        

    print("\n" + "=" * 50)
    print(f"Sample result: {len(sampled_data)} data")


    sampled_data = np.array(sampled_data)
    
    positions = [3, 9, 15, 21, 27, 33, 39]
    mask = np.ones(sampled_data.shape[1], dtype=bool)
    mask[positions] = False
    sampled_data_PMCT = sampled_data[:, mask]
    st_tensor = torch.from_numpy(sampled_data_PMCT).float().to(device)
    print(f"st_tensor shape: {st_tensor.shape}, device: {st_tensor.device}")


    if 'model_state_dict' in checkpoint :
        model.load_state_dict(checkpoint['model_state_dict'])
    else : 
        model.load_state_dict(checkpoint)
    
    model.eval()

    start_time = time.time()
    q_st, msc_pred = model(st_tensor)
    end_time = time.time()
    print(f"PMCT compute time: {end_time - start_time:.6f} s")
    msc_pred = (msc_pred - 1) * 0.5
    pred_values = msc_pred.cpu().tolist()
    true_values = (np.array(sampled_label) - 1) * 0.5
    true_values = true_values.tolist()


    for i in range(len(true_values)):
        with open(PMCT_dir, 'a') as f:
            t = true_values[i]
            r = pred_values[i]
            f.write(f'{t:.2f}')
            f.write(' ')
            f.write(f'{r:.2f}\n')

    start_time = time.time()
    pred_values_pet = PET_solver(sampled_data.tolist())
    end_time = time.time()
    print(f"PET compute time: {end_time - start_time:.6f} s")

    for i in range(len(true_values)):
        with open(PET_dir, 'a') as f:
            t = true_values[i]
            r = pred_values_pet[i]
            f.write(f'{t:.2f}')
            f.write(' ')
            f.write(f'{r:.2f}\n')

    start_time = time.time()
    pred_values_ttc = TTC_solver(sampled_data.tolist())
    end_time = time.time()
    print(f"TTC compute time: {end_time - start_time:.6f} s")

    for i in range(len(true_values)):
        with open(TTC_dir, 'a') as f:
            t = true_values[i]
            r = pred_values_ttc[i]
            f.write(f'{t:.2f}')
            f.write(' ')
            f.write(f'{r:.2f}\n')
    
    start_time = time.time()
    pred_values_ACT = ACT_solver(sampled_data.tolist())
    end_time = time.time()
    print(f"ACT compute time: {end_time - start_time:.6f} s")
    
    for i in range(len(true_values)):
        with open(ACT_dir, 'a') as f:
            t = true_values[i]
            r = pred_values_ACT[i]
            f.write(f'{t:.2f}')
            f.write(' ')
            f.write(f'{r:.2f}\n')
    
    start_time = time.time()
    pred_values_TA = TA_solver(sampled_data.tolist())
    end_time = time.time()
    print(f"TA compute time: {end_time - start_time:.6f} s")
    
    for i in range(len(true_values)):
        with open(TA_dir, 'a') as f:
            t = true_values[i]
            r = pred_values_TA[i]
            f.write(f'{t:.2f}')
            f.write(' ')
            f.write(f'{r:.2f}\n')
    
    start_time = time.time()
    pred_values_2D_TTC = TwoD_TTC_solver(sampled_data.tolist())
    end_time = time.time()
    print(f"2D-TTC compute time: {end_time - start_time:.6f} s")
    
    for i in range(len(true_values)):
        with open(TwoDTTC_dir, 'a') as f:
            t = true_values[i]
            r = pred_values_2D_TTC[i]
            f.write(f'{t:.2f}')
            f.write(' ')
            f.write(f'{r:.2f}\n')
            
    start_time = time.time()
    sample_list = sampled_data.tolist()
    pred_values_mprism = []
    for ooo in range(len(sample_list)):
        pred_values_mprism.append(MPrISM_solver(sample_list[ooo]))
    end_time = time.time()
    print(f"MPrISM compute time: {end_time - start_time:.6f} s")
    
    for i in range(len(true_values)):
        with open(mprism_dir, 'a') as f:
            t = true_values[i]
            r = pred_values_mprism[i]
            f.write(f'{t:.2f}')
            f.write(' ')
            f.write(f'{r:.2f}\n')




metrics = [
    {"name": "PMCT", "data_path": PMCT_dir, 
     "color": "red", "marker": "^", "linestyle": "-."},
    {"name": "TTC", "data_path": TTC_dir, 
     "color": "green", "marker": "s", "linestyle": "--"},
    {"name": "PET", "data_path": PET_dir, 
     "color": "blue", "marker": "o", "linestyle": "-"},
    {"name": "ACT", "data_path": ACT_dir,
     "color": "orange", "marker": "d", "linestyle": ":"},
    {"name": "TA", "data_path": TA_dir,
     "color": "brown", "marker": "v", "linestyle": "--"},
    {"name": "2D-TTC", "data_path": TwoDTTC_dir,
     "color": "cyan", "marker": "*", "linestyle": "-."},
    {"name": "MPRISM", "data_path": mprism_dir,
     "color": "purple", "marker": "x", "linestyle": ":"}
]

all_results = []

    
for metric in metrics:
    if metric["name"] == "MPRISM":
        mprism_raw = np.loadtxt(metric["data_path"])
        mprism_by_label = defaultdict(list)
        for entry in mprism_raw:
            mprism_by_label[round(float(entry[0]), 2)].append(entry[1])
        
        sampled_counts = Counter([round(float(t), 2) for t in true_values])
        
        ratios = []
        for val, count in sampled_counts.items():
            available = len(mprism_by_label.get(val, []))
            if count > 0:
                ratios.append(available / count)
        
        scale_factor = min(ratios) if ratios else 0
        
        final_true = []
        final_pred = []
        for val, count in sampled_counts.items():
            num_to_keep = int(count * scale_factor)
            samples = mprism_by_label.get(val, [])
            if num_to_keep > 0 and samples:
                selected = random.sample(samples, num_to_keep)
                final_true.extend([val] * num_to_keep)
                final_pred.extend(selected)
        
        true_values = np.array(final_true)
        pred_values = np.array(final_pred)
        

    else:
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


plt.figure(figsize=(12, 9))
for metric in all_results:
    fpr = [res['FPR'] for res in metric['results']]
    recall = [res['Recall'] for res in metric['results']]
    plt.plot(fpr, recall, 
             color=metric['color'], 
             marker=metric['marker'], 
             markersize=5,
             linestyle=metric['linestyle'],
             label=f"{metric['name']} (AUC = {metric['auc']:.4f})")

plt.xlabel('False Positive Rate (FPR)', fontsize=28)
plt.ylabel('True Positive Rate (Recall)', fontsize=28)
plt.grid(False)
plt.legend(fontsize=24, loc='lower right')
plt.xticks(np.arange(0, 1.1, 0.1), fontsize=26)
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=26)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.tight_layout()
plt.savefig(figure_dir + 'ROC_Curves.png')
plt.show()

plt.figure(figsize=(12, 9))
for metric in all_results:
    recall = [res['Recall'] for res in metric['results']]
    precision = [res['Precision'] for res in metric['results']]
    plt.plot(recall, precision, 
             color=metric['color'], 
             marker=metric['marker'], 
             markersize=5,
             linestyle=metric['linestyle'],
             label=f"{metric['name']}")


plt.xlabel('True Positive Rate (Recall)', fontsize=28)
plt.ylabel('Positive Predictive Value (Precision)', fontsize=28)
plt.grid(False)

plt.xticks(np.arange(0, 1.1, 0.1), fontsize=26)
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=26)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.tight_layout()
plt.legend(fontsize=24, loc='lower right')
plt.savefig(figure_dir + 'PR_Curves.png')
plt.show()

print("\nAUC Values:")
for metric in all_results:
    print(f"{metric['name']}: {metric['auc']:.4f}")





import json
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import torch
from tools.metric_solver import PET_solver, TTC_solver, TA_solver, ACT_solver, TwoD_TTC_solver, MPrISM_solver 
from tools.PMCTNetwork_attention import PMCTNetwork_attention


do_calculate = False

plt.rcParams.update({
    'font.size': 24,
    'axes.titlesize': 26,
    'axes.labelsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 22
})

def is_in_front(dx, dy, heading_deg):

    h_rad = math.radians(heading_deg)
    forward_x = math.sin(h_rad)
    forward_y = math.cos(h_rad)
    
    projection = dx * forward_x + dy * forward_y
    
    if projection >= 0:
        return 1   
    elif projection < 0:
        return -1 

def reorder_vehicles(vector_33):

    empty_vehicle = [120.0, 120.0, 0.0, 0.0, 0.0]
    
    ego_heading = vector_33[2]
    
    background_cars = []
    for i in range(6):
        start_idx = 3 + i * 5
        car_info = list(vector_33[start_idx:start_idx+5])
        dist = math.sqrt(car_info[0]**2 + car_info[1]**2)
        background_cars.append({
            'info': car_info,
            'dist': dist,
            'idx': i
        })
    
    background_cars.sort(key=lambda x: x['dist'])
    
    front_cars = [] 
    rear_cars = []
    
    for car in background_cars:
        dx, dy = car['info'][0], car['info'][1]
        pos = is_in_front(dx, dy, ego_heading)
        
        if pos >= 0: 
            front_cars.append(car)
        else:
            rear_cars.append(car)

    front_cars.sort(key=lambda x: x['dist'])
    rear_cars.sort(key=lambda x: x['dist'])
    
    front_cars = front_cars[:3]
    rear_cars = rear_cars[:3]
    
    result = list(vector_33[:3])  
    

    new_background = []
    
    for i in range(3):
        if i < len(front_cars):
            new_background.append(front_cars[i]['info'])
        else:
            new_background.append(empty_vehicle)
    
    for i in range(3):
        if i < len(rear_cars):
            new_background.append(rear_cars[i]['info'])
        else:
            new_background.append(empty_vehicle)
    

    final_background = [None] * 6
    final_background[0] = new_background[0]  
    final_background[1] = new_background[3]  
    final_background[2] = new_background[1]  
    final_background[3] = new_background[4]  
    final_background[4] = new_background[2]  
    final_background[5] = new_background[5]  

    for car_info in final_background:
        result.extend(car_info)
    
    return result

if do_calculate :
    save_dir = 'figures/ROCO_figure/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    collision_folder = ['data/ROCO_collision_data/real_data_collision1/', 'data/ROCO_collision_data/real_data_collision2/', 'data/ROCO_collision_data/real_data_collision3/']

    collision_ending = [35, 14, 43]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_road = PMCTNetwork_attention().to(device)
    checkpoint_road = torch.load('model_params/saved_pmct_models_onlyroad/best_model_attention.pth', map_location=device)
    if 'model_state_dict' in checkpoint_road:   
        model_road.load_state_dict(checkpoint_road['model_state_dict'])
    else:
        model_road.load_state_dict(checkpoint_road)
    model_road.eval()

    model_cross = PMCTNetwork_attention().to(device)    
    checkpoint_cross = torch.load('model_params/saved_pmct_models_cross/best_model_attention.pth', map_location=device)
    if 'model_state_dict' in checkpoint_cross:   
        model_cross.load_state_dict(checkpoint_cross['model_state_dict'])
    else:
        model_cross.load_state_dict(checkpoint_cross)
    model_cross.eval()

    collision_true = [[], [], []]

    for i in range(3):
        for j in range(0, collision_ending[i] + 1):
            collision_true[i].append(max(min(5.0, (collision_ending[i]-j)*0.4),0.0))

    collision_data = [[[],[]], [[],[]], [[],[]]]

    for i in range(3):
        for j in range(0, collision_ending[i] + 1):
            data_path_veh1 = os.path.join(collision_folder[i], f'veh1_{j}.json')
            data_path_veh2 = os.path.join(collision_folder[i], f'veh2_{j}.json')
            with open(data_path_veh1, 'r') as f:
                data_veh1 = json.load(f)
            with open(data_path_veh2, 'r') as f:
                data_veh2 = json.load(f)
            
            if len(data_veh1) == 2:
                collision_data[i][0].append(data_veh1[0])
            else:
                collision_data[i][0].append(data_veh1)
            if len(data_veh2) == 2:
                collision_data[i][1].append(data_veh2[0])
            else:   
                collision_data[i][1].append(data_veh2)

    predicted_times_PMCT = [[[],[]], [[],[]], [[],[]]]

    for i in range(3):
        for j in range(2):
            input_data = collision_data[i][j]
            for k in range(len(input_data)):
                data_original = collision_data[i][j][k]
                heading_ego = data_original[2]
                heading_nearest = data_original[7]
                heading_diff = abs(heading_ego - heading_nearest)
                if heading_diff < 120 and heading_diff > 60:
                    print('Crossing Scenario detected, use Cross Model prediction.')
                    with torch.no_grad():
                        _, msc_pred_cross = model_cross(torch.tensor(data_original).float().unsqueeze(0).to(device))
                    msc_pred = msc_pred_cross.cpu().numpy().item()
                    predicted_times_PMCT[i][j].append(msc_pred)
                else:
                    data_original = collision_data[i][j][k]
                    with torch.no_grad():
                        _, msc_pred_road = model_road(torch.tensor(data_original).float().unsqueeze(0).to(device))
                    msc_pred = msc_pred_road.cpu().numpy().item()
                    predicted_times_PMCT[i][j].append(msc_pred)
            

    plot_bound = [[22, 35],[1, 14],[30, 43]]
    PMCT_predicted = [[],[],[]]
    for i in range(3):
        raw0 = predicted_times_PMCT[i][0][plot_bound[i][0]:plot_bound[i][1]+1]
        raw1 = predicted_times_PMCT[i][1][plot_bound[i][0]:plot_bound[i][1]+1]
        PMCT_predicted[i].append([ (v - 1) * 0.5 for v in raw0 ])
        PMCT_predicted[i].append([ (v - 1) * 0.5 for v in raw1 ])



    PET_results = [[[], []] for _ in range(3)]
    TTC_results = [[[], []] for _ in range(3)]
    ACT_results = [[[], []] for _ in range(3)]
    TA_results = [[[], []] for _ in range(3)]
    TwoD_TTC_results = [[[], []] for _ in range(3)]
    MPrISM_results = [[[], []] for _ in range(3)]


    def _angle_diff_deg(a, b):
        d = (b - a + 180.0) % 360.0 - 180.0
        return d


    def make_40_state(collision_list, idx):

        yaw_positions = {3: 'ego', 9: 'bg1', 15: 'bg2', 21: 'bg3', 27: 'bg4', 33: 'bg5', 39: 'bg6'}

        full = collision_list
        orig = full[idx]

        if idx - 1 >= 0 and idx + 1 < len(full):
            h_prev = full[idx - 1][2]
            h_next = full[idx + 1][2]
            yaw_ego = _angle_diff_deg(h_prev, h_next) / 0.8
        else:
            yaw_ego = 0.0


        try:
            if idx - 1 >= 0 and idx + 1 < len(full):
                hb_prev = full[idx - 1][7]
                hb_next = full[idx + 1][7]
                yaw_bg1 = _angle_diff_deg(hb_prev, hb_next) / 0.8
            else:
                yaw_bg1 = 0.0
        except Exception:
            yaw_bg1 = 0.0

        yaw_map = {
            3: yaw_ego,
            9: yaw_bg1,
            15: 0.0,
            21: 0.0,
            27: 0.0,
            33: 0.0,
            39: 0.0
        }

        st40 = []
        orig_idx = 0
        for p in range(40):
            if p in yaw_map:
                st40.append(yaw_map[p])
            else:
                if orig_idx < len(orig):
                    st40.append(orig[orig_idx])
                else:
                    st40.append(0.0)
                orig_idx += 1

        return st40

    for i in range(3):
            lb, ub = plot_bound[i][0], plot_bound[i][1]
            for j in range(2):
                seq = collision_data[i][j][lb:ub+1]
                if not seq:
                    PET_results[i][j] = []
                    TTC_results[i][j] = []
                    ACT_results[i][j] = []
                    TA_results[i][j] = []
                    TwoD_TTC_results[i][j] = []
                    continue

                try:
                    PET_results[i][j] = PET_solver(seq)
                except Exception as e:
                    print(f"PET_solver error for collision {i} vehicle {j}:", e)
                    PET_results[i][j] = []

                try:
                    TTC_results[i][j] = TTC_solver(seq)
                except Exception as e:
                    print(f"TTC_solver error for collision {i} vehicle {j}:", e)
                    TTC_results[i][j] = []

                try:
                    ACT_results[i][j] = ACT_solver(seq)
                except Exception as e:
                    print(f"ACT_solver error for collision {i} vehicle {j}:", e)
                    ACT_results[i][j] = []

                try:
                    TA_results[i][j] = TA_solver(seq)
                except Exception as e:
                    print(f"TA_solver error for collision {i} vehicle {j}:", e)
                    TA_results[i][j] = []

                try:
                    TwoD_TTC_results[i][j] = TwoD_TTC_solver(seq)
                except Exception as e:
                    print(f"TwoD_TTC_solver error for collision {i} vehicle {j}:", e)
                    TwoD_TTC_results[i][j] = []

                try:
                    mpr_list = []
                    full_list = collision_data[i][j]
                    for t_local in range(len(seq)):
                        print("--------------------------")
                        print(f"MPrISM: Number {t_local}")
                        idx_global = lb + t_local
                        st40 = make_40_state(full_list, idx_global)
                        try:
                            mpr = MPrISM_solver(st40)
                        except Exception as e:
                            print(f"MPrISM single error idx {idx_global}:", e)
                            mpr = 5.0

                        if mpr is None:
                            mpr = 5.0
                        mpr_list.append(max(min(mpr, 5.0), 0.0))
                    MPrISM_results[i][j] = mpr_list
                except Exception as e:
                    print(f"MPrISM_solver error for collision {i} vehicle {j}:", e)
                    MPrISM_results[i][j] = []

                print(f'Collision {i+1}, Vehicle {j+1}: PET {len(PET_results[i][j])} TTC {len(TTC_results[i][j])} ACT {len(ACT_results[i][j])} TA {len(TA_results[i][j])} 2D-TTC {len(TwoD_TTC_results[i][j])}')

            try:
                acc_dict = {}
                for jj in range(2):
                    key_j = f'veh{jj+1}'
                    acc_dict[key_j] = {
                        'PET': list(map(float, PET_results[i][jj])) if PET_results[i][jj] else [],
                        'TTC': list(map(float, TTC_results[i][jj])) if TTC_results[i][jj] else [],
                        'ACT': list(map(float, ACT_results[i][jj])) if ACT_results[i][jj] else [],
                        'TA':  list(map(float, TA_results[i][jj])) if TA_results[i][jj] else [],
                        '2D-TTC': list(map(float, TwoD_TTC_results[i][jj])) if TwoD_TTC_results[i][jj] else [],
                        'MPrISM': list(map(float, MPrISM_results[i][jj])) if MPrISM_results[i][jj] else [],
                        'PMCT': list(map(float, PMCT_predicted[i][jj])) if (i < len(PMCT_predicted) and PMCT_predicted[i][jj]) else []
                    }
                out_acc_path = os.path.join(save_dir, f'collision_{i+1}_metrics.json')
                with open(out_acc_path, 'w') as af:
                    json.dump(acc_dict, af, indent=2)
                print(f'Saved per-accident metrics to {out_acc_path}')
            except Exception as e:
                print(f'Error saving per-accident metrics for collision {i+1}:', e)

else :
    save_dir = 'figures/ROCO_figure/'
    json_path = os.path.join(save_dir, 'accident_predictions_all.json')

    PET_results = [[[], []] for _ in range(3)]
    TTC_results = [[[], []] for _ in range(3)]
    ACT_results = [[[], []] for _ in range(3)]
    TA_results = [[[], []] for _ in range(3)]
    TwoD_TTC_results = [[[], []] for _ in range(3)]
    MPrISM_results = [[[], []] for _ in range(3)]
    PMCT_predicted = [[[], []] for _ in range(3)]

    try:
        with open(json_path, 'r') as jf:
            all_predictions = json.load(jf)
    except Exception as e:
        print(f"Failed to load {json_path}: {e}")
        all_predictions = {}

    for i in range(3):
        col_key = f'collision_{i+1}'
        col_data = all_predictions.get(col_key, {})
        for j in range(2):
            veh_key = f'veh{j+1}'
            veh_data = col_data.get(veh_key, {})
            PET_results[i][j] = veh_data.get('PET', []) or []
            TTC_results[i][j] = veh_data.get('TTC', []) or []
            ACT_results[i][j] = veh_data.get('ACT', []) or []
            TA_results[i][j] = veh_data.get('TA', []) or []
            TwoD_TTC_results[i][j] = veh_data.get('2D-TTC', []) or []
            MPrISM_results[i][j] = veh_data.get('MPrISM', []) or []
            PMCT_predicted[i][j] = veh_data.get('PMCT', []) or []

    print(f"Loaded predictions from {json_path}")

metric_names = ['PMCT', 'PET', 'TTC', 'ACT', 'TA', '2D-TTC', 'MPrISM']

style_map = {
    'PMCT':   {'color': 'black',   'linestyle': '-',  'marker': '*', 'linewidth': 3.5, 'markersize': 10},
    'PET':    {'color': 'tab:blue', 'linestyle': '-.', 'marker': 'o', 'linewidth': 1.5, 'markersize': 6},
    'TTC':    {'color': 'tab:orange','linestyle': '--','marker': 's', 'linewidth': 1.5, 'markersize': 6},
    'ACT':    {'color': 'tab:green','linestyle': '-','marker': 'd', 'linewidth': 1.5, 'markersize': 6},
    'TA':     {'color': 'tab:red',  'linestyle': ':','marker': '^', 'linewidth': 1.5, 'markersize': 6},
    '2D-TTC': {'color': 'tab:purple','linestyle': (0, (3, 1, 1, 1)),'marker': 'v', 'linewidth': 1.5, 'markersize': 6},
    'MPrISM': {'color': 'tab:cyan', 'linestyle': (0, (1,1)),'marker': 'x', 'linewidth': 1.5, 'markersize': 6}
}

all_predictions = {}
for i in range(3):
    key_i = f'collision_{i+1}'
    all_predictions[key_i] = {}
    for j in range(2):
        key_j = f'veh{j+1}'
        all_predictions[key_i][key_j] = {
            'PET': list(map(float, PET_results[i][j])) if PET_results[i][j] else [],
            'TTC': list(map(float, TTC_results[i][j])) if TTC_results[i][j] else [],
            'ACT': list(map(float, ACT_results[i][j])) if ACT_results[i][j] else [],
            'TA': list(map(float, TA_results[i][j])) if TA_results[i][j] else [],
            '2D-TTC': list(map(float, TwoD_TTC_results[i][j])) if TwoD_TTC_results[i][j] else [],
            'MPrISM': list(map(float, MPrISM_results[i][j])) if MPrISM_results[i][j] else [],
            'PMCT': list(map(float, PMCT_predicted[i][j])) if PMCT_predicted[i][j] else []
        }

json_out = os.path.join(save_dir, 'accident_predictions_all.json')
with open(json_out, 'w') as jf:
    json.dump(all_predictions, jf, indent=2)
print(f'Saved all predictions to {json_out}')

for i in range(3):
    for j in range(2):

        n = len(PET_results[i][j])
        if n == 0:
            print(f"Skip plotting collision {i+1} vehicle {j+1}: no data")
            continue

        x = np.arange(0, n) * 0.4  
        plt.figure(figsize=(12, 9))


        true_y = 5.2 - x
        plt.plot(x, true_y, linestyle='--', color='gray', label='Remaining Time to \nthe Collision Step')

        pmct_ser = PMCT_predicted[i][j] if PMCT_predicted[i][j] else []
        if pmct_ser:
            y_pmct = np.array(pmct_ser)
            s = style_map['PMCT']
            plt.plot(x, y_pmct, marker=s['marker'], color=s['color'], linestyle=s['linestyle'], linewidth=s['linewidth'], markersize=s['markersize'], label='PMCT Prediction')

        other_series = {
            'PET': PET_results[i][j],
            'TTC': TTC_results[i][j],
            'ACT': ACT_results[i][j],
            'TA': TA_results[i][j],
            '2D-TTC': TwoD_TTC_results[i][j],
            'MPrISM': MPrISM_results[i][j]
        }

        for name, ser in other_series.items():
            if not ser:
                continue
            y = np.array(ser)
            s = style_map.get(name, {'color':'gray','linestyle':'-','marker':'o','linewidth':2,'markersize':8})
            plt.plot(x, y, marker=s['marker'], color=s['color'], linestyle=s['linestyle'], linewidth=s['linewidth'], markersize=s['markersize'], label=f'{name} Prediction')


        plt.xlabel('Elapsed Time (s)')
        plt.ylabel('Predicted Collision Time (s)')

        max_x = max(x) if len(x) else 0.0
        plt.xticks(np.arange(0, max_x + 0.5, 0.5))
        plt.yticks(np.arange(0, 6.0 + 0.5, 0.5))
        plt.ylim(-0.2, 5.3)


        ax = plt.gca()
       

        out_path = os.path.join(save_dir, f'collision_{i+1}_veh{j+1}.png')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved plot: {out_path}')

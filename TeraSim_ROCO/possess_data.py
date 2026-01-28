# -*- coding: utf-8 -*-
import json
import os
import numpy as np
from datetime import datetime
import shutil
import random

def process_float_dict(input_dict, safe_flag=False, accident_type=None):
    assert accident_type in ['collision', 'junction']
    keys = sorted([float(k) for k in input_dict.keys()])
    if not keys:
        return []
    
    min_key = min(keys)
    max_key = max(keys)
    
    result_list = []
    EPSILON = 1e-9
    

    current_key1 = max_key
    current_key2 = max_key - 0.5
    
    while current_key2 >= min_key - EPSILON:
        try:
            value1 = input_dict.get(f"{current_key1:.1f}")
            value2 = input_dict.get(f"{current_key2:.1f}")
        except:
            current_key1 -= 0.1
            current_key2 -= 0.1
            continue
        
        s1_lane = value1["Ego"]["veh_lane"]
        s2_lane = value2["Ego"]["veh_lane"]
        
        if accident_type == 'collision':
            if s1_lane.startswith(':') or s2_lane.startswith(':'):
                current_key1 -= 0.1
                current_key2 -= 0.1
                continue
        elif accident_type == 'junction':
            if not s1_lane.startswith(':') or not s2_lane.startswith(':'):
                current_key1 -= 0.1
                current_key2 -= 0.1
                continue
            
        value1_np = turn_datadict_to_np(value1)
        value2_np = turn_datadict_to_np(value2)

        smaller_key = min(current_key1, current_key2)
        diff = max_key - smaller_key
        if safe_flag:
            label = 11
        else:
            label = int(diff / 0.5 + EPSILON) + 1
            if label > 10:
                label = 11
        

        result_list.append([value2_np, value1_np, label])
        current_key1 -= 0.1
        current_key2 -= 0.1
 
    if safe_flag:
        return result_list
    
    for offset in [0.4, 0.3, 0.2, 0.1, 0.0]:
        key1 = max_key - offset
        
        value1 = input_dict.get(f"{key1:.1f}")
        value1_np = turn_datadict_to_np(value1)
        value2 = None
        value2_np = turn_datadict_to_np(value2)
        
        diff = max_key - key1
        label = int(diff / 0.5 + EPSILON) + 1
        
        if label > 10:
            label = 11
        
        result_list.append([value1_np, value2_np, label])
    
    return result_list

def turn_datadict_to_np(data_dict):
    if data_dict == None:
        return [0.0]*33
    
    ego_state = data_dict["Ego"]
    ego_x = ego_state["position"][0]
    ego_y = ego_state["position"][1]
    data_np = [0]*33
    data_np[0] = ego_state["velocity"]
    data_np[1] = ego_state["lateral_speed"]
    data_np[2] = ego_state["heading"]
    name_list = ["Lead", "LeftLead", "RightLead", "Foll", "LeftFoll", "RightFoll"]
    i = 0
    for name in name_list:
        BV_state = data_dict[name]
        if BV_state is not None:
            data_np[i*5 + 3] = BV_state["position"][0] - ego_x
            data_np[i*5 + 4] = BV_state["position"][1] - ego_y
            data_np[i*5 + 5] = BV_state["velocity"]
            data_np[i*5 + 6] = BV_state["lateral_speed"]
            data_np[i*5 + 7] = BV_state["heading"]
        else:
            data_np[i*5 + 3] = 120.0
            data_np[i*5 + 4] = 120.0
            data_np[i*5 + 5] = 0.0
            data_np[i*5 + 6] = 0.0
            data_np[i*5 + 7] = 0.0
        i += 1
    return data_np
            


K = 10
delta = 0.5
T = K * delta

data_base_dir = 'data/terasim_data/temp'
train_dir = 'data/terasim_data/train_road'
val_dir = 'data/terasim_data/val_raod'
test_dir = 'data/terasim_data/test_road'
train_cross_dir = 'data/terasim_data/train_cross'
val_cross_dir = 'data/terasim_data/val_cross'
test_cross_dir = 'data/terasim_data/test_cross'

data_folder_list = os.listdir(data_base_dir)
random.shuffle(data_folder_list)
current_time = datetime.now()
name_now =  current_time.strftime("%Y%m%d_%H%M%S") + '_39'
i = 0
num = 0
for data_folder in data_folder_list:
    
    data_possess_list_total_train = []
    data_possess_list_total_val = []
    data_possess_list_total_test = []
    data_possess_list_total_cross_train = []
    data_possess_list_total_cross_val = []
    data_possess_list_total_cross_test = []
    info_data_file = os.path.join(data_base_dir, data_folder, 'collision_analysis.json')
    if not os.path.exists(info_data_file) :
        continue
    
    try:
        with open(info_data_file, 'r') as f:
            info_data = json.load(f)
    except:
        continue
    
    ran = np.random.uniform(0,1)
    
    flag = None
    for collision_info in info_data:
        collision_time = collision_info['time']
        collision_data_total = collision_info['data']
        collision_vehicle_list = collision_info['vehicles']
        collision_type = collision_info['collision_type']
        if collision_type == 'collision':
            flag = 'collision'

        elif collision_type == 'junction':
            flag = 'junction'
            
        else:
            continue
            
            
        safe_flag = False
        
        for j in range(len(collision_vehicle_list)):
            vehicle = collision_vehicle_list[j]
            if j == 2:
                safe_flag = True
            else:
                safe_flag = False
            single_vehicle_info_total = {}
            for key, value in collision_data_total.items():
                if value is None:
                    continue
                if vehicle in value.keys():
                    single_vehicle_info_total[key] = value[vehicle]
            
            data_possess_list = process_float_dict(single_vehicle_info_total, safe_flag=safe_flag, accident_type=flag)
            if flag == 'collision':
                ran = np.random.uniform(0,1)
                if ran < 0.6:
                    data_possess_list_total_train.extend(data_possess_list)
                elif ran < 0.8 and ran >= 0.6:
                    data_possess_list_total_val.extend(data_possess_list)
                else:
                    data_possess_list_total_test.extend(data_possess_list)
            elif flag == 'junction':
                ran = np.random.uniform(0,1)
                if ran < 0.6:
                    data_possess_list_total_cross_train.extend(data_possess_list)
                elif ran < 0.8 and ran >= 0.6:
                    data_possess_list_total_cross_val.extend(data_possess_list)
                else:
                    data_possess_list_total_cross_test.extend(data_possess_list)
    
    
    
    
    with open(os.path.join(train_dir, name_now + '_' + str(i) + '.json'), 'w') as f:
        json.dump(data_possess_list_total_train, f, indent=4)
        
    with open(os.path.join(val_dir, name_now + '_' + str(i) + '.json'), 'w') as f:
        json.dump(data_possess_list_total_val, f, indent=4)
        
    with open(os.path.join(test_dir, name_now + '_' + str(i) + '.json'), 'w') as f:
        json.dump(data_possess_list_total_test, f, indent=4)
        
    with open(os.path.join(train_cross_dir, name_now + '_' + str(i) + '.json'), 'w') as f:
        json.dump(data_possess_list_total_cross_train, f, indent=4)
        
    with open(os.path.join(val_cross_dir, name_now + '_' + str(i) + '.json'), 'w') as f:
        json.dump(data_possess_list_total_cross_val, f, indent=4)
        
    with open(os.path.join(test_cross_dir, name_now + '_' + str(i) + '.json'), 'w') as f:
        json.dump(data_possess_list_total_cross_test, f, indent=4)
    
    l = len(data_possess_list_total_train) + len(data_possess_list_total_val) + len(data_possess_list_total_test) + len(data_possess_list_total_cross_train) + len(data_possess_list_total_cross_val) + len(data_possess_list_total_cross_test)
    num += l
    
    shutil.rmtree(os.path.join(data_base_dir, data_folder))
    
    i += 1
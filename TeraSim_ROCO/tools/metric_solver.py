
import torch
import numpy as np
import math
from tools.MPrISM_core.MPrISM_algorithm import *
import random
from tools.PMCTNetwork_attention import PMCTNetwork_attention


def PMCT_solver(st_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_road = PMCTNetwork_attention().to(device)
    checkpoint_road = torch.load('model_params/saved_pmct_models_onlyroad/best_model_attention.pth', map_location=device)
    if 'model_state_dict' in checkpoint_road:   
        model_road.load_state_dict(checkpoint_road['model_state_dict'])
    else:
        model_road.load_state_dict(checkpoint_road)
    model_road.eval()
    
    model_cross = PMCTNetwork_attention().to(device)
    model_cross.eval()
    checkpoint_cross = torch.load('model_params/saved_pmct_models_cross/best_model_attention.pth', map_location=device)
    if 'model_state_dict' in checkpoint_cross:   
        model_cross.load_state_dict(checkpoint_cross['model_state_dict'])
    else:
        model_cross.load_state_dict(checkpoint_cross)
    
    st_list = np.array(st_list)
    if st_list.shape[1] == 40:    
        positions = [3, 9, 15, 21, 27, 33, 39]
        mask = np.ones(st_list.shape[1], dtype=bool)
        mask[positions] = False
        st_list = st_list[:, mask]
    
    st_tensor = torch.tensor(st_list)
    
    heading_ego = st_tensor[:, 2]  
    heading_nearest = st_tensor[:, 7] 
    heading_diff = torch.abs(heading_ego - heading_nearest)

    condition_mask = (heading_diff < 120) & (heading_diff > 60)
    pred = torch.full_like(st_tensor[:, 0], fill_value=0, dtype=torch.float32)  
    if condition_mask.any():  
        with torch.no_grad():
            _, msc_pred_cross = model_cross(st_tensor[condition_mask].float().to(device))
        msc_pred = msc_pred_cross.cpu()
        pred[condition_mask] = (msc_pred - 1) * 0.5  
        

    if (~condition_mask).any():  
        with torch.no_grad():
            _, msc_pred_road = model_road(st_tensor[~condition_mask].float().to(device))
        msc_pred = msc_pred_road.cpu()
        pred[~condition_mask] = (msc_pred - 1) * 0.5
    
    return pred.tolist()


def PET_solver(st_list):
    pet_list = []
    
    for state in st_list:
        
        try:
            if hasattr(state, 'tolist'):
                state = state.tolist()
                
            if len(state) == 40:
                positions = [3, 9, 15, 21, 27, 33, 39]
                state = [state[i] for i in range(len(state)) if i not in positions]
            ego_state = state[:3]  
            bg_cars = []  
            
            for i in range(6):
                start_idx = 3 + i * 5
                end_idx = start_idx + 5
                if end_idx <= len(state):
                    bg_car = state[start_idx:end_idx]
                    bg_cars.append(bg_car)
            
            ego_v_abs, ego_v_lat, ego_heading = ego_state

            ego_heading_rad = math.radians(ego_heading)
            

            car_length = 5.0 
            car_width = 2.5  

            ego_vx = ego_v_abs * math.sin(ego_heading_rad)
            ego_vy = ego_v_abs * math.cos(ego_heading_rad)
            
            ego_x, ego_y = 0.0, 0.0
            
            pet_values = []
            
            for bg_car in bg_cars:
                try:
                    rel_x, rel_y, bg_v_abs, bg_v_lat, bg_heading = bg_car
                    bg_x = rel_x
                    bg_y = rel_y
                    bg_heading_rad = math.radians(bg_heading)
                    
                    bg_vx = bg_v_abs * math.sin(bg_heading_rad)
                    bg_vy = bg_v_abs * math.cos(bg_heading_rad)
                    
                    rel_vx = bg_vx - ego_vx
                    rel_vy = bg_vy - ego_vy

                    rel_x0 = bg_x - ego_x
                    rel_y0 = bg_y - ego_y
                    
                    rel_v_mag = math.sqrt(rel_vx**2 + rel_vy**2)
                    
                    if rel_v_mag < 1e-6:  
                        pet = float('inf')
                    else:

                        rel_v_dir_x = rel_vx / rel_v_mag
                        rel_v_dir_y = rel_vy / rel_v_mag
                        
                        half_length = car_length
                        half_width = car_width                    
                        times = []

                        if abs(rel_v_dir_x) > 1e-6:
 
                            t_left = (-half_length - rel_x0) / rel_vx if rel_vx != 0 else float('inf')
                            t_right = (half_length - rel_x0) / rel_vx if rel_vx != 0 else float('inf')

                            if t_left >= 0:
                                y_at_t_left = rel_y0 + rel_vy * t_left
                                if -half_width <= y_at_t_left <= half_width:
                                    times.append(t_left)
                            if t_right >= 0:
                                y_at_t_right = rel_y0 + rel_vy * t_right
                                if -half_width <= y_at_t_right <= half_width:
                                    times.append(t_right)
                        
                        if abs(rel_v_dir_y) > 1e-6:
                            t_front = (-half_width - rel_y0) / rel_vy if rel_vy != 0 else float('inf')
                            t_back = (half_width - rel_y0) / rel_vy if rel_vy != 0 else float('inf')
                            
                            if t_front >= 0:
                                x_at_t_front = rel_x0 + rel_vx * t_front
                                if -half_length <= x_at_t_front <= half_length:
                                    times.append(t_front)
                            if t_back >= 0:
                                x_at_t_back = rel_x0 + rel_vx * t_back
                                if -half_length <= x_at_t_back <= half_length:
                                    times.append(t_back)

                        valid_times = [t for t in times if t >= 0 and not math.isinf(t)]
                        if valid_times:
                            pet = min(valid_times)
                        else:
                            pet = float('inf')

                    if pet > 5.0:
                        pet = 5.0
                    
                    pet_values.append(pet)
                    
                except Exception as e:
                    pet_values.append(5.0)
            
            if pet_values:
                min_pet = min(pet_values)
                pet_list.append(min_pet)
            else:
                pet_list.append(5.0)
                
        except Exception as e:
            pet_list.append(5.0)
    
    return pet_list
    


def mprism_single_snapshot(
    sv_state: dict, 
    bv_state: dict, 
    time_resolution: float = 0.1,
    look_ahead_steps: int = 50,
    crash_threshold: float = 5.0,
    acc_limits_sv: dict = {"x_max": 1.84, "x_min": -3.0, "y_max": 0.2, "y_min": -0.2},
    acc_limits_bv: dict = {"x_max": 1.84, "x_min": -3.0, "y_max": 0.2, "y_min": -0.2},
    safety_distance: float = 120.0
) -> float:
    """
    Arg:
        sv_state: {
            "x": float (m), "y": float (m), 
            "v": float (m/s), 
            "heading": float (rad), 
            "phi_dot": float (rad/s)
        }
        bv_state: same as sv_state, plus "id": str
        time_resolution: time step for simulation (default 0.1s)
        look_ahead_steps: number of steps to look ahead (default 50)
        crash_threshold: crash distance threshold (default 4.0m)
        acc_limits_sv/bv: acceleration limits for sv and bv
        safety_distance: safety distance to consider (default 118m)
    
    Return:
        float: Minimum MPrISM value (float), float('inf') if no collision predicted within safety distance
    """

    dx = sv_state["x"] - bv_state["x"]
    dy = sv_state["y"] - bv_state["y"]
    if np.hypot(dx, dy) > safety_distance:
        return float("inf")
    
    L_sv, b_sv = get_kamm_circle(acc_limits_sv["x_max"], acc_limits_sv["x_min"], acc_limits_sv["y_max"], acc_limits_sv["y_min"])
    L_pov, b_pov = get_kamm_circle(acc_limits_bv["x_max"], acc_limits_bv["x_min"], acc_limits_bv["y_max"], acc_limits_bv["y_min"])
    
    data_pair = (
        0.0,  # sim_time
        bv_state["id"],  # POV_id
        np.array([[0], [0], [sv_state["v"]], [sv_state["phi_dot"]]]),  # x_sv
        [sv_state["x"], sv_state["y"]],  # initial_sv_offset
        sv_state["heading"],  # initial_sv_heading
        np.array([[0], [0], [bv_state["v"]], [bv_state["phi_dot"]]]),  # x_pov
        [bv_state["x"], bv_state["y"]],  # initial_pov_offset
        bv_state["heading"],  # initial_pov_heading
        time_resolution,  # delta
        look_ahead_steps,  # steps
        crash_threshold,  # crash_threshold
        L_sv,  # L_sv
        b_sv,  # b_sv
        L_pov,  # L_pov
        b_pov,  # b_pov
        False  # plot_MPrISM_planned_traj_video_flag
    )
    _, tau, _, _, _, _ = MPrISM_algorithm_evaluate_traj(data_pair)
    return tau

def get_kamm_circle(acc_x_max, acc_x_min, acc_y_max, acc_y_min):
    """
    Calculate the Kmma circle constraints.
    """
    L_x_min, L_x_max, L_y_max = get_L_min(acc_x_min), get_L_min(acc_x_max), get_L_max(acc_y_max)

    L = np.hstack([np.vstack([L_x_min, L_x_min, -L_x_max, -L_x_max]), np.vstack([L_y_max, -L_y_max, L_y_max, -L_y_max])])
    b = np.ones((L.shape[0], 1)) * np.sin(5 / 12 * np.pi)

    return L, b

def av_list_to_dict(state_list):
    state_dict = {}
    state_dict["x"] = state_list[0]
    state_dict["y"] = state_list[1]
    state_dict["v"] = state_list[2]
    state_dict["heading"] = state_list[3]
    state_dict["phi_dot"] = state_list[4]
    return state_dict

def bv_list_to_dict(state_list_list):
    state_dict_list = []
    for i in range(len(state_list_list)):
        state_dict = av_list_to_dict(state_list_list[i])
        car_id = "car_" + str(i)
        state_dict["id"] = car_id
        state_dict_list.append(state_dict)
    return state_dict_list

def MPrISM_solver(st):

    if hasattr(st, "tolist"):
        st = st.tolist()
    st = list(st)
    if len(st) == 33:
        insert_positions = [3, 9, 15, 21, 27, 33, 39]
        st40 = []
        j = 0
        for i in range(40):
            if i in insert_positions:
                st40.append(0.0)
            else:
                st40.append(st[j])
                j += 1
        st = st40
    
    AV_state = [0, 0, st[0], st[1], st[2], st[3]]
    BV_list = []
    for i in range(6):
        BV_list.append([
            st[4 + i * 6],
            st[5 + i * 6],
            st[6 + i * 6],
            st[8 + i * 6],
            st[9 + i * 6]
        ])
    
    sv_state = av_list_to_dict([
        AV_state[0], 
        AV_state[1], 
        AV_state[2], 
        math.radians(AV_state[4]-89.9999999999), 
        math.radians(AV_state[5])
    ])
    
    bv_list = []
    for i in range(len(BV_list)):
        if BV_list[i] == None:
            continue
        bv_list.append([
            BV_list[i][0],
            BV_list[i][1],
            BV_list[i][2],
            math.radians(BV_list[i][3]-89.9999999999),
            math.radians(BV_list[i][4])
        ])
    
    bv_dict_list = bv_list_to_dict(bv_list)
    mprism_list = []
    
    for i in range(len(bv_dict_list)):
        bv_state = bv_dict_list[i]
        try:
            mprism_value = mprism_single_snapshot(sv_state, bv_state)
        except:
            mprism_value = 5.0
        mprism_list.append(max(min(mprism_value, 5.0), 0.0) if mprism_value != float("inf") else 5.0)
    
    if mprism_list != []:
        return min(mprism_list) 
    else :
        return 5.0
    
def TTC_solver(st_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st_array = np.array(st_list)
    if st_array.shape[1] == 40:
        positions = [3, 9, 15, 21, 27, 33, 39]
        mask = np.ones(st_array.shape[1], dtype=bool)
        mask[positions] = False
        st_array = st_array[:, mask]
    st_tensor = torch.tensor(st_array).to(device)
    v_ego =  st_tensor[:, 0]      
    ttc = torch.full((len(st_tensor),), 5, dtype=torch.float32).to(device)
    for i in range(6):
        x_r = st_tensor[:, 3 + 5 * i]  
        y_r = st_tensor[:, 4 + 5 * i]  
        v_b = st_tensor[:, 5 + 5 * i]  
        
        distance = torch.sqrt(torch.maximum(x_r ** 2 + y_r ** 2, 
                                            torch.tensor(0.0, device=device)))
        
        relative_velocity = torch.abs(v_ego) + torch.abs(v_b) + 1e-10
        
        ttc_new = distance / relative_velocity
        ttc = torch.minimum(ttc, ttc_new)
    return ttc.cpu().tolist()


from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from shapely.ops import nearest_points



def ACT_solver(data_list, L=5.0, W=1.8, max_act=5.0):
    final_results = []
    for row in data_list:
        if len(row) == 40:
            v_e_abs = row[0]
            h_e_rad = np.radians(row[2])
            ve_vec = np.array([v_e_abs * np.sin(h_e_rad), v_e_abs * np.cos(h_e_rad)])
            
            poly_e = translate(rotate(Polygon([(-W/2, L/2), (W/2, L/2), (W/2, -L/2), (-W/2, -L/2)]), -row[2]), 0, 0)
            
            car_acts = []
            for i in range(6):
                idx = 4 + i * 6
                rx, ry, vb_abs, hb_deg = row[idx], row[idx+1], row[idx+2], row[idx+4]
                
                if abs(rx) < 0.1 and abs(ry) < 0.1: continue

                hb_rad = np.radians(hb_deg)
                vb_vec = np.array([vb_abs * np.sin(hb_rad), vb_abs * np.cos(hb_rad)])
                
                poly_b = translate(rotate(Polygon([(-W/2, L/2), (W/2, L/2), (W/2, -L/2), (-W/2, -L/2)]), -hb_deg), rx, ry)
                
                dist_min = poly_e.distance(poly_b)
                if dist_min <= 0:
                    car_acts.append(0.0)
                    continue
                
                p1, p2 = nearest_points(poly_e, poly_b)
                
                direction_vec = np.array([p2.x - p1.x, p2.y - p1.y])
                unit_direction = direction_vec / (np.linalg.norm(direction_vec) + 1e-10)
                
                v_rel_vec = ve_vec - vb_vec
                v_approach = np.dot(v_rel_vec, unit_direction)
                
                if v_approach > 1e-3:
                    act = dist_min / v_approach
                    car_acts.append(act)
                else:
                    car_acts.append(max_act)
            
            res = min(car_acts) if car_acts else max_act
            final_results.append(min(res, max_act))
        
        elif len(row) == 33:
            v_e_abs = row[0]
            h_e_rad = np.radians(row[2])
            ve_vec = np.array([v_e_abs * np.sin(h_e_rad), v_e_abs * np.cos(h_e_rad)])
            
            poly_e = translate(rotate(Polygon([(-W/2, L/2), (W/2, L/2), (W/2, -L/2), (-W/2, -L/2)]), -row[2]), 0, 0)
            
            car_acts = []
            for i in range(6):
                idx = 3 + i * 5
                rx, ry, vb_abs, hb_deg = row[idx], row[idx+1], row[idx+2], row[idx+4]
                
                if abs(rx) < 0.1 and abs(ry) < 0.1: continue

                hb_rad = np.radians(hb_deg)
                vb_vec = np.array([vb_abs * np.sin(hb_rad), vb_abs * np.cos(hb_rad)])
                
                poly_b = translate(rotate(Polygon([(-W/2, L/2), (W/2, L/2), (W/2, -L/2), (-W/2, -L/2)]), -hb_deg), rx, ry)
                

                dist_min = poly_e.distance(poly_b)
                if dist_min <= 0:
                    car_acts.append(0.0)
                    continue
                
                p1, p2 = nearest_points(poly_e, poly_b)
                
                direction_vec = np.array([p2.x - p1.x, p2.y - p1.y])
                unit_direction = direction_vec / (np.linalg.norm(direction_vec) + 1e-10)
                
                v_rel_vec = ve_vec - vb_vec
                v_approach = np.dot(v_rel_vec, unit_direction)
                
                if v_approach > 1e-3:
                    act = dist_min / v_approach
                    car_acts.append(act)
                else:
                    car_acts.append(max_act)
            
            res = min(car_acts) if car_acts else max_act
            final_results.append(min(res, max_act))
            
    return final_results



def TwoD_TTC_solver(st_list, L=5.0, W=1.85, max_ttc=5.0):
    import math
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st_tensor = torch.tensor(np.array(st_list), dtype=torch.float32).to(device)
    
    v_ego_abs, h_ego_rad = st_tensor[:, 0], torch.deg2rad(st_tensor[:, 2])
    ve_x = v_ego_abs * torch.sin(h_ego_rad)
    ve_y = v_ego_abs * torch.cos(h_ego_rad)
    
    final_ttc = torch.full((len(st_tensor),), max_ttc, device=device)
    
    if st_tensor.shape[1] == 40:
    
        for i in range(6):
            idx = 4 + 6 * i 
            rx, ry = st_tensor[:, idx], st_tensor[:, idx+1]
            vb_abs, hb_rad = st_tensor[:, idx+2], torch.deg2rad(st_tensor[:, idx+4])
            
            vb_x = vb_abs * torch.sin(hb_rad)
            vb_y = vb_abs * torch.cos(hb_rad)
            
            rel_px, rel_py = rx, ry 
            rel_vx, rel_vy = vb_x - ve_x, vb_y - ve_y 
            
            dist_center = torch.sqrt(rel_px**2 + rel_py**2) + 1e-10
            v_rel_mag = torch.sqrt(rel_vx**2 + rel_vy**2) + 1e-10

            v_close = -(rel_px * rel_vx + rel_py * rel_vy) / dist_center
            
            cos_theta = torch.abs(rel_px * torch.sin(h_ego_rad) + rel_py * torch.cos(h_ego_rad)) / dist_center
            dynamic_l = L * cos_theta + W * (1 - cos_theta)
            real_dist = torch.clamp(dist_center - dynamic_l, min=0.0)
            
            ttc_val = torch.where(v_close > 0.01, 
                                real_dist / (v_close + 1e-10), 
                                torch.tensor(max_ttc, device=device))
            
            final_ttc = torch.minimum(final_ttc, ttc_val)
    
    elif st_tensor.shape[1] == 33:
        for i in range(6):
            idx = 3 + 5 * i 
            rx, ry = st_tensor[:, idx], st_tensor[:, idx+1]
            vb_abs, hb_rad = st_tensor[:, idx+2], torch.deg2rad(st_tensor[:, idx+4])
            
            vb_x = vb_abs * torch.sin(hb_rad)
            vb_y = vb_abs * torch.cos(hb_rad)
            

            rel_px, rel_py = rx, ry 
            rel_vx, rel_vy = vb_x - ve_x, vb_y - ve_y 
            
            dist_center = torch.sqrt(rel_px**2 + rel_py**2) + 1e-10
            v_rel_mag = torch.sqrt(rel_vx**2 + rel_vy**2) + 1e-10
            
            v_close = -(rel_px * rel_vx + rel_py * rel_vy) / dist_center
            
            cos_theta = torch.abs(rel_px * torch.sin(h_ego_rad) + rel_py * torch.cos(h_ego_rad)) / dist_center
            dynamic_l = L * cos_theta + W * (1 - cos_theta)
            real_dist = torch.clamp(dist_center - dynamic_l, min=0.0)
            

            ttc_val = torch.where(v_close > 0.01, 
                                real_dist / (v_close + 1e-10), 
                                torch.tensor(max_ttc, device=device))
            
            final_ttc = torch.minimum(final_ttc, ttc_val)    
    return final_ttc.cpu().tolist()


    
    
def TA_solver(st_list, max_ta=5.0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    st_tensor = torch.tensor(np.array(st_list), dtype=torch.float32).to(device)

    ve_abs = st_tensor[:, 0]
    he_rad = torch.deg2rad(st_tensor[:, 2])
    ve_x = ve_abs * torch.sin(he_rad)
    ve_y = ve_abs * torch.cos(he_rad)
    
    final_ta = torch.full((len(st_tensor),), max_ta, device=device)
    
    if st_tensor.shape[1] == 40:

        for i in range(6):
 
            idx = 4 + 6 * i 
            rx = st_tensor[:, idx]    
            ry = st_tensor[:, idx+1]   
            vb_abs = st_tensor[:, idx+2]
            hb_rad = torch.deg2rad(st_tensor[:, idx+4])
            
            vb_x = vb_abs * torch.sin(hb_rad)
            vb_y = vb_abs * torch.cos(hb_rad)

            det = -ve_x * vb_y + vb_x * ve_y

            safe_det_mask = torch.abs(det) > 1e-6

            t = torch.where(safe_det_mask, (rx * (-vb_y) - (-vb_x) * ry) / (det + 1e-10), torch.tensor(-1.0, device=device))
            u = torch.where(safe_det_mask, (ve_x * ry - ve_y * rx) / (det + 1e-10), torch.tensor(-1.0, device=device))
            
            ta_val = torch.where(
                (t > 0) & (u > -0.5), 
                torch.abs(t - u), 
                torch.tensor(max_ta, device=device)
            )
            
           
            final_ta = torch.minimum(final_ta, ta_val)
    elif st_tensor.shape[1] == 33:
        for i in range(6):

            idx = 3 + 5 * i 
            rx = st_tensor[:, idx]     
            ry = st_tensor[:, idx+1]  
            vb_abs = st_tensor[:, idx+2]
            hb_rad = torch.deg2rad(st_tensor[:, idx+4])
            
            vb_x = vb_abs * torch.sin(hb_rad)
            vb_y = vb_abs * torch.cos(hb_rad)

            det = -ve_x * vb_y + vb_x * ve_y

            safe_det_mask = torch.abs(det) > 1e-6
 
            t = torch.where(safe_det_mask, (rx * (-vb_y) - (-vb_x) * ry) / (det + 1e-10), torch.tensor(-1.0, device=device))
            u = torch.where(safe_det_mask, (ve_x * ry - ve_y * rx) / (det + 1e-10), torch.tensor(-1.0, device=device))

            ta_val = torch.where(
                (t > 0) & (u > -0.5), 
                torch.abs(t - u), 
                torch.tensor(max_ta, device=device)
            )

            final_ta = torch.minimum(final_ta, ta_val)

    return final_ta.cpu().tolist()
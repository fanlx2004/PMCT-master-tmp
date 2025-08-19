from safety_metric_tool.MPrISM_core.MPrISM_algorithm import *
import numpy as np
import math

def mprism_single_snapshot(
    sv_state: dict, 
    bv_state: dict, 
    time_resolution: float = 0.1,
    look_ahead_steps: int = 50,
    crash_threshold: float = 5.0,
    acc_limits_sv: dict = {"x_max": 2.0, "x_min": -4.0, "y_max": 1.0, "y_min": -1.0},
    acc_limits_bv: dict = {"x_max": 2.0, "x_min": -4.0, "y_max": 1.0, "y_min": -1.0},
    safety_distance: float = 118.0
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

def MPrISM_solver(AV_state, BV_list):
    '''
        This function is used to calculate the minimum MPrISM value for a given AV state and a list of BV states.
        Args:
            AV_state: a list of AV state, [x, y, vx, vy, heading, yaw_rate].
            BV_list: a list of BV states, [x, y, vx, heading, yaw_rate].
        Return:
            MPrISM value: a float, the minimum MPrISM value among all BV states
    '''
    
    sv_state = av_list_to_dict([
        AV_state[0], 
        AV_state[1], 
        math.sqrt(AV_state[2]**2 + AV_state[3]**2), 
        math.radians(AV_state[4]-89.9999999999), 
        AV_state[-1] / 180 * math.pi
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
            BV_list[i][-1] / 180 * math.pi
        ])
    
    bv_dict_list = bv_list_to_dict(bv_list)
    mprism_list = []
    
    for i in range(len(bv_dict_list)):
        bv_state = bv_dict_list[i]
        try:
            mprism_value = mprism_single_snapshot(sv_state, bv_state)
        except:
            print("Error calculating MPrISM")
            return None
        mprism_list.append(max(min(mprism_value, 5.0), 0.0) if mprism_value != float("inf") else 5.0)
    
    if mprism_list != []:
        return min(mprism_list) 
    else :
        return 5.0


    
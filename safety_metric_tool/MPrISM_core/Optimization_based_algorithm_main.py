# This file develops both MILP (Mixed Integer Linear Programming) and QCP (Quadratic Constrained Programming)
# formulations to solve the evasive path planning problem. QCP formulation has better performance and it is recommended to use.
# Author: Xintao Yan
# Date: 2/13/2021
# Affiliation: Michigan Traffic Lab (MTL)

import numpy as np
import os
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 30
plt.rcParams["font.family"] = "Times New Roman"
import copy
import pandas as pd
import pickle
import math
import gurobipy as gp
from gurobipy import GRB
from tqdm import tqdm

from safety_metric_tool.MPrISM_core.Visualization_functions import *

import sys
sys.path.append('G:/My Drive/Study in Michigan/2020-06-NHTSA_Project')
from safety_metric_tool.MPrISM_core.MPrISM_algorithm import get_Kamm_circle


# ====== Optimization Algorithm =========
def MILP_evasive_traj_planning(dangerous_set_matrices, veh_dynamic_matrices, look_ahead_steps, SV_state_dim, u_dim, N,
                               a_lb=-20, a_ub=20, big_M=1e4):
    # Input value
    G_all, C_all, H_all = dangerous_set_matrices["G_all"], dangerous_set_matrices["C_all"], dangerous_set_matrices[
        "H_all"]
    A, B, F, SV_initial_state, L_sv, b_sv = veh_dynamic_matrices["A"], veh_dynamic_matrices["B"], veh_dynamic_matrices[
        "F"], veh_dynamic_matrices["SV_initial_state"], veh_dynamic_matrices["L_sv"], veh_dynamic_matrices["b_sv"]

    # Create state decision variables bound. s: look_ahead_steps*SV_state_dim
    s_lb, s_ub = np.zeros((SV_state_dim, look_ahead_steps + 1)), np.zeros((SV_state_dim, look_ahead_steps + 1))
    for t in range(look_ahead_steps + 1):
        for state_dim_idx in range(SV_state_dim):
            if (state_dim_idx == 0) or (state_dim_idx == 1):  # x, y position
                s_lb[state_dim_idx, t] = -float('inf')
                s_ub[state_dim_idx, t] = float('inf')
            elif state_dim_idx == 2:  # velocity
                s_lb[state_dim_idx, t] = 0
                s_ub[state_dim_idx, t] = float('inf')
            else:  # heading
                s_lb[state_dim_idx, t] = -float('inf')
                s_ub[state_dim_idx, t] = float('inf')

    # Create action decision variables bound. u: look_ahead_steps*u_dim 
    u_lb, u_ub = np.zeros((u_dim, look_ahead_steps)), np.zeros((u_dim, look_ahead_steps))
    for t in range(look_ahead_steps):
        for state_dim_idx in range(u_dim):
            # Both ax and ay
            u_lb[state_dim_idx, t], u_ub[state_dim_idx, t] = a_lb, a_ub

    # Create a new model
    model = gp.Model("MILP_X")
    model.setParam('OutputFlag', 0)

    # Create variables
    # State
    s = model.addMVar((SV_state_dim, look_ahead_steps + 1), lb=s_lb, ub=s_ub, vtype=GRB.CONTINUOUS, name="s")

    # Action
    u = model.addMVar((u_dim, look_ahead_steps), lb=u_lb, ub=u_ub, vtype=GRB.CONTINUOUS, name="u")

    # Integer
    delta = model.addMVar((look_ahead_steps + 1, N, 4), vtype=GRB.BINARY, name="delta")
    model.update()

    # Set objective
    model.setObjective(1, GRB.MINIMIZE)
    # model.setObjective(sum(u[1,t] for t in range(look_ahead_steps)), GRB.MAXIMIZE)
    # model.setObjective(sum(s[1,t]-BV_state[1,t] for t in range(look_ahead_steps+1)), GRB.MAXIMIZE)

    # Add constraint: s[0] = SV initial state
    model.addConstr(s[:, 0] == SV_initial_state)

    # Add constraint: vehicle dynamics s[t+1] = As[t]+Bu[t]+F
    model.addConstrs(A @ s[:, t] + B @ u[:, t] + F == s[:, t + 1] for t in range(look_ahead_steps))

    # Add constraint: Not in dangerous set
    for t in range(look_ahead_steps + 1):
        G_t, c_t, h_t = G_all[t], C_all[t], H_all[t]
        for i in range(N):
            G_t_i, c_t_i, h_t_i = G_t[i], c_t[i], h_t[i]
            for j in range(G_t_i.shape[0]):
                model.addConstr(G_t_i[j, :] @ s[:, t] + c_t_i[j] >= h_t_i[j] - big_M * (1 - delta[t, i, j]))

    # Add constraint: at least one constraint is violated for each BV at each time
    for t in range(look_ahead_steps + 1):
        G_t = G_all[t]
        for i in range(N):
            G_t_i = G_t[i]
            num_j = G_t_i.shape[0]
            model.addConstr(np.ones((1, num_j)) @ delta[t, i, :] >= 1)

    # Add action admissible space
    model.addConstrs(L_sv @ u[:, t] <= b_sv for t in range(look_ahead_steps))

    # TODO: add generalized road geometry constraints
    # model.addConstrs(s[1,t] <= 9 for t in range(look_ahead_steps+1))
    model.addConstrs(s[1, t] >= -1 for t in range(look_ahead_steps + 1))

    # Optimize model
    model.optimize()

    # for v in model.getVars():
    #     print('%s %g' % (v.varName, v.x))

    model_status = model.status
    evasive_traj_exist, evasive_traj = None, None
    if model_status == 2:
        evasive_traj_exist = True
        evasive_traj = s.X
    elif model_status == 3:
        evasive_traj_exist = False
    else:
        raise ValueError("Model Status Error: the status is {0}".format(model_status))

    # with open("result/MILP/evasive_traj.npy", "wb") as f:
    #     np.save(f, evasive_traj)
    return evasive_traj_exist, evasive_traj


def QCP_evasive_traj_planning(veh_dynamic_matrices, dangerous_level_dict, BVs_all_circle_pos_array, look_ahead_steps, SV_state_dim, u_dim, N,
                              collision_threshold=4., a_lb=-20., a_ub=20.):
    """
    This function do QCP optimization to determine whether the current snapshot is safe or not.

    :param veh_dynamic_matrices: Include the SV dynamics transition matrices, and the linear transition matrices of
      the three circles that approximates the SV.
    :param dangerous_level_dict: the dictionary of the dangerous level, key: dangerous level, value: action percentage.
    :param BVs_all_circle_pos_array: The three circles information of all BVs.
    :param look_ahead_steps: number of look ahead steps starting from this snapshot.
    :param SV_state_dim: the dimension of the SV state
    :param u_dim: the dimension of the action
    :param N: number of BVs considered
    :param collision_threshold: collision threshold (m)
    :param a_lb: action lower bound
    :param a_ub: action upper bound
    :return: evasive_traj_exist: the flag of whether evasive trajectory exists, evasive_traj: the planned evasive
      traj if it exists, evasive_traj_three_circles: the evasive traj of all three circles if it exists.
    """
    assert (BVs_all_circle_pos_array.shape == (N, 3, 4, look_ahead_steps + 1))

    # Input value
    A, B, F, SV_initial_state, L_sv_dict, b_sv_dict, M_front_circle, M_rear_circle, N_front_circle, N_rear_circle = \
        veh_dynamic_matrices["A"], veh_dynamic_matrices["B"], veh_dynamic_matrices["F"], veh_dynamic_matrices[
            "SV_initial_state"], veh_dynamic_matrices["L_sv_dict"], veh_dynamic_matrices["b_sv_dict"], veh_dynamic_matrices[
            "M_front"], veh_dynamic_matrices["M_rear"], veh_dynamic_matrices["N_front"], veh_dynamic_matrices["N_rear"]

    # Create state decision variables bound. s: look_ahead_steps*SV_state_dim
    s_lb, s_ub = np.zeros((SV_state_dim, look_ahead_steps + 1)), np.zeros((SV_state_dim, look_ahead_steps + 1))
    for t in range(look_ahead_steps + 1):
        for state_dim_idx in range(SV_state_dim):
            if (state_dim_idx == 0) or (state_dim_idx == 1):  # x, y position
                s_lb[state_dim_idx, t] = -float('inf')
                s_ub[state_dim_idx, t] = float('inf')
            elif state_dim_idx == 2:  # velocity
                s_lb[state_dim_idx, t] = 0
                s_ub[state_dim_idx, t] = float('inf')
            else:  # heading
                s_lb[state_dim_idx, t] = -float('inf')
                s_ub[state_dim_idx, t] = float('inf')

    # Create action decision variables bound. u: look_ahead_steps*u_dim 
    u_lb, u_ub = np.zeros((u_dim, look_ahead_steps)), np.zeros((u_dim, look_ahead_steps))
    for t in range(look_ahead_steps):
        for state_dim_idx in range(u_dim):
            # Both ax and ay
            u_lb[state_dim_idx, t], u_ub[state_dim_idx, t] = a_lb, a_ub

    # Create a new model
    model = gp.Model("QCP_X")
    model.setParam('OutputFlag', 0)
    model.params.NonConvex = 2

    # Create variables
    # the center circle State
    s_c = model.addMVar((SV_state_dim, look_ahead_steps + 1), lb=s_lb, ub=s_ub, vtype=GRB.CONTINUOUS, name="s_c")
    s_f = model.addMVar((SV_state_dim, look_ahead_steps + 1), lb=s_lb, ub=s_ub, vtype=GRB.CONTINUOUS, name="s_f")
    s_r = model.addMVar((SV_state_dim, look_ahead_steps + 1), lb=s_lb, ub=s_ub, vtype=GRB.CONTINUOUS, name="s_r")

    # Action
    u = model.addMVar((u_dim, look_ahead_steps), lb=u_lb, ub=u_ub, vtype=GRB.CONTINUOUS, name="u")
    model.update()

    # Set objective
    model.setObjective(1, GRB.MINIMIZE)
    # model.setObjective(sum(u[1,t] for t in range(look_ahead_steps)), GRB.MAXIMIZE)
    # model.setObjective(sum(s[1,t]-BV_state[1,t] for t in range(look_ahead_steps+1)), GRB.MAXIMIZE)

    # Add constraint: s_c[0] = SV initial state
    model.addConstr(s_c[:, 0] == SV_initial_state)

    # Add constraint: the center circle follows the vehicle dynamics s[t+1] = As[t]+Bu[t]+F
    model.addConstrs(A @ s_c[:, t] + B @ u[:, t] + F == s_c[:, t + 1] for t in range(look_ahead_steps))

    # Add constraints: the front the rear circle state
    model.addConstrs(M_front_circle @ s_c[:, t] + N_front_circle == s_f[:, t] for t in range(look_ahead_steps + 1))
    model.addConstrs(M_rear_circle @ s_c[:, t] + N_rear_circle == s_r[:, t] for t in range(look_ahead_steps + 1))

    # Add constraint: Not conflict with any BV's circle at any time
    for i in range(N):
        one_BV_all_circles_pos = BVs_all_circle_pos_array[i, :, :, :]
        for j in range(3):  # Number of circles for BV TODO: hard code 3 circles here
            one_BV_certain_circle = one_BV_all_circles_pos[j, :, :]
            for t in range(look_ahead_steps + 1):
                tmp = one_BV_certain_circle[0:2, t] @ one_BV_certain_circle[0:2, t]
                model.addMQConstr(Q=np.eye(2), c=-2 * one_BV_certain_circle[0:2, t], sense=">",
                                  rhs=collision_threshold ** 2 - tmp, xQ_L=s_r[0:2, t], xQ_R=s_r[0:2, t],
                                  xc=s_r[0:2, t])
                model.addMQConstr(Q=np.eye(2), c=-2 * one_BV_certain_circle[0:2, t], sense=">",
                                  rhs=collision_threshold ** 2 - tmp, xQ_L=s_c[0:2, t], xQ_R=s_c[0:2, t],
                                  xc=s_c[0:2, t])
                model.addMQConstr(Q=np.eye(2), c=-2 * one_BV_certain_circle[0:2, t], sense=">",
                                  rhs=collision_threshold ** 2 - tmp, xQ_L=s_f[0:2, t], xQ_R=s_f[0:2, t],
                                  xc=s_f[0:2, t])

    # TODO: add generalized road geometry constraints
    # model.addConstrs(s_c[1,t] <= 9 for t in range(look_ahead_steps+1))
    # model.addConstrs(s_c[1,t] >= -1 for t in range(look_ahead_steps+1))

    # Solve the problem multiple times with different action admissible space constraints
    dangerous_level, evasive_traj_exist, evasive_traj, evasive_traj_three_circles = None, None, None, None
    solve_first_time_flag = True
    for check_dangerous_level in dangerous_level_dict.keys():
        assert (check_dangerous_level >= 1)
        L_sv, b_sv = L_sv_dict[check_dangerous_level], b_sv_dict[check_dangerous_level]
        if solve_first_time_flag:  # If it is the first time, then just need to add it
            # model.addConstrs(L_sv @ u[:, t] <= b_sv for t in range(look_ahead_steps))
            model.addConstrs((L_sv @ u[:, t] <= b_sv for t in range(look_ahead_steps)), name="admissible_space_constrs")
            solve_first_time_flag = False
        else:
            # Remove previous admissible space constrs first then add the updated one
            admissible_constrs = model.getConstrs()[-12*(look_ahead_steps):]  # TODO: hardcode here to retrieve the last 12*look_ahead_steps constrs for the admissible space
            model.remove(admissible_constrs)
            # Update the constrs
            model.addConstrs((L_sv @ u[:, t] <= b_sv for t in range(look_ahead_steps)), name="admissible_space_constrs")
            model.update()

        # Optimize model
        model.optimize()

        model_status = model.status
        if model_status == 2:
            dangerous_level = check_dangerous_level - 1
            evasive_traj_exist = True
            evasive_traj = s_c.X
            evasive_traj_three_circles = [s_r.X, s_c.X, s_f.X]
            break
        elif model_status == 3:
            dangerous_level = check_dangerous_level
            evasive_traj_exist = False
        else:
            print("Model Status Error: the status is {0}".format(model_status))
            # exit()
            # raise ValueError("Model Status Error: the status is {0}".format(model_status))
    assert(dangerous_level == list(dangerous_level_dict.keys())[-1] if evasive_traj_exist is False else dangerous_level < list(dangerous_level_dict.keys())[-1])
    return dangerous_level, evasive_traj_exist, evasive_traj, evasive_traj_three_circles


def first_step_safety_check(truncate_traj, ax_threshold=None, ay_threshold=None, steering_degree_threshold=None):
    """
    This function checks that whether actual actions of the SV within the look-ahead steps exceeds the ax or ay (or
    steering) threshold or collision happens during the look-ahead steps. If not, then this moment is directly
    identified as safe and no optimization-based check is needed. Otherwise, optimization-based method will be
    performed.

    :param truncate_traj: the truncated trajectory starting from the current snapshot until the look ahead steps.
    :param ax_threshold: the longitudinal acceleration threshold. Negative value such as -3 m/s^2 since it is for
      braking
    :param ay_threshold: the lateral acceleration threshold. Positive value since for both sides are evasive.
    :param steering_degree_threshold: the steering angle threshold. Positive value since for both sides are evasive. In
      degrees, e.g., 5 degrees.
    :return: evasive_traj_plan_needed_flag, if it is False then the snapshot is safe, otherwise, we will perform the
      optimization-based method.
    """
    SV_truncate_traj = truncate_traj[truncate_traj["veh_id"] == "CAV"]
    assert ('acc' in truncate_traj.columns and ('acc_y' in truncate_traj.columns or 'steering' in truncate_traj.columns))
    assert (ax_threshold < 0)
    assert (ay_threshold > 0 if ay_threshold is not None else ay_threshold is None)
    assert (steering_degree_threshold > 0 if steering_degree_threshold is not None else steering_degree_threshold is None)

    crash_flag, ax_violate_flag, ay_violate_flag, steering_violate_flag = False, False, False, False

    # Whether crash happens
    if "Crash" in SV_truncate_traj["mode"].values:
        crash_flag = True

    # Whether action exceeds the threshold during the look-ahead steps
    ax_violate_flag = (SV_truncate_traj.acc < ax_threshold).any()
    if 'acc_y' in SV_truncate_traj.columns:
        ay_violate_flag = (np.abs(SV_truncate_traj.acc_y) > ay_threshold).any()
    if 'steering' in SV_truncate_traj.columns:
        steering_violate_flag = (np.abs(SV_truncate_traj.steering) > math.radians(steering_degree_threshold)).any()
    evasive_traj_plan_needed_flag = crash_flag or ax_violate_flag or ay_violate_flag or steering_violate_flag
    # print(SV_truncate_traj.iloc[0]["time"].item(), crash_flag, SV_truncate_traj.acc.min(), np.abs(SV_truncate_traj.steering).max())
    return evasive_traj_plan_needed_flag


# ====== Load traj and extend traj =========
def load_traj():
    traj_address = "G:/My Drive/Study in Michigan/2020-06-NHTSA_Project"
    traj = pd.read_csv(os.path.join(traj_address, str(case_id) + ".csv"))
    traj["y"] = -traj["y"] + 8
    traj["heading"] = -traj["heading"]

    # traj.loc[(traj["veh_id"]=="2e0844b0-ac7f-466a-be1b-80bfb0e3de84"),"x"] += 5
    # traj.loc[(traj["veh_id"]=="2e0844b0-ac7f-466a-be1b-80bfb0e3de84"),"y"] += 4

    return traj


def extend_traj(traj, extend_steps, extend_delta):
    """This function extend the BVs trajectories assuming they will maintain the current velocity and heading

    :param traj: the dataframe of the original
    :param extend_steps: number of steps to extend. e.g. 15
    :param extend_delta: time resolution. e.g. 1/15 second 
    
    :return: a new dataframe that includes the extended BVs trajectories.
    """
    t_last = traj.time.iloc[-1]
    last_moment = traj.groupby('time').get_group(t_last).reset_index(drop=True)
    last_moment_except_CAV = last_moment[last_moment["veh_id"] != "CAV"].reset_index(drop=True)

    extend_traj_dict = {}  # key: veh_id, value: extended traj np.array
    for veh_id, x, y, v, heading in zip(list(last_moment_except_CAV.veh_id), list(last_moment_except_CAV.x),
                                        list(last_moment_except_CAV.y), list(last_moment_except_CAV.v),
                                        list(last_moment_except_CAV.heading)):
        initial_state = [x, y, v, heading]
        predicted_traj = _traj_predict(initial_state, extend_steps, extend_delta)
        # extend_traj_dict[veh_id] = predicted_traj  # If AA rdbt data is used, then no need to bound the traj, then use this line and comment the next two lines.
        bound_pred_traj = bound_predict_traj(predicted_traj, y_min=0, y_max=8, y_epsilon=0.2)
        extend_traj_dict[veh_id] = bound_pred_traj

    extend_traj_list = []
    for t in range(extend_steps):
        t_idx = t + t_last + 1
        for veh_id in extend_traj_dict.keys():
            x, y, v, heading = extend_traj_dict[veh_id][:, t].tolist()
            tmp = [t_idx, veh_id, x, y, v, heading, True]
            extend_traj_list.append(tmp)
    extend_df = pd.DataFrame(extend_traj_list, columns=["time", 'veh_id', 'x', 'y', 'v', 'heading', 'is_extend_traj'])

    extend_traj = traj.append(extend_df, ignore_index=True)

    return extend_traj


def _traj_predict(state, steps, delta):
    """
    This function is to predict the vehicle trajectory using the 
    initial state (x,y,v,heading) and assuming the the static velocity
    and heading. SV dynamics in global coordinate system: s(t+1) = As(t) + Bu(t) + F.
    
    :param state: list of the initial state [x,y,v,heading]
    :param steps: number of steps to extend. e.g. 15
    :param delta: time resolution. e.g. 1/15 second
            
    :return: vehicle states in the following steps. Matrix size = 4 X steps
    """
    ini_x, ini_y, ini_v, ini_heading = state
    initial_coord = [ini_x, ini_y]
    local_state = np.array(state).reshape(-1, 1)
    # Get vehicle dynamic matrices
    A, B, F = get_SV_dynamics_transition_matrices(delta, ini_v, ini_heading, initial_coord, state_space="global")

    res = np.zeros((4, steps + 1))
    res[:, 0] = local_state[:, 0]
    for T in range(0, steps, 1):
        # Vehicle dynamics transition matrix As(t) + F, since assume static so u=0
        s_prev = res[:, T].reshape(-1, 1)
        res[:, T + 1] = (A @ s_prev + F)[:, 0]
    res = res[:, 1:]
    return res


def bound_predict_traj(traj, y_min, y_max, y_epsilon=0.2):
    """
    The predict traj could exceed the road boundary, especially in y axis.
    Therefore, we will bound the predict traj to the boundary
    Input:
        1. predicted trajectory (2*steps) at each timestep
        2. ymin coordinate, ymax coordinate given by geometry
        3. y_epsilon is the buffer to position error. e.g. the max y is 8
            but the data could be 8.01, etc.
    Output:
        1. modified predicted trajectory satisfying the road boundary.
    """
    y_min_hat, y_max_hat = y_min - y_epsilon, y_max + y_epsilon  # the modified bound considering epsilon

    bound_pred_traj = copy.deepcopy(traj)
    bound_pred_traj[1, bound_pred_traj[1, :] > y_max_hat] = y_max_hat
    bound_pred_traj[1, bound_pred_traj[1, :] < y_min_hat] = y_min_hat
    assert ((y_min_hat <= bound_pred_traj[1, :]).all() and (bound_pred_traj[1, :] <= y_max_hat).all())
    return bound_pred_traj


# ====== Get SV dynamics transition matrices =========
def get_SV_dynamics_transition_matrices(delta, SV_initial_v, SV_initial_heading, SV_initial_coord,
                                        state_space="global"):
    A, B = get_A(delta, SV_initial_v, initial_heading=SV_initial_heading, state_space=state_space), get_B(delta,
                                                                                                          SV_initial_v,
                                                                                                          initial_heading=SV_initial_heading,
                                                                                                          state_space=state_space)
    F = get_F(delta, SV_initial_v, initial_heading=SV_initial_heading, initial_coord=SV_initial_coord,
              state_space=state_space)

    return A, B, F


def get_A(delta, v_til, initial_heading=None, state_space="global"):
    """
    SV dynamics: s(t+1) = As(t) + Bu(t) + F
    If the state_space == "global" the transition is in the global coordinate system.
    """
    A_sv = np.array([[1, 0, delta, 0], [0, 1, 0, v_til * delta], [0, 0, 1, 0], [0, 0, 0, 1]])
    if state_space == "global":
        R, R_inv = get_rotation_matrix(initial_heading, direction="local2global"), get_rotation_matrix(initial_heading,
                                                                                                       direction="global2local")
        R, R_inv = block_diag(R, np.eye(2)), block_diag(R_inv, np.eye(2))
        A_sv = R @ A_sv @ R_inv
    return A_sv


def get_B(delta, v_til, initial_heading=None, state_space="global"):
    """
    SV dynamics: s(t+1) = As(t) + Bu(t) + F
    """
    B_sv = np.array([[0.5 * delta ** 2, 0], [0, 0.5 * delta ** 2], [delta, 0], [0, delta / v_til]])
    if state_space == "global":
        R = get_rotation_matrix(initial_heading, direction="local2global")
        R = block_diag(R, np.eye(2))
        B_sv = R @ B_sv
    return B_sv


def get_F(delta, v_til, initial_heading=None, initial_coord=None, state_space="global"):
    """
    SV dynamics: s(t+1) = As(t) + Bu(t) + F
    """
    if state_space == "local":
        F = np.zeros((4, 1))
    else:
        assert (state_space == "global")
        A_sv = np.array([[1, 0, delta, 0], [0, 1, 0, v_til * delta], [0, 0, 1, 0], [0, 0, 0, 1]])
        R, R_inv = get_rotation_matrix(initial_heading, direction="local2global"), get_rotation_matrix(initial_heading,
                                                                                                       direction="global2local")
        R, R_inv = block_diag(R, np.eye(2)), block_diag(R_inv, np.eye(2))
        O = np.array([[initial_coord[0]], [initial_coord[1]], [0], [0]])
        H = np.array([[0], [0], [0], [initial_heading]])
        F = (np.eye(4) - R @ A_sv @ R_inv) @ (O + H)
    return F


def get_rotation_matrix(theta, direction="local2global"):
    if direction == "local2global":
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    elif direction == "global2local":
        R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    return R


# ======================================================

# ====== Get three circle position transition matrices =========
def get_circle_transition_matrices(initial_heading, center_point_distance, initial_coord, circle_pos=None):
    """This function gets the linear transition matrices for the front and rear circle of the vehicle using the
    center circle.
    The vehicle is approximated using three circles, and the center one follows the linear dynamics transition. The
    cicle in position p (could be f(front) or r(rear)) follows s^p(t) = A@s^c(t) + F

    :param initial_heading: the initial heading at the starting moment.
    :param center_point_distance: the distances between the front and rear circles.
    :param initial_coord: [x,y] initial global coordinates of the center circle.
    :param circle_pos: could be "rear" or "front".
    :return: the A matrix and the F matrix.
    """
    assert (circle_pos == "rear" or circle_pos == "front")
    if circle_pos == "front":
        A = np.array([[1, 0, 0, 0], [0, 1, 0, center_point_distance / 2], [0, 0, 1, 0], [0, 0, 0, 1]])
        F = np.array([[center_point_distance / 2], [0], [0], [0]])
    else:
        A = np.array([[1, 0, 0, 0], [0, 1, 0, -center_point_distance / 2], [0, 0, 1, 0], [0, 0, 0, 1]])
        F = np.array([[-center_point_distance / 2], [0], [0], [0]])
    R, R_inv = get_rotation_matrix(initial_heading, direction="local2global"), get_rotation_matrix(initial_heading,
                                                                                                   direction="global2local")
    R, R_inv = block_diag(R, np.eye(2)), block_diag(R_inv, np.eye(2))
    A_final = R @ A @ R_inv

    O = np.array([[initial_coord[0]], [initial_coord[1]], [0], [0]])
    H = np.array([[0], [0], [0], [initial_heading]])
    F_final = (np.eye(4) - R @ A @ R_inv) @ (O + H) + R @ F

    return A_final, F_final


# ======================================================

# ====== Get SV admissible space constrs matrices =========
def get_SV_admissible_space_constrs_matrices(acc_x_max_sv=2., acc_x_min_sv=-4., acc_y_max_sv=2., acc_y_min_sv=-2.,
                                             action_constraint_percent=1., admissible_space="Kamm_circle"):
    # Action constraints
    if admissible_space == "Kamm_circle":
        L_sv, b_sv = get_Kamm_circle(action_constraint_percent * acc_x_max_sv, action_constraint_percent * acc_x_min_sv,
                                     action_constraint_percent * acc_y_max_sv, action_constraint_percent * acc_y_min_sv)
    if admissible_space == "Box":
        L_sv, b_sv = get_Box_admissible_space(action_constraint_percent * acc_x_max_sv,
                                              action_constraint_percent * acc_x_min_sv,
                                              action_constraint_percent * acc_y_max_sv,
                                              action_constraint_percent * acc_y_min_sv)
    b_sv = b_sv.reshape(b_sv.shape[0], )
    return L_sv, b_sv


def get_Box_admissible_space(acc_x_max_sv, acc_x_min_sv, acc_y_max_sv, acc_y_min_sv):
    L_sv = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    b_sv = np.array([[acc_x_max_sv], [-acc_x_min_sv], [acc_y_max_sv], [-acc_y_min_sv]])
    return L_sv, b_sv


# ======================================================

# ====== Get dangerous set constrs matrices =========
# MILP dangerous set constrs
def get_X0_constrs_matrices(x_collision_threshold, y_collision_threshold, look_ahead_steps, t_interval, N, BVs_id_list,
                            one_episode):
    G_0 = np.array([[1, 0, 0, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, -1, 0, 0]])
    h_0 = np.array([[x_collision_threshold], [x_collision_threshold], [y_collision_threshold], [y_collision_threshold]])

    G_all, C_all, H_all = {}, {}, {}
    for t_idx in range(look_ahead_steps + 1):
        time = t_interval[t_idx]
        G_t_list, c_t_list, h_t_list = [], [], []
        for i in range(N):
            G_i, h_i = copy.deepcopy(G_0), copy.deepcopy(h_0)
            G_t_list.append(G_i), h_t_list.append(h_i)

            BV_id = BVs_id_list[i]
            veh_data_specific_moment = one_episode[(one_episode["veh_id"] == BV_id) & (one_episode["time"] == time)]
            BV_x, BV_y = veh_data_specific_moment["x"].item(), veh_data_specific_moment["y"].item()
            c_i = np.array([[-BV_x], [BV_x], [-BV_y], [BV_y]])
            c_t_list.append(c_i)
        G_all[t_idx], C_all[t_idx], H_all[t_idx] = G_t_list, c_t_list, h_t_list
    X0_dict = {"G_all": G_all, "C_all": C_all, "H_all": H_all}
    return X0_dict


# Get all BVs all circles state, used for collision constrs in QCP formulation
def get_BVs_circles_position(t_interval, N, BVs_id_list, one_episode):
    BVs_circle_position_list = []  # []
    start_time, end_time = t_interval[0], t_interval[-1]
    for i in range(N):
        try: # Situation that there is no enough vehicle in the SV observation range
            BV_id = BVs_id_list[i]
            specific_BV = one_episode[(one_episode["veh_id"] == BV_id) & (one_episode["time"] >= start_time) & (one_episode["time"] <= end_time)]
            one_BV_rear_state_array = specific_BV.loc[:, ["rear_circle_x", "rear_circle_y", "v", "heading"]].to_numpy().T
            one_BV_center_state_array = specific_BV.loc[:, ["x", "y", "v", "heading"]].to_numpy().T
            one_BV_front_state_array = specific_BV.loc[:, ["front_circle_x", "front_circle_y", "v", "heading"]].to_numpy().T
            try:  # Situation that the vehicle is on the boundaries of SV observation range and not observed along the whole look-ahead steps.
                assert(one_BV_rear_state_array.shape == (4, len(t_interval)))
                assert(one_BV_center_state_array.shape == (4, len(t_interval)))
                assert(one_BV_front_state_array.shape == (4, len(t_interval)))
            except:  # Generate a dummy vehicle that will not influence with the SV
                SV_x = one_episode.loc[(one_episode["veh_id"]=="CAV") & (one_episode["time"]==t_interval[0]), "x"].item()
                BV_x = one_BV_center_state_array[0,0]
                dummy_state_array =  _generate_dummy_BV_state_array(SV_x, BV_x, len(t_interval))
                one_BV_rear_state_array = one_BV_center_state_array = one_BV_front_state_array = dummy_state_array
        except:
            SV_x = one_episode.loc[(one_episode["veh_id"] == "CAV") & (one_episode["time"] == t_interval[0]), "x"].item()
            dummy_state_array = _generate_dummy_BV_state_array(SV_x, BV_x=-9999999999, t_interval_length=len(t_interval))
            one_BV_rear_state_array = one_BV_center_state_array = one_BV_front_state_array = dummy_state_array
        
        BVs_circle_position_list.append(np.array([one_BV_rear_state_array, one_BV_center_state_array, one_BV_front_state_array]))            
    # Should be a N*3*4*t array
    BVs_all_circle_pos_array = np.array(BVs_circle_position_list)
    return BVs_all_circle_pos_array


def _generate_dummy_BV_state_array(SV_x, BV_x, t_interval_length):
    """
    This function is to deal with corner case situation that (1) there is not enough BVs within the SV observation range at all, (2) the BV does not occur entirely within the look-ahead steps.

    :param SV_x: the initial position of the SV.
    :param BV_x: the initial position of the BV.
    :return: dummy BV state array.
    """
    assert (np.abs(SV_x - BV_x) > 50)
    dummy_state = [SV_x - 300, 0, 0, 0]
    dummy_state_array = np.array([dummy_state for i in range(t_interval_length)]).T
    return dummy_state_array


def util_cal_circle_position(x, y, heading, center_point_distance, circle_pos=None):
    """Util tools to calculate front or rear circle position using global center position and heading.
    """
    assert (circle_pos == "front" or circle_pos == "rear")
    if circle_pos == "front":
        x_cir, y_cir = x + (center_point_distance / 2) * np.cos(heading), y + (center_point_distance / 2) * np.sin(
            heading)
    else:
        x_cir, y_cir = x - (center_point_distance / 2) * np.cos(heading), y - (center_point_distance / 2) * np.sin(
            heading)
    return x_cir, y_cir


# ======================================================


case_id = 19337  # 1061 3767 19337 151316
# Save result parameters
plot_static_evasive_fig_flag, plot_evasive_video_flag, plot_whole_video_flag = True, True, True
save_evaluation_df_flag = True
main_folder_address = "E:/2021-01-Safety-Metric-Evaluation/result/QCP"
# Static evasive trajectory figure
plot_box_freq = 5
fig_size_factor = 5
SV_static_evasive_fig_fill_flag = True
crash_BV_color = "k"
BV_fill_flag, alpha = True, 0.1
SV_center_point_plot_flag = True
plot_evasive_traj_flag, plot_evasive_box, plot_evasive_traj_line_flag = True, True, True
plot_three_circles_flag, plot_three_circles_original_CAV_flag = True, False
plot_evasive_traj_three_circles_flag = True
evasive_traj_color, evasive_fill_flag = "blue", True
extend_traj_color = None  # If None then the extend traj color will be the same as its original color
# Evasive traj video
slow_version_evasive_video = True
# Whole traj video
slow_version_whole_traj_video = False


def MILP_main():
    # Settings
    num_steps_tracing_back_from_crash = 30
    desired_look_ahead_steps, SV_state_dim = 45, 4
    extend_traj_flag, extend_steps, extend_delta = True, 30, 1 / 15
    u_dim = 2
    a_lb, a_ub = -20, 20
    N = 4  # number of surrouding BVs considered
    delta = 1 / 15
    state_space = "global"
    admissible_space = "Kamm_circle"  # "Kamm_circle"/ "Box"
    x_collision_threshold, y_collision_threshold = 5.1, 2.2  # collision threshold
    big_M = 1e4

    # Vehicle parameters
    acc_x_max_sv, acc_x_min_sv, acc_y_max_sv, acc_y_min_sv = 2., -4., 2., -2.
    assert (np.abs(acc_y_min_sv) == acc_y_max_sv)
    action_constraint_percent = 0.5

    # Load the trajectory for evaluation
    traj = load_traj()

    # Get desired time interval list for each moment
    avail_time = list(traj.time.unique())
    assert (num_steps_tracing_back_from_crash >= 1)
    interval_waiting_to_be_evaluated = []
    for t_idx in range(num_steps_tracing_back_from_crash, 0, -1):
        reference_idx = t_idx + 1
        start_time = avail_time[-reference_idx]
        desired_end_time = start_time + desired_look_ahead_steps
        t_interval = range(start_time, desired_end_time + 1, 1)
        interval_waiting_to_be_evaluated.append(t_interval)

    # Extend the trajectory if needed
    # TODO: how to extend the trajectory when the BV has heading angle?
    if extend_traj_flag:
        extended_traj = extend_traj(traj=traj, extend_steps=extend_steps, extend_delta=extend_delta)
        traj = extended_traj

    # main loop, evaluate each moment
    one_episode = copy.deepcopy(traj)
    avail_time = list(one_episode.time.unique())
    for t_interval in interval_waiting_to_be_evaluated:
        t_interval = sorted(list(set(t_interval) & set(avail_time)))
        current_time = t_interval[0]
        look_ahead_steps = len(t_interval) - 1
        # print(t_interval, look_ahead_steps)
        # continue

        # Get surrounding BVs id list
        BVs_traj_tmp = one_episode.groupby('time').get_group(t_interval[0]).reset_index(drop=True)
        BVs_traj_tmp = BVs_traj_tmp[BVs_traj_tmp["veh_id"] != "CAV"].reset_index(drop=True)
        BVs_id_list = BVs_traj_tmp.iloc[:N]["veh_id"].tolist()
        assert (N == len(BVs_id_list))

        # SV dynamics matrices
        SV_initial_moment = one_episode[(one_episode["veh_id"] == "CAV") & (one_episode["time"] == t_interval[0])]
        SV_initial_v, SV_initial_heading, SV_initial_x, SV_initial_y = SV_initial_moment["v"].item(), SV_initial_moment[
            "heading"].item(), SV_initial_moment["x"].item(), SV_initial_moment["y"].item()
        SV_initial_coord = [SV_initial_x, SV_initial_y]
        A, B, F = get_SV_dynamics_transition_matrices(delta, SV_initial_v, SV_initial_heading, SV_initial_coord,
                                                      state_space=state_space)
        F = F.reshape(4, )  # For gurobi settings.

        # SV action admissible space
        L_sv, b_sv = get_SV_admissible_space_constrs_matrices(acc_x_max_sv=acc_x_max_sv, acc_x_min_sv=acc_x_min_sv,
                                                              acc_y_max_sv=acc_y_max_sv, acc_y_min_sv=acc_y_min_sv,
                                                              action_constraint_percent=action_constraint_percent,
                                                              admissible_space=admissible_space)

        SV_initial_state = np.array([SV_initial_x, SV_initial_y, SV_initial_v, SV_initial_heading])
        veh_dynamic_matrices_dict = {"A": A, "B": B, "F": F, "SV_initial_state": SV_initial_state, "L_sv": L_sv,
                                     "b_sv": b_sv}

        # Get X0 dangerous set
        X0_dict = get_X0_constrs_matrices(x_collision_threshold, y_collision_threshold, look_ahead_steps, t_interval, N,
                                          BVs_id_list, one_episode)

        # MILP algorithm
        evasive_traj_exist, evasive_planned_traj = MILP_evasive_traj_planning(dangerous_set_matrices=X0_dict,
                                                                              veh_dynamic_matrices=veh_dynamic_matrices_dict,
                                                                              look_ahead_steps=look_ahead_steps,
                                                                              SV_state_dim=SV_state_dim, u_dim=u_dim,
                                                                              N=N, a_lb=a_lb, a_ub=a_ub, big_M=big_M)
        print("Time idx: {0}, evasive_traj_exist: {1}".format(str(current_time), str(evasive_traj_exist)))



def QCP_main():
    # Vehicle dynamics assumed parameters
    acc_x_max_sv, acc_x_min_sv, acc_y_max_sv, acc_y_min_sv = 6., -8., 6., -6.
    assert (np.abs(acc_y_min_sv) == acc_y_max_sv)
    safe_dangerous_level, unavoidable_dangerous_level = 0, 4
    dangerous_level_dict = {1: 0.4, 2: 0.6, 3: 0.8, 4: 1.0}  # Key: dangerous level, Value: need action constraint percentage
    first_dangerous_level_percentage = dangerous_level_dict[1]
    initial_check_ax_threshold, initial_check_ay_threshold, initial_check_steering_degree_threshold = acc_x_min_sv * first_dangerous_level_percentage, acc_y_max_sv * first_dangerous_level_percentage, 5  # m/s^2, m/s^2, degrees.
    color_dict = generate_discrete_color_dict(safe_state_idx=safe_dangerous_level, unavoidable_state_idx=unavoidable_dangerous_level)

    # Settings
    num_steps_tracing_back_from_crash = 120
    desired_look_ahead_steps, SV_state_dim = 30, 4
    extend_traj_flag, extend_steps, extend_delta = True, 30, 1 / 15
    radius, center_point_distance = 1.3, 3.5
    collision_threshold = 2 * radius
    u_dim = 2
    a_lb, a_ub = -20, 20
    N = 4  # number of surrouding BVs considered
    delta = 1 / 15
    state_space = "global"
    admissible_space = "Kamm_circle"  # "Kamm_circle"/ "Box"

    folder_name = "{0}_look_ahead_steps_{1}_extend_{2}_percent_start_{3}_end_{4}_nlevel_{5}".format(str(case_id), str(desired_look_ahead_steps), str(extend_steps),
                                                                                                    str(int(100 * dangerous_level_dict[1])),
                                                                                                    str(int(100 * dangerous_level_dict[unavoidable_dangerous_level])),
                                                                                           str(unavoidable_dangerous_level - 1))
    folder_address = os.path.join(main_folder_address, folder_name)
    os.makedirs(folder_address, exist_ok=True)

    # Load the trajectory for evaluation
    traj = load_traj()
    traj["dangerous_level"] = None

    # Get desired time interval list for each moment
    avail_time = list(traj.time.unique())
    assert (num_steps_tracing_back_from_crash >= 1)
    interval_waiting_to_be_evaluated = []
    for t_idx in range(num_steps_tracing_back_from_crash, -1, -1):
        reference_idx = t_idx + 1
        start_time = avail_time[-reference_idx]
        desired_end_time = start_time + desired_look_ahead_steps
        t_interval = range(start_time, desired_end_time + 1, 1)
        interval_waiting_to_be_evaluated.append(t_interval)

    # Extend the trajectory if needed
    # TODO: how to extend the trajectory when the BV has heading angle?
    if extend_traj_flag:
        extended_traj = extend_traj(traj=traj, extend_steps=extend_steps, extend_delta=extend_delta)

    # Generate three circles position of each vehicle
    extended_traj["front_circle_x"] = extended_traj.apply(lambda row: util_cal_circle_position(row.x, row.y, row.heading, center_point_distance, circle_pos="front")[0], axis=1)
    extended_traj["front_circle_y"] = extended_traj.apply(lambda row: util_cal_circle_position(row.x, row.y, row.heading, center_point_distance, circle_pos="front")[1], axis=1)
    extended_traj["rear_circle_x"] = extended_traj.apply(lambda row: util_cal_circle_position(row.x, row.y, row.heading, center_point_distance, circle_pos="rear")[0], axis=1)
    extended_traj["rear_circle_y"] = extended_traj.apply(lambda row: util_cal_circle_position(row.x, row.y, row.heading, center_point_distance, circle_pos="rear")[1], axis=1)

    # Generate action admissible space for different dangerous levels.
    L_sv_dict, b_sv_dict = {}, {}
    for dangerous_level in dangerous_level_dict.keys():
        action_constraint_percent = dangerous_level_dict[dangerous_level]
        L_sv, b_sv = get_SV_admissible_space_constrs_matrices(acc_x_max_sv=acc_x_max_sv, acc_x_min_sv=acc_x_min_sv,
                                                              acc_y_max_sv=acc_y_max_sv, acc_y_min_sv=acc_y_min_sv,
                                                              action_constraint_percent=action_constraint_percent,
                                                              admissible_space=admissible_space)
        L_sv_dict[dangerous_level], b_sv_dict[dangerous_level] = L_sv, b_sv

    # main loop, evaluate each moment
    one_episode = copy.deepcopy(extended_traj)
    avail_time = list(one_episode.time.unique())
    for t_interval in tqdm(interval_waiting_to_be_evaluated):
        t_interval = sorted(list(set(t_interval) & set(avail_time)))
        current_time = t_interval[0]
        look_ahead_steps = len(t_interval) - 1

        # Check whether it is safe using current SV actions
        truncate_traj = one_episode[(one_episode["time"] >= t_interval[0]) & (one_episode["time"] <= t_interval[-1])]
        evasive_traj_plan_needed_flag = first_step_safety_check(truncate_traj, ax_threshold=initial_check_ax_threshold, ay_threshold=initial_check_ay_threshold,
                                                                steering_degree_threshold=initial_check_steering_degree_threshold)
        if not evasive_traj_plan_needed_flag:
            traj.loc[(traj["veh_id"] == "CAV") & (traj["time"] == current_time), "dangerous_level"] = safe_dangerous_level
            print("Time idx: {0}, evasive_traj_exist: {1}".format(str(current_time), str(True)))
            continue

        # truncate_traj = one_episode[(one_episode["time"]>=t_interval[0]) & ((one_episode["time"]<=t_interval[-1]))]
        # plot_static_illustration_figure(truncate_traj, xlim, plot_box_freq=1,fill_flag=fill_flag,
        # BV_fill_flag=BV_fill_flag,SV_center_point_plot_flag=SV_center_point_plot_flag,
        # crash_BV_color=crash_BV_color,fig_size_factor=fig_size_factor,alpha=alpha,save_fig_flag=True,
        # file_name="E:/2021-01-Safety-Metric-Evaluation/result/MILP/static")
        # break

        # print(t_interval, look_ahead_steps)
        # continue

        # Get surrounding BVs id list
        BVs_traj_tmp = one_episode.groupby('time').get_group(t_interval[0]).reset_index(drop=True)
        BVs_traj_tmp = BVs_traj_tmp[BVs_traj_tmp["veh_id"] != "CAV"].reset_index(drop=True)
        BVs_id_list = BVs_traj_tmp.iloc[:N]["veh_id"].tolist()
        assert (N == len(BVs_id_list))

        # SV dynamics matrices
        SV_initial_moment = one_episode[(one_episode["veh_id"] == "CAV") & (one_episode["time"] == t_interval[0])]
        SV_initial_v, SV_initial_heading, SV_initial_x, SV_initial_y = SV_initial_moment["v"].item(), SV_initial_moment[
            "heading"].item(), SV_initial_moment["x"].item(), SV_initial_moment["y"].item()
        SV_initial_coord = [SV_initial_x, SV_initial_y]
        A, B, F = get_SV_dynamics_transition_matrices(delta, SV_initial_v, SV_initial_heading, SV_initial_coord,
                                                      state_space=state_space)
        F = F.reshape(4, )  # For gurobi settings.

        # Circle state transition matrices
        M_front, N_front = get_circle_transition_matrices(initial_heading=SV_initial_heading,
                                                          center_point_distance=center_point_distance,
                                                          initial_coord=SV_initial_coord, circle_pos="front")
        M_rear, N_rear = get_circle_transition_matrices(initial_heading=SV_initial_heading,
                                                        center_point_distance=center_point_distance,
                                                        initial_coord=SV_initial_coord, circle_pos="rear")
        N_front, N_rear = N_front.reshape(4, ), N_rear.reshape(4, )  # For gurobi settings.

        SV_initial_state = np.array([SV_initial_x, SV_initial_y, SV_initial_v, SV_initial_heading])
        veh_dynamic_matrices_dict = {"A": A, "B": B, "F": F, "SV_initial_state": SV_initial_state, "L_sv_dict": L_sv_dict,
                                     "b_sv_dict": b_sv_dict, "M_front": M_front, "N_front": N_front, "M_rear": M_rear,
                                     "N_rear": N_rear}

        # Get BVs all circles position. Used to determine dangerous in QCP formulation
        BVs_all_circle_pos_array = get_BVs_circles_position(t_interval=t_interval, N=N, BVs_id_list=BVs_id_list, one_episode=one_episode)

        # QCP solving algorithm
        dangerous_level, evasive_traj_exist, evasive_planned_traj, evasive_traj_three_circles = QCP_evasive_traj_planning(
            veh_dynamic_matrices_dict, dangerous_level_dict, BVs_all_circle_pos_array, look_ahead_steps, SV_state_dim, u_dim, N,
            collision_threshold=collision_threshold, a_lb=a_lb, a_ub=a_ub)
        traj.loc[(traj["veh_id"] == "CAV") & (traj["time"] == current_time), "dangerous_level"] = dangerous_level
        print("Time idx: {0}, dangerous level: {1}".format(str(current_time), str(dangerous_level)))

        # Visualize results
        # Static evasive fig.
        file_name = "traj_id_{0}_t_idx_{1}_dangerous_level_{2}".format(str(case_id), str(current_time), str(dangerous_level))
        fig_address = os.path.join(folder_address, file_name)
        if plot_static_evasive_fig_flag and (dangerous_level > safe_dangerous_level):
            plot_static_illustration_figure(truncate_traj, plot_box_freq, fill_flag=SV_static_evasive_fig_fill_flag,
                                            BV_fill_flag=BV_fill_flag, SV_center_point_plot_flag=SV_center_point_plot_flag,
                                            crash_BV_color=crash_BV_color, fig_size_factor=fig_size_factor, alpha=alpha,
                                            save_fig_flag=plot_static_evasive_fig_flag, file_name=fig_address, given_interval=None,
                                            given_POV_id=None, plot_evasive_traj_flag=evasive_traj_exist,
                                            evasive_planned_traj=evasive_planned_traj,
                                            evasive_traj_color=evasive_traj_color, evasive_fill_flag=evasive_fill_flag,
                                            plot_evasive_box=plot_evasive_box,
                                            plot_evasive_traj_line_flag=plot_evasive_traj_line_flag,
                                            extend_traj_color=extend_traj_color,
                                            plot_three_circles_flag=plot_three_circles_flag,
                                            plot_evasive_traj_three_circles_flag=plot_evasive_traj_three_circles_flag * evasive_traj_exist,
                                            evasive_traj_three_circles=evasive_traj_three_circles, radius=radius,
                                            center_point_distance=center_point_distance)

        # Generate evasive trajectory video.
        if plot_evasive_video_flag and evasive_traj_exist and (dangerous_level > safe_dangerous_level):
            # Replay the evasive trajectory
            new_df = generate_evasive_video_df(truncate_traj, evasive_planned_traj, dangerous_level=dangerous_level)
            replay_one_simulation(new_df, str(case_id), metric=None, slow_version=slow_version_evasive_video, save_video_flag=plot_evasive_video_flag,
                                  file_name=fig_address, color_dict=color_dict, whole_traj_flag=False, evasive_traj_flag=True)

    # Replay the overall trajectory video.
    if plot_whole_video_flag:
        whole_traj_file_name = "Whole_traj_id_{0}_video".format(str(case_id))
        whole_traj_fig_address = os.path.join(folder_address, whole_traj_file_name)
        replay_one_simulation(traj, str(case_id), metric=None, slow_version=slow_version_whole_traj_video, save_video_flag=plot_whole_video_flag,
                              file_name=whole_traj_fig_address, color_dict=color_dict, whole_traj_flag=True, evasive_traj_flag=False)

    if save_evaluation_df_flag:
        safety_evaluation_res_df = traj.loc[traj["veh_id"] == "CAV", ["episode", "time", "dangerous_level"]].reset_index(drop=True)
        evaluation_res_df_name = "evaluation_res_df.csv"
        evaluation_res_address = os.path.join(folder_address, evaluation_res_df_name)
        safety_evaluation_res_df.to_csv(evaluation_res_address, index=False)


if __name__ == "__main__":
    QCP_main()

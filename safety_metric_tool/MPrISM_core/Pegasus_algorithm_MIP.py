# This file is to formulate the Pegasus algorithm using mixed integer formulation (to tackle safety constraints).
# Author: Xintao Yan
# Date: 4/23/2021
# Affiliation: Michigan Traffic Lab (MTL)

import gurobipy as gp
from gurobipy import GRB

from safety_metric_tool.MPrISM_core.Pegasus_algorithm import *
from safety_metric_tool.MPrISM_core.Pegasus_algorithm import _map_risk_level
from safety_metric_tool.MPrISM_core.Optimization_based_algorithm_main import _traj_predict, bound_predict_traj

# =================== Parameters =======================
# mpc

NX = 4  # x = x,y,v,heading
NU = 2  # ax, ay
w_x, w_y, w_ax, w_ay = 1, 1, 0.1, 1
# w_x, w_y, w_ax, w_ay = 0., 0., 0., 0.

# Map geometry parameters
lane_number, lane_width = 3, 4
y_min, y_max, y_epsilon = 0, (lane_number - 1) * lane_width, 0.2  # TODO: Hard code the map geometry here.

# Vehicle geometry parameters
# VEH_LENGTH, VEH_WIDTH = 5., 2.
v_max = 40.


def find_cf_front(x_set, horizontal_buffer):
    """
    Find the front boundary
    """
    cf = x_set + horizontal_buffer
    # print(x_set, cf)
    return cf

def _determine_whether_length_or_width_to_use_in_cal_cf_cl_cr(pov_heading, AA_rdbt_data=False):

    if AA_rdbt_data:
        BUFFER1, BUFFER2 = 5., 4
    else:
        BUFFER1, BUFFER2 = 5., 3.

    # TODO: rule-based consideration to determine whether vehicle length/width should be used in different vehicle heading situation.
    # Make sure the heading within [-pi, pi)
    pov_heading_wrap = (pov_heading + np.pi) % (2 * np.pi) - np.pi
    assert (-np.pi <= pov_heading_wrap <= np.pi)

    # whether using the width or length depending on whether the POV is heading horizontally or vertically.
    if -(3/4)*np.pi <= pov_heading_wrap <= -(1/4)*np.pi or (1/4)*np.pi <= pov_heading_wrap <= (3/4)*np.pi:
        cf_param = BUFFER2
        cr_param, cl_param = BUFFER1, BUFFER1
    else:
        cf_param = BUFFER1
        cr_param, cl_param = BUFFER2, BUFFER2

    return cf_param, cr_param, cl_param


# Formulation
def pegasus_algorithm_MIP(input_val, verbose=False, AA_rdbt_data=False):
    big_M = 1e5
    A, B, F, sv_state, pov_state, steps, delta, mu, L_sv, b_sv, L_sv_risk_level_dict, b_sv_risk_level_dict = input_val  # SV global state follows As+Bu+F = s(t+1)

    ini_sv_x, ini_sv_y, ini_sv_v, ini_sv_heading = sv_state[0].item(), sv_state[1].item(), sv_state[2].item(), sv_state[3].item()
    ini_pov_x, ini_pov_y, ini_pov_v, ini_pov_heading = pov_state[0].item(), pov_state[1].item(), pov_state[2].item(), pov_state[3].item()

    initial_state = [ini_pov_x, ini_pov_y, ini_pov_v, ini_pov_heading]
    pred_traj = _traj_predict(initial_state, steps=steps, delta=delta)
    bound_pred_traj = pred_traj
    # bound_pred_traj = bound_predict_traj(pred_traj, y_min=y_min, y_max=y_max, y_epsilon=y_epsilon)

    # Find ry, dy
    ry_set, dy_set = find_ry_dy(bound_pred_traj[1, :], y_min, y_max, y_epsilon)
    # Find rx, dx
    rx_set, dx_set = find_rx_dx(bound_pred_traj[0, :], sv_v=ini_sv_v)
    # Find cf, cl, cr
    cf_param, cr_param, cl_param = _determine_whether_length_or_width_to_use_in_cal_cf_cl_cr(ini_pov_heading, AA_rdbt_data)
    cf_behind = find_cf(bound_pred_traj[0, :], horizontal_buffer=cf_param)
    cf_front = find_cf_front(bound_pred_traj[0, :], horizontal_buffer=cf_param)
    cr = find_cr(bound_pred_traj[1, :], vertical_buffer=cr_param, y_min=y_min, y_max=y_max, y_epsilon=0.2)
    cl = find_cl(bound_pred_traj[1, :], vertical_buffer=cl_param, y_min=y_min, y_max=y_max, y_epsilon=0.2)

    # Create state decision variables bound. s: look_ahead_steps*SV_state_dim
    s_lb, s_ub = np.zeros((NX, steps + 1)), np.zeros((NX, steps + 1))
    for t in range(steps + 1):
        for state_dim_idx in range(NX):
            if (state_dim_idx == 0) or (state_dim_idx == 1):  # x, y position
                s_lb[state_dim_idx, t] = -float('inf')
                s_ub[state_dim_idx, t] = float('inf')
            elif state_dim_idx == 2:  # velocity
                s_lb[state_dim_idx, t] = 0
                s_ub[state_dim_idx, t] = float('inf')
            else:  # heading
                s_lb[state_dim_idx, t] = -float('inf')
                s_ub[state_dim_idx, t] = float('inf')

    # Create a new model
    model = gp.Model("PCM_MIP")
    model.setParam('OutputFlag', 0)
    model.params.NonConvex = 2
    model.Params.MIPGap = 0.05  # 5% Optimality gap.
    model.Params.TimeLimit = 300  # 5 minutes

    # Decision variables
    u = model.addMVar((NU, steps), lb=-10000000., ub=10000000., vtype=GRB.CONTINUOUS, name="u")
    state = model.addMVar((NX, steps + 1), lb=s_lb, ub=s_ub, vtype=GRB.CONTINUOUS, name="state")
    model.update()

    # Objective function
    # auxiliary Rx variables
    Rx_list = model.addMVar((1, steps), lb=0., ub=float('inf'), vtype=GRB.CONTINUOUS, name="Rx_list")
    # assert (ini_pov_x >= ini_sv_x)
    for i in range(steps):
        model.addConstr(Rx_list[0, i] >= 0.)
        model.addConstr(Rx_list[0, i] >= state[0, i + 1]/float(dx_set[i]) - float(rx_set[i]) / float(dx_set[i]))
        # model.addConstr(Rx_list[0, i] == gp.max_(0., state[0, i + 1]/float(dx_set[i]) - float(rx_set[i]) / float(dx_set[i])))
    # auxiliary Ry_2_obj variables
    P_Ry_2 = np.diag(ini_sv_v / (v_max * np.power(dy_set, 2)))
    Ry_2_obj = model.addMVar((1, ), lb=0., ub=float('inf'), vtype=GRB.CONTINUOUS, name="Ry_2_obj")
    # model.addMQConstr(Q=P_Ry_2, c=np.ones(1), xc=Ry_2_obj, sense="=", rhs=0., xQ_L=state[1, 1:] - np.array(ry_set), xQ_R=state[1, 1:] - np.array(ry_set))
    tmp = model.addMVar(state[1, 1:].shape, lb=-float('inf'), ub=float('inf'), vtype=GRB.CONTINUOUS, name="tmp")
    model.addConstr(tmp - state[1, 1:] == -np.array(ry_set))
    model.addMQConstr(Q=P_Ry_2, c=-np.ones(1), sense="=", rhs=0., xQ_L=tmp, xQ_R=tmp, xc=Ry_2_obj)
    # auxiliary ax_obj variables
    ax_obj = model.addMVar((1, ), lb=0., ub=float('inf'), vtype=GRB.CONTINUOUS, name="ax_obj")
    model.addMQConstr(Q=np.diag([1 / ((mu * 10) ** 2)] * steps), c=-np.ones(1), xc=ax_obj, sense="=", rhs=0., xQ_L=u[0, :], xQ_R=u[0, :])
    # auxiliary ay_obj variables
    ay_obj = model.addMVar((1, ), lb=0., ub=float('inf'), vtype=GRB.CONTINUOUS, name="ay_obj")
    model.addMQConstr(Q=np.diag([1 / ((mu * 10) ** 2)] * steps), c=-np.ones(1), xc=ay_obj, sense="=", rhs=0., xQ_L=u[1, :], xQ_R=u[1, :])
    # Set objective
    model.setObjective(w_x * Rx_list.sum() + w_y * Ry_2_obj + w_ax * ax_obj + w_ay * ay_obj, GRB.MINIMIZE)
    # model.setObjective(1, GRB.MINIMIZE)
    model.update()

    # Constraints
    # 1. Kmma circle
    model.addConstrs(L_sv @ u[:, i] <= b_sv.reshape(12, ) for i in range(steps))

    # 2. Vehicle dynamics
    model.addConstr(state[:, 0] == sv_state.reshape(4, ))
    model.addConstrs(A @ state[:, t] + B @ u[:, t] + F.reshape(4, ) == state[:, t + 1] for t in range(steps))

    # 3. Crash constraints
    # Add integer variables
    delta = model.addMVar((4, steps), vtype=GRB.BINARY, name="delta")
    # Behind of the BV
    model.addConstr(state[0, 1:] <= cf_behind + big_M * (1 - delta[0, :]))
    # Front of the BV
    model.addConstr(state[0, 1:] >= cf_front - big_M * (1 - delta[1, :]))
    # Left of the BV
    model.addConstr(state[1, 1:] <= cl + big_M * (1 - delta[2, :]))
    # Right of the BV
    model.addConstr(state[1, 1:] >= cr - big_M * (1 - delta[3, :]))
    # Add constraint: at least one constraint is satisfied at each time
    for t in range(steps):
        model.addConstr(np.ones((1, 4)) @ delta[:, t] >= 1)

    # Optimize model
    model.optimize()

    model_status = model.status
    evasive_traj = None
    if model_status == 2:
        obj_val = model.getObjective().getValue()
        max_pred_acc = max(np.sqrt(u.X[0, :] ** 2 + u.X[1, :] ** 2))
        obj_Rx = (w_x * np.sum(Rx_list.X)).item()
        obj_Ry = (w_y * Ry_2_obj.X).item()
        obj_ax = (w_ax * ax_obj.X).item()
        obj_ay =( w_ay * ay_obj.X).item()
        risk_level = _map_risk_level(u.X, L_sv_risk_level_dict, b_sv_risk_level_dict)
        assert (abs(obj_val - (obj_Rx + obj_Ry + obj_ax + obj_ay)) < 1e-3)
        evasive_traj = state.X
    elif model_status == 3:
        # print("Error: MIP Cannot solve mpc..")
        obj_val, obj_Rx, obj_Ry, obj_ax, obj_ay, max_pred_acc, risk_level = np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, max(list(L_sv_risk_level_dict.keys()))
    elif model_status == 9:
        print("Time Limit. Optimality gap: {0}%.".format(model.ObjBound))
        obj_val, obj_Rx, obj_Ry, obj_ax, obj_ay, max_pred_acc, risk_level = np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, max(list(L_sv_risk_level_dict.keys()))
    else:
        raise ValueError("Model Status Error: the status is {0}".format(model_status))

    return obj_val, obj_Rx, obj_Ry, obj_ax, obj_ay, max_pred_acc, risk_level, evasive_traj, bound_pred_traj


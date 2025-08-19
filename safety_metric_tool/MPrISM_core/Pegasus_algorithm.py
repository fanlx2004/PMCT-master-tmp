import numpy as np
import cvxpy
from cvxopt import matrix, solvers, spmatrix
solvers.options['show_progress'] = False
from safety_metric_tool.MPrISM_core.Optimization_based_algorithm_main import _traj_predict, bound_predict_traj

# =================== Parameters =======================
# mpc
NX = 4  # x = x,y,v,heading
NU = 2  # ax, ay
w_x, w_y, w_ax, w_ay = 1, 1, 0.1, 1

# Map geometry parameters
lane_number, lane_width = 3, 4
y_min, y_max, y_epsilon = 0, (lane_number - 1) * lane_width, 0.2  # TODO: Hard code the map geometry here.

# Vehicle geometry parameters
VEH_LENGTH, VEH_WIDTH = 5., 2.
v_max = 40.


# =================== Functions ========================
# Model the problem using cvxpy
def pegasus_algorithm(input_val, verbose=False):
    A, B, F, sv_state, pov_state, steps, delta, mu, L_sv, b_sv, L_sv_risk_level_dict, b_sv_risk_level_dict = input_val  # SV global state follows As+Bu+F = s(t+1)

    ini_sv_x, ini_sv_y, ini_sv_v, ini_sv_heading = sv_state[0].item(), sv_state[1].item(), sv_state[2].item(), sv_state[3].item()
    ini_pov_x, ini_pov_y, ini_pov_v, ini_pov_heading = pov_state[0].item(), pov_state[1].item(), pov_state[2].item(), pov_state[3].item()

    initial_state = [ini_pov_x, ini_pov_y, ini_pov_v, ini_pov_heading]
    pred_traj = _traj_predict(initial_state, steps=steps, delta=delta)
    bound_pred_traj = bound_predict_traj(pred_traj, y_min=y_min, y_max=y_max, y_epsilon=y_epsilon)

    # Find ry, dy
    ry_set, dy_set = find_ry_dy(bound_pred_traj[1, :], y_min, y_max, y_epsilon)
    # Find rx, dx
    rx_set, dx_set = find_rx_dx(bound_pred_traj[0, :], sv_v=ini_sv_v)
    # Find cf, cl, cr
    cf = find_cf(bound_pred_traj[0, :], VEH_LENGTH)
    cr = find_cr(bound_pred_traj[1, :], VEH_WIDTH, y_min, y_max, y_epsilon=0.2)
    cl = find_cl(bound_pred_traj[1, :], VEH_WIDTH, y_min, y_max, y_epsilon=0.2)

    # Optimization
    u, state = cvxpy.Variable((NU, steps)), cvxpy.Variable((NX, steps + 1))

    # Objective function
    Rx_list = []
    if ini_pov_x >= ini_sv_x:
        for i in range(steps):
            Rx = cvxpy.maximum(0, (state[0, i + 1] - float(rx_set[i])) / float(dx_set[i]))
            Rx_list.append(Rx)

    P_Ry_2 = np.diag(ini_sv_v / (v_max * np.power(dy_set, 2)))
    Ry_2_obj = cvxpy.quad_form(state[1, 1:] - np.array(ry_set), P_Ry_2)
    ax_obj = cvxpy.quad_form(u[0, :], np.diag([1 / ((mu * 10) ** 2)] * steps))
    ay_obj = cvxpy.quad_form(u[1, :], np.diag([1 / ((mu * 10) ** 2)] * steps))
    obj = w_x * cvxpy.sum(Rx_list) + w_y * Ry_2_obj + w_ax * ax_obj + w_ay * ay_obj

    # Constraints
    constraints = []
    x_ini_flag, y_ini_flag = False, False  # At least one distance should satisfied at initial condition.
    if ini_pov_x - VEH_LENGTH >= ini_sv_x: x_ini_flag = True
    if abs(ini_pov_y - ini_sv_y) >= VEH_WIDTH: y_ini_flag = True

    # If both x, y initial conditions are satisfied, choose the lateral one.
    if x_ini_flag and y_ini_flag:
        x_ini_flag = False

    # 1. Kmma circle
    for i in range(steps):
        constraints += [L_sv @ u[:, i] <= b_sv.reshape(12, )]

    # 2. x bound
    if x_ini_flag:
        constraints += [state[0, 1:] <= cf]

    # 3. y bound
    y_threshold = 4
    if y_ini_flag and ini_pov_y - VEH_WIDTH >= ini_sv_y:
        constraints += [state[1, 1:] <= cl]
    elif y_ini_flag and ini_pov_y + VEH_WIDTH < ini_sv_y:
        constraints += [state[1, 1:] >= cr]
    # Road boundary
    # constraints += [state[1, 1:] <= y_max + y_threshold]
    # constraints += [state[1, 1:] >= y_min - y_threshold]

    # If both x,y not satisfy the initial condition, choose the more potential one as the bound.
    if (not x_ini_flag) and (not y_ini_flag) and (ini_sv_x < ini_pov_x):
        x_margin, y_margin = (ini_pov_x - ini_sv_x) - VEH_LENGTH, abs(ini_pov_y - ini_sv_y) - VEH_WIDTH
        if abs(x_margin) <= abs(y_margin):
            constraints += [state[0, 1:] <= (cf + abs(x_margin))]
            assert (abs(x_margin) < 0.5)
        else:
            if ini_pov_y > ini_sv_y:
                constraints += [state[1, 1:] <= (cl + abs(y_margin))]
            else:
                constraints += [state[1, 1:] >= (cr - abs(y_margin))]
            assert (abs(y_margin) < 0.5)

    # 4. Vehicle dynamics
    constraints += [state[:, 0] == sv_state.reshape(4, )]
    for t in range(steps):
        constraints += [A @ state[:, t] + B @ u[:, t] + F.reshape(4, ) == state[:, t + 1]]

    prob = cvxpy.Problem(cvxpy.Minimize(obj), constraints)
    try:
        prob.solve(solver=cvxpy.ECOS, verbose=verbose)
    except:
        prob.solve(solver=cvxpy.CVXOPT, verbose=True)
        print(sv_state, pov_state)

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        obj_val = prob.value
        max_pred_acc = max(np.sqrt(u.value[0, :] ** 2 + u.value[1, :] ** 2))
        obj_Ry = w_y * (state.value[1, 1:] - np.array(ry_set)).T @ P_Ry_2 @ (state.value[1, 1:] - np.array(ry_set))
        obj_ax = w_ax * (u.value[0, :].T @ np.diag([1 / ((mu * 10) ** 2)] * steps) @ u.value[0, :])
        obj_ay = w_ay * (u.value[1, :].T @ np.diag([1 / ((mu * 10) ** 2)] * steps) @ u.value[1, :])
        obj_Rx = 0.0
        risk_level = _map_risk_level(u.value, L_sv_risk_level_dict, b_sv_risk_level_dict)
        if ini_pov_x >= ini_sv_x:
            for i in range(steps):
                Rx = max(0, (state.value[0, i + 1] - float(rx_set[i])) / float(dx_set[i]))
                obj_Rx += w_x * Rx
        try:
            assert (abs(obj_val - (obj_Ry + obj_ax + obj_ay + obj_Rx)) < 1e-3)
        except:
            print(obj_val, obj_Ry + obj_ax + obj_ay + obj_Rx)
            raise ValueError("a")
    else:
        print("Error: Cannot solve mpc..")
        obj_val, obj_Rx, obj_Ry, obj_ax, obj_ay, max_pred_acc, risk_level = np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, max(list(L_sv_risk_level_dict.keys()))

    return obj_val, obj_Rx, obj_Ry, obj_ax, obj_ay, max_pred_acc, risk_level, state.value, bound_pred_traj


def _map_risk_level(acc_array, L_sv_risk_level_dict, b_sv_risk_level_dict):
    """
    This function map the Pegasus predicted acceleration to the risk level

    :param ax_array: the longitudinal acceleration array
    :param ay_array: the lateral acceleration array
    :param L_sv_risk_level_dict: key: risk level, value: the L correspond to that risk level
    :param b_sv_risk_level_dict: key: risk level, value: the b correspond to that risk level
    :return: risk level
    """
    assert (L_sv_risk_level_dict.keys() == b_sv_risk_level_dict.keys())
    risk_level = None
    for risk_level_tmp in L_sv_risk_level_dict.keys():
        L_sv_risk_level, b_sv_risk_level = L_sv_risk_level_dict[risk_level_tmp], b_sv_risk_level_dict[risk_level_tmp] + 1e-5  # Add small value to avoid numerical issue.
        if (L_sv_risk_level @ acc_array <= b_sv_risk_level).all():
            risk_level = risk_level_tmp - 1
            break
    try:
        assert (risk_level <= max(L_sv_risk_level_dict.keys()) - 1)
    except:
        print("acc_array exceed Kamm circle:", acc_array)
        # pass
    return risk_level


def find_cl(y_set, vertical_buffer, y_min, y_max, y_epsilon=0.2):
    """
    Find the left boundary
    """
    y_min_hat, y_max_hat = y_min + y_epsilon, y_max - y_epsilon  # the modified bound considering epsilon
    cl = y_set - vertical_buffer
    # cl = np.clip(y_set - veh_width, y_min_hat, y_max_hat)
    # cl = np.clip(y_set, y_min_hat, y_max_hat)
    # print(y_set, cl)
    return cl


def find_cr(y_set, vertical_buffer, y_min, y_max, y_epsilon=0.2):
    """
    Find the right boundary
    """
    y_min_hat, y_max_hat = y_min + y_epsilon, y_max - y_epsilon  # the modified bound considering epsilon
    cr = y_set + vertical_buffer
    # cr = np.clip(y_set + veh_width, y_min_hat, y_max_hat)
    # cr = np.clip(y_set, y_min_hat, y_max_hat)
    # print(y_set, cr)
    return cr


def find_cf(x_set, horizontal_buffer):
    """
    Find the front boundary
    """
    cf = x_set - horizontal_buffer
    # print(x_set, cf)
    return cf


def find_rx_dx(x_set, sv_v):
    """
    Find the reference car-following position based on the initial sv
    velocity and predicted pov position
    The Germany recommended car-following distance is 0.5*v/(km/h)
    """
    steps = x_set.size
    v_km_h = 3.6 * sv_v
    dx = 0.5 * v_km_h
    assert (dx > 0)
    dx_set = [dx] * steps
    rx_set = []
    for T in range(steps):
        x = x_set[T]
        rx = x - dx
        rx_set.append(rx)
        # print(x, rx, dx)
    return rx_set, dx_set


def find_ry_dy(y_set, y_min, y_max, y_epsilon=0.2):
    """
    Find the maximum lateral reference position and the maximum lateral deviation.
    Input:
        1. predicted trajectory y position at each timestep
        2. ymin coordinate, ymax coordinate given by geometry
        3. y_epsilon is the buffer to position error. e.g. the max y is 8
            but the data could be 8.01, etc.
    Output:
        1. the maximum lateral distance reference position
        2. the maximum lateral deviation at each timestep
    """
    steps = y_set.size
    y_min_hat, y_max_hat = y_min - y_epsilon, y_max + y_epsilon  # the modified bound considering epsilon
    ry_set, dy_set = [], []
    for T in range(steps):
        y = y_set[T]
        # assert (y <= y_max_hat and y >= y_min_hat)
        ry = y_min_hat if np.argmax([np.abs(y - y_min_hat), np.abs(y_max_hat - y)]) == 0 else y_max_hat
        ry_set.append(ry)
        dy = max(np.abs(ry - y_min_hat), np.abs(y_max_hat - ry))
        dy_set.append(dy)
        # print(y, ry, dy)
    return ry_set, dy_set


def get_rotation_offset_matrix(x, y, heading):
    R = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
    O = np.array([[x], [y]])
    return R, O


def get_A(delta, v_til):
    """
    Calculate the A matrix in Eq. (8) in MPrISM paper.
    """
    A_sv = np.array([[1, 0, delta, 0], [0, 1, 0, v_til * delta], [0, 0, 1, 0], [0, 0, 0, 1]])
    return A_sv


def get_B(delta, v_til):
    """
    Calculate the B matrix in Eq. (8) in MPrISM paper.
    """
    B_sv = np.array([[0.5 * delta ** 2, 0], [0, 0.5 * delta ** 2], [delta, 0], [0, delta / v_til]])
    return B_sv

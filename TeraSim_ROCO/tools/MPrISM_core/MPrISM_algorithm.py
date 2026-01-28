# This file implements the MPrISM algorithm to calculate the safety metric
# Author: Xintao Yan
# Date: 2/22/2021
# Affiliation: Michigan Traffic Lab (MTL)

import numpy as np
import copy
from cvxopt import matrix, solvers, spmatrix
solvers.options['show_progress'] = False

# ================== Algorithm =====================
def MPrISM_algorithm_evaluate_traj(input_val):
    """Calculate the MPrTTC w.r.t. a given POV.
    Args:
        sim_time, x_sv, initial_sv_offset, initial_sv_heading, x_pov, initial_pov_offset, initial_pov_heading, delta, steps, crash_threshold = input_val
        sim_time: Current simulation time.
        x_sv (np.array size:4*1): the state of the SV in the local(natural) coordinates system, [x_loc, y_loc, v, heading_loc]: x_loc: local x coordinates (normally 0),
        y_loc: local y coordinates (normally 0), v: velocity, heading_loc: local heading (normally 0).
        initial_sv_offset (list): the position of the origin of the local coordinates w.r.t the global coordinates system, [x_offset, y_offset]: x_offset: the SV center x
        coordinates in the global coordinates system, y_offset: the SV center y coordinates in the global coordinates system.
        initial_sv_heading: the SV heading angle (anticlockwise) of the local coordinates system w.r.t the global coordinates system. Use Radian system (e.g., 1/6 pi).
        x_pov, initial_pov_offset, initial_pov_heading are the same information for the POV follows the same format as SV.
        delta: time resolution parameter of the MPrISM algorithm. (e.g., 0.02s or 0.1s).
        steps: number of look-ahead steps parameter. (e.g., 50 steps or 10 steps).
        crash_threshold: the parameter of the MPrISM. (e.g., 4m).
    """
    # Input val
    sim_time, POV_id, x_sv, initial_sv_offset, initial_sv_heading, x_pov, initial_pov_offset, initial_pov_heading, delta, steps, crash_threshold, L_sv, b_sv, L_pov, \
    b_pov, plot_MPrISM_planned_traj_video_flag = input_val

    # Construct Rotation and Offset Matrix for the SV and POV
    R_sv, R_pov = np.array([[np.cos(initial_sv_heading), -np.sin(initial_sv_heading)], [np.sin(initial_sv_heading), np.cos(initial_sv_heading)]]), np.array(
        [[np.cos(initial_pov_heading), -np.sin(initial_pov_heading)], [np.sin(initial_pov_heading), np.cos(initial_pov_heading)]])
    O_sv, O_pov = np.array([[initial_sv_offset[0]], [initial_sv_offset[1]]]), np.array([[initial_pov_offset[0]], [initial_pov_offset[1]]])

    # Use the initial velocity as the constant velocity
    v_til_sv, v_til_pov = x_sv[2, 0], x_pov[2, 0]
    if np.abs(v_til_sv) < 5:
        if v_til_sv >= 0:
            v_til_sv = 5
        else:
            v_til_sv = -5

    A_sv, B_sv = get_A(delta, v_til_sv), get_B(delta, v_til_sv)
    A_pov, B_pov = get_A(delta, v_til_pov), get_B(delta, v_til_pov)

    time = None  # The output MPrTTC.
    # Time incremental scheme to find the MPrTC.
    for T in range(1, steps + 1, 1):
        # Generate Vehicle dynamics transition matrix A^hat, B^hat
        # x' = A^hat @ x + B_hat @ u, x is the initial state, u is the actions along the interval
        A_hat_sv, A_hat_pov = np.linalg.matrix_power(A_sv, T), np.linalg.matrix_power(A_pov, T)
        B_hat_list_sv, B_hat_list_pov = [], []
        for i in range(T - 1, -1, -1):
            B_hat_list_sv.append(np.linalg.matrix_power(A_sv, i) @ B_sv)
            B_hat_list_pov.append(np.linalg.matrix_power(A_pov, i) @ B_pov)
        B_hat_sv, B_hat_pov = np.hstack(B_hat_list_sv), np.hstack(B_hat_list_pov)

        # Transform the problem to a QP, derive all coefficent matrices
        C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # Transformation matrix to get the first 2-d state
        P, Q, R, U, V, H = get_PQRUVH(A_hat_sv, B_hat_sv, A_hat_pov, B_hat_pov, x_sv, x_pov, C, R_sv, R_pov, O_sv, O_pov)

        # Step 1
        # Solve the critical point first to see which cases it belongs to
        u_critical_sv, u_critical_pov = QP_solve_critical_point(P, Q, R, U, V)
        critical_obj = func_J(u_critical_sv, u_critical_pov, P, Q, R, U, V, H)
        if critical_obj > crash_threshold ** 2:
            continue
            # print("critical_obj=", critical_obj)

        # Determine whether this solution belongs to the feasible set
        u_critical_sv_feasible_flag = (((L_sv @ (u_critical_sv.reshape(-1, 2)).T) - b_sv) <= 0).all()
        u_critical_pov_feasible_flag = (((L_pov @ (u_critical_pov.reshape(-1, 2)).T) - b_pov) <= 0).all()
        case_2_flag = u_critical_sv_feasible_flag and u_critical_pov_feasible_flag

        # Kmma circle constraints
        num_time_step = int(u_critical_sv.shape[0] / 2)
        assert (np.abs(num_time_step - u_critical_sv.shape[0] / 2) < 1e-5)
        tmp = np.kron(np.eye(num_time_step), L_sv)
        G_kmma_circle = np.vstack([np.hstack([np.kron(np.eye(num_time_step), L_sv), np.zeros(tmp.shape)]),
                                   np.hstack([np.zeros(tmp.shape), np.kron(np.eye(num_time_step), L_pov)])])
        h_kmma_circle = np.vstack([np.kron(np.ones((num_time_step, 1)), b_sv),
                                   np.kron(np.ones((num_time_step, 1)), b_pov)])

        # If case 1, use AGD
        if not case_2_flag:
            # print("Case 1")
            # Get a initialized feasible solution to start.
            initial_u_sv_AGD, initial_u_pov_AGD = solve_chebyshev_center(G_kmma_circle, h_kmma_circle)
            # u_sv, u_pov, u_sv_list, u_pov_list, obj_val_list, gradient = AGD_adam(initial_u_sv_AGD, initial_u_pov_AGD, P, Q, R, U, V, H, G_sv=L_sv, G_pov=L_pov, h_sv=b_sv,
            # h_pov=b_pov)
            # Perform AGD algorithm.
            u_sv, u_pov, u_sv_list, u_pov_list = AGD_adam(initial_u_sv_AGD, initial_u_pov_AGD, P, Q, R, U, V, H, G_sv=L_sv, G_pov=L_pov, h_sv=b_sv, h_pov=b_pov)
            # plt.plot(range(len(obj_val_list)), obj_val_list)
            # plt.plot(range(len(gradient)), gradient)
            distance_square = func_J(u_sv, u_pov, P, Q, R, U, V, H)

        # Case 2
        if case_2_flag:
            # print("Case 2")
            L_extra = np.ones((1, u_critical_sv.size + u_critical_pov.size))
            for i in range(u_critical_sv.size):
                if u_critical_sv[i, 0] != 0:
                    L_extra[0, i] = - (np.sum(u_critical_sv[:i]) + np.sum(u_critical_sv[i + 1:]) +
                                       np.sum(u_critical_pov)) / u_critical_sv[i, 0]
                    break
            assert (-1e-10 <= (L_extra @ np.vstack([u_critical_sv, u_critical_pov])) <= 1e-10)

            # Get initial points
            G_whold_leq, h_whole_leq = np.vstack([G_kmma_circle, L_extra]), np.vstack([h_kmma_circle, np.zeros((1, 1))])
            G_whold_geq, h_whole_geq = np.vstack([G_kmma_circle, -L_extra]), np.vstack([h_kmma_circle, np.zeros((1, 1))])

            initial_u_sv_AGD_leq, initial_u_pov_AGD_leq = solve_chebyshev_center(G_whold_leq, h_whole_leq)
            initial_u_sv_AGD_geq, initial_u_pov_AGD_geq = solve_chebyshev_center(G_whold_geq, h_whole_geq)

            assert ((G_whold_leq @ np.vstack([initial_u_sv_AGD_leq, initial_u_pov_AGD_leq]) <= h_whole_leq).all())
            assert ((G_whold_geq @ np.vstack([initial_u_sv_AGD_geq, initial_u_pov_AGD_geq]) <= h_whole_geq).all())

            # Branch 1, leq
            u_sv_branch1, u_pov_branch1, u_sv_list_branch1, u_pov_list_branch1 = AGD_adam(initial_u_sv_AGD_leq, initial_u_pov_AGD_leq, P, Q, R, U, V, H, G_sv=L_sv, G_pov=L_pov,
                                                                                          h_sv=b_sv, h_pov=b_pov, G_extra=L_extra, case2_flag=True, sign="leq")

            distance_square_branch1 = func_J(u_sv_branch1, u_pov_branch1, P, Q, R, U, V, H)

            # Branch 2, geq
            u_sv_branch2, u_pov_branch2, u_sv_list_branch2, u_pov_list_branch2 = AGD_adam(initial_u_sv_AGD_geq, initial_u_pov_AGD_geq, P, Q, R, U, V, H, G_sv=L_sv, G_pov=L_pov,
                                                                                          h_sv=b_sv, h_pov=b_pov, G_extra=L_extra, case2_flag=True, sign="geq")
            distance_square_branch2 = func_J(u_sv_branch2, u_pov_branch1, P, Q, R, U, V, H)

            # print("Case 2, time {0}, dis: {1}, {2}".format(sim_time, np.sqrt(distance_square_branch1), np.sqrt(distance_square_branch2)))
            if distance_square_branch1 >= distance_square_branch2:
                u_sv, u_pov, distance_square = u_sv_branch1, u_pov_branch1, distance_square_branch1
            else:
                u_sv, u_pov, distance_square = u_sv_branch2, u_pov_branch2, distance_square_branch2

        if distance_square <= (crash_threshold ** 2):
            # print(sim_time, np.sqrt(distance_square))
            time = T * delta
            break

    # Generate the MPrISM planned traj if it is dangerous and want to plot
    SV_global_state_array, POV_global_state_array = None, None
    if time is not None and plot_MPrISM_planned_traj_video_flag:
        SV_ini_state, POV_ini_state = [x_sv, initial_sv_offset, initial_sv_heading], [x_pov, initial_pov_offset, initial_pov_heading]
        SV_local_state_array, SV_global_state_array = MPrISM_dynamics(SV_ini_state, u_sv, delta=delta, T=T)
        POV_local_state_array, POV_global_state_array = MPrISM_dynamics(POV_ini_state, u_pov, delta=delta, T=T)

    if time is None:
        assert (T == steps)
        time = 9999999999.0  # If the situation is not dangerous, give a very large MPrTTC which indicates safe

    return sim_time, time, T, POV_id, SV_global_state_array, POV_global_state_array  # pov_pos, sv_v  pov_v, sv_v
# ====================== Functions ======================
def solve_chebyshev_center(G_whole, h_whole):
    """
    Get the chebyshev center of the LMIs to get a feasible interior points
    G_whole @ [u_sv, u_pov] \leq h_whole is all of the LMI

    min (r)
    s.t. g_i.T@xc + r||g_i||2 \leq h_whole
    """
    c = np.zeros((G_whole.shape[1] + 1, 1))
    c[-1, 0] = -1

    norm = np.linalg.norm(G_whole, axis=1).reshape(-1, 1)
    A = matrix(np.hstack([G_whole, norm]), tc="d")
    b = matrix(h_whole, tc="d")
    c = matrix(c, tc="d")

    sol = solvers.lp(c, A, b)
    chebyshev_center = np.array(sol['x'])

    num_var = int(G_whole.shape[1] / 2)
    assert (np.abs(num_var - G_whole.shape[1] / 2) < 1e-5)
    u_sv, u_pov = chebyshev_center[:num_var], chebyshev_center[num_var:-1]

    return u_sv, u_pov


def QP_solve_critical_point(P, Q, R, U, V):
    """
    Solve the critical problem to decide which cases belong to
    """
    M = 2 * np.hstack([np.vstack([Q, 0.5 * R]),
                       np.vstack([0.5 * R.T, P])])
    N = np.vstack([V, U])
    M_matrix = matrix(M, tc="d")
    N_matrix = matrix(N, tc="d")

    G1, G2 = np.eye(M.shape[0]), -np.eye(M.shape[0])
    h1, h2 = 10000 * np.ones((M.shape[0], 1)), 10000 * np.ones((M.shape[0], 1))
    G_matrix = matrix(np.vstack([G1, G2]), tc="d")
    h_matrix = matrix(np.vstack([h1, h2]), tc="d")

    res = np.array(solvers.qp(P=M_matrix, q=N_matrix, G=G_matrix, h=h_matrix)['x'])
    u_sv, u_pov = res[:int(0.5 * (len(res)))].reshape(-1, 1), res[int(0.5 * (len(res))):].reshape(-1, 1)
    return u_sv, u_pov


def func_J(u_sv, u_pov, P, Q, R, U, V, H):
    obj = u_pov.T @ P @ u_pov + u_sv.T @ Q @ u_sv + u_pov.T @ R @ u_sv + U.T @ u_pov + V.T @ u_sv + H
    return obj


def grad_J_sv(u_sv, u_pov, P, Q, R, U, V):
    """
    Calculate the gradient of the objective function for the subject vehicle
    """
    grad = (Q + Q.T) @ u_sv + (R.T @ u_pov) + V
    return grad


def grad_J_pov(u_sv, u_pov, P, Q, R, U, V):
    """
    Calculate the gradient of the objective function for the POV
    """
    grad = (P + P.T) @ u_pov + (R @ u_sv) + U
    return grad


def grad_J_critical_point(u_sv, u_pov, P, Q, R, U, V):
    """
    Calculate the gradient of the objective function for the critical point problem
    x = [u_sv, u_pov].T, the objective function can be written as
    x.T @ A @ x + x.T @ B @ x + C @ x where
    A = [[Q,0], [0,P]], B = [[0,0],[R,0]], C = [V.T, U.T]
    """
    A = np.hstack([np.vstack([Q, np.zeros((P.shape[0], Q.shape[1]))]),
                   np.vstack([np.zeros((Q.shape[0], P.shape[1])), P])])

    B = np.hstack([np.vstack([np.zeros(Q.shape), np.zeros((P.shape[0], Q.shape[1]))]),
                   np.vstack([R, np.zeros((P.shape))])])

    C = np.vstack([V, U]).T

    x = np.vstack([u_sv, u_pov])
    grad = (A + A.T + B + B.T) @ x + C.T
    return grad


def solve_QP_projection(x_prev, G, h):
    """
    Solve the projection of the 1 case
    Solve:
    0.5||x_prev-x_proj||2
    s.t. G @ x_proj <= h
    """
    solvers.options['show_progress'] = False
    assert (x_prev.shape[1] == 1)
    num_time_step = int(x_prev.shape[0] / 2)
    assert (np.abs(num_time_step - x_prev.shape[0] / 2) < 1e-5)

    P_matrix = matrix(np.eye(x_prev.shape[0]), tc="d")
    q_matrix = matrix(-x_prev, tc="d")

    G = np.kron(np.eye(num_time_step), G)
    h = np.kron(np.ones((num_time_step, 1)), h)

    G_matrix = matrix(G, tc="d")
    h_matrix = matrix(h, tc="d")

    res = np.array(solvers.qp(P=P_matrix, q=q_matrix, G=G_matrix, h=h_matrix)['x'])
    x_proj = res.reshape(-1, 1)
    return x_proj


def solve_QP_projection_case2(x_prev_sv, x_prev_pov, G, h, G_extra_whole=None,
                              pov_flag=False, sv_flag=False, sign="leq"):
    """
    Solve the projection of the 2 case
    Solve:
    0.5||x_prev-x_proj||2
    s.t. G @ x_proj <= h
    """
    assert (G_extra_whole is not None)
    assert (pov_flag * sv_flag == 0 and pov_flag + sv_flag == 1)

    if pov_flag:
        G_extra = G_extra_whole[0, x_prev_sv.size:]
        h_extra = - G_extra_whole[0, :x_prev_sv.size] @ x_prev_sv
        q_matrix = matrix(-x_prev_pov, tc="d")

    elif sv_flag:
        G_extra = G_extra_whole[0, :x_prev_sv.size]
        h_extra = - G_extra_whole[0, x_prev_sv.size:] @ x_prev_pov
        q_matrix = matrix(-x_prev_sv, tc="d")

    solvers.options['show_progress'] = False
    assert (x_prev_sv.shape[1] == 1)
    num_time_step = int(x_prev_sv.shape[0] / 2)
    assert (np.abs(num_time_step - x_prev_sv.shape[0] / 2) < 1e-5)

    P_matrix = matrix(np.eye(x_prev_sv.shape[0]), tc="d")

    G_origin = np.kron(np.eye(num_time_step), G)
    h_origin = np.kron(np.ones((num_time_step, 1)), h)
    G_extra = G_extra if sign == "leq" else -G_extra
    h_extra = h_extra if sign == "leq" else -h_extra

    G_matrix = matrix(np.vstack([G_origin, G_extra]), tc="d")
    h_matrix = matrix(np.vstack([h_origin, h_extra]), tc="d")

    res = np.array(solvers.qp(P=P_matrix, q=q_matrix, G=G_matrix, h=h_matrix)['x'])
    x_proj = res.reshape(-1, 1)
    return x_proj


def AGD_adam(initial_u_sv, initial_u_pov, P, Q, R, U, V, H, G_sv, G_pov, h_sv, h_pov,
             G_extra=None, case2_flag=False, sign="leq"):
    """
    AGD version of ADAM
    Gx \leq h
    """
    if case2_flag or G_extra: assert (G_extra is not None and case2_flag)

    alpha = 0.01
    beta_1 = 0.9
    beta_2 = 0.999  # initialize the values of the parameters
    epsilon = 1e-8

    u_sv, u_pov = initial_u_sv, initial_u_pov  # initialize the vector
    m_t_sv, m_t_pov = 0, 0
    v_t_sv, v_t_pov = 0, 0
    Adam_iter_num = 0

    # Quit criterior
    tolerance = 1e-3
    min_steps, max_steps = 40, 200
    obj_change_tol = 1e-3
    sv_proj_flag = False

    constraint_tol = 0
    u_sv_list, u_pov_list = [u_sv], [u_pov]

    proj_pov_num, proj_sv_num = 0, 0
    obj_val_list = [func_J(u_sv, u_pov, P, Q, R, U, V, H)]
    gradient = []
    # print(obj_val_list)
    while (1):  # till it gets converged
        Adam_iter_num += 1
        joint_action = np.vstack([u_sv, u_pov])
        obj_val_prev = func_J(u_sv, u_pov, P, Q, R, U, V, H)
        obj_val_list.append(obj_val_prev.item())
        # POV
        g_t_pov = grad_J_pov(u_sv, u_pov, P, Q, R, U, V)  # computes the gradient of the object function
        m_t_pov = beta_1 * m_t_pov + (1 - beta_1) * g_t_pov  # updates the moving averages of the gradient
        v_t_pov = beta_2 * v_t_pov + (1 - beta_2) * (g_t_pov ** 2)  # updates the moving averages of the squared gradient
        m_cap_pov = m_t_pov / (1 - (beta_1 ** Adam_iter_num))  # calculates the bias-corrected estimates
        v_cap_pov = v_t_pov / (1 - (beta_2 ** Adam_iter_num))  # calculates the bias-corrected estimates
        u_pov_prev = copy.deepcopy(u_pov)
        u_pov = u_pov - (alpha * m_cap_pov) / (np.sqrt(v_cap_pov) + epsilon)  # updates the u_pov
        gradient.append(np.linalg.norm((alpha * m_cap_pov) / (np.sqrt(v_cap_pov) + epsilon)))
        # Check whether needs to do projection
        u_pov_feasible_flag = (((G_pov @ (u_pov.reshape(-1, 2)).T) - h_pov) <= constraint_tol).all()
        # If it is case2, check the extra constraint also
        if u_pov_feasible_flag and case2_flag:
            u_pov_feasible_flag_extra = (G_extra @ joint_action <= 0).all() if sign == "leq" else \
                (G_extra @ joint_action >= 0).all()
            u_pov_feasible_flag = (u_pov_feasible_flag and u_pov_feasible_flag_extra)
        # Do projection if necessary
        if not u_pov_feasible_flag:
            if not case2_flag:
                # print("pov_b4proj", u_pov)
                proj_pov_num += 1
                u_pov = solve_QP_projection(u_pov, G=G_pov, h=h_pov)
                # print("pov_a4proj",u_pov)
            else:
                u_pov = solve_QP_projection_case2(u_sv, u_pov, G=G_pov, h=h_pov,
                                                  G_extra_whole=G_extra, pov_flag=True, sv_flag=False, sign=sign)

        joint_action = np.vstack([u_sv, u_pov])
        # SV
        g_t_sv = grad_J_sv(u_sv, u_pov, P, Q, R, U, V)  # computes the gradient of the object function
        m_t_sv = beta_1 * m_t_sv + (1 - beta_1) * g_t_sv  # updates the moving averages of the gradient
        v_t_sv = beta_2 * v_t_sv + (1 - beta_2) * (g_t_sv ** 2)  # updates the moving averages of the squared gradient
        m_cap_sv = m_t_sv / (1 - (beta_1 ** Adam_iter_num))  # calculates the bias-corrected estimates
        v_cap_sv = v_t_sv / (1 - (beta_2 ** Adam_iter_num))  # calculates the bias-corrected estimates
        u_sv_prev = copy.deepcopy(u_sv)
        u_sv = u_sv + (alpha * m_cap_sv) / (np.sqrt(v_cap_sv) + epsilon)  # updates the u_sv
        # Check whether needs to do projection
        u_sv_feasible_flag = (((G_sv @ (u_sv.reshape(-1, 2)).T) - h_sv) <= constraint_tol).all()
        # If it is case2, check the extra constraint also
        if u_sv_feasible_flag and case2_flag:
            u_sv_feasible_flag_extra = (G_extra @ joint_action <= 0).all() if sign == "leq" else \
                (G_extra @ joint_action >= 0).all()
            u_sv_feasible_flag = (u_sv_feasible_flag and u_sv_feasible_flag_extra)
        # Do projection if necessary
        if not u_sv_feasible_flag:
            if not case2_flag:
                proj_sv_num += 1
                u_sv = solve_QP_projection(u_sv, G=G_sv, h=h_sv)
                sv_proj_flag = True
            else:
                u_sv = solve_QP_projection_case2(u_sv, u_pov, G=G_sv, h=h_sv,
                                                 G_extra_whole=G_extra, pov_flag=False, sv_flag=True, sign=sign)

        u_pov_list.append(u_pov)
        u_sv_list.append(u_sv)
        obj_val = func_J(u_sv, u_pov, P, Q, R, U, V, H)

        # print("Obj value change:", np.abs(obj_val_prev-obj_val))
        # print("Old obj val:", obj_val_prev, "New obj val:", obj_val)
        # print("Old sv action:", u_sv_prev, "New sv action:", u_sv)
        # print("Old pov action:", u_pov_prev, "New sv action:", u_pov)
        # print("sv gradient:", g_t_sv)
        # print("pov gradient:",g_t_pov)

        if (np.abs(obj_val_prev - obj_val) < obj_change_tol and Adam_iter_num >= min_steps and sv_proj_flag):  # checks if it is converged or not
            break
        if (Adam_iter_num >= max_steps):
            break
        # if((np.abs(u_pov-u_pov_prev)<tolerance).all() and (np.abs(u_sv-u_sv_prev)<tolerance).all()): #checks if it is converged or not
        #     break

    assert (u_sv.shape == u_pov.shape)
    # print("Number of Iterations = ",Adam_iter_num, "Proj pov num = ",proj_pov_num, "Proj sv num = ",proj_sv_num,"\n")
    # print("Minima is at = ",u_sv, u_pov)
    # print("Minimum value of Cost Function= ",func_J(u_sv, u_pov, P, Q, R, U, V, H))
    return u_sv, u_pov, u_sv_list, u_pov_list  # , obj_val_list


def get_L_min(a):
    """
    Input the acceleration and out put the Lmin array
    """
    L_min = np.array([[np.cos(7 / 12 * np.pi) / np.abs(a)],
                      [np.cos(9 / 12 * np.pi) / np.abs(a)],
                      [np.cos(11 / 12 * np.pi) / np.abs(a)]])
    # L_min = np.array([[6/(5*np.abs(a))*np.cos(7/12*np.pi)],
    #                     [6/(5*np.abs(a))*np.cos(9/12*np.pi)],
    #                     [6/(5*np.abs(a))*np.cos(11/12*np.pi)]])
    return L_min


def get_L_max(a):
    """
    Input the acceleration and out put the Lmax array
    """
    L_max = np.array([[np.sin(7 / 12 * np.pi) / np.abs(a)],
                      [np.sin(9 / 12 * np.pi) / np.abs(a)],
                      [np.sin(11 / 12 * np.pi) / np.abs(a)]])
    return L_max


def get_Kamm_circle(acc_x_max, acc_x_min, acc_y_max, acc_y_min):
    """
    Calculate the Kmma circle constraints.
    """
    L_x_min, L_x_max, L_y_max = get_L_min(acc_x_min), get_L_min(acc_x_max), get_L_max(acc_y_max)

    L = np.hstack([np.vstack([L_x_min, L_x_min, -L_x_max, -L_x_max]), np.vstack([L_y_max, -L_y_max, L_y_max, -L_y_max])])
    b = np.ones((L.shape[0], 1)) * np.sin(5 / 12 * np.pi)

    return L, b


# def get_Kamm_circle(acc_x_max_sv, acc_x_min_sv, acc_y_max_sv, acc_y_min_sv, acc_x_max_pov, acc_x_min_pov, acc_y_max_pov, acc_y_min_pov):
#     """
#     Calculate the Kmma circle constraints.
#     """
#     L_x_min_sv, L_x_max_sv, L_y_max_sv = get_L_min(acc_x_min_sv), get_L_min(acc_x_max_sv), get_L_max(acc_y_max_sv)
#     L_x_min_pov, L_x_max_pov, L_y_max_pov = get_L_min(acc_x_min_pov), get_L_min(acc_x_max_pov), get_L_max(acc_y_max_pov)

#     L_sv = np.hstack([np.vstack([L_x_min_sv, L_x_min_sv, -L_x_max_sv, -L_x_max_sv]), np.vstack([L_y_max_sv, -L_y_max_sv, L_y_max_sv, -L_y_max_sv])])
#     b_sv = np.ones((L_sv.shape[0],1))*np.sin(5/12*np.pi)

#     L_pov = np.hstack([np.vstack([L_x_min_pov, L_x_min_pov, -L_x_max_pov, -L_x_max_pov]), np.vstack([L_y_max_pov, -L_y_max_pov, L_y_max_pov, -L_y_max_pov])])
#     b_pov = np.ones((L_pov.shape[0],1))*np.sin(5/12*np.pi)

#     return L_sv, b_sv, L_pov, b_pov

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


def get_PQRUVH(A_hat_sv, B_hat_sv, A_hat_pov, B_hat_pov, x_sv, x_pov, C, R_sv, R_pov, O_sv, O_pov):
    """
    R_sv, R_pov is the rotation matrix, O_sv, O_pov is the offset coordinates
    """
    # Offset matrix
    O = O_pov - O_sv

    P = B_hat_pov.T @ C.T @ R_pov.T @ R_pov @ C @ B_hat_pov
    Q = B_hat_sv.T @ C.T @ R_sv.T @ R_sv @ C @ B_hat_sv
    R = - 2 * B_hat_pov.T @ C.T @ R_pov.T @ R_sv @ C @ B_hat_sv
    U = 2 * B_hat_pov.T @ C.T @ R_pov.T @ R_pov @ C @ A_hat_pov @ x_pov - 2 * B_hat_pov.T @ C.T @ R_pov.T @ R_sv @ C @ A_hat_sv @ x_sv + 2 * B_hat_pov.T @ C.T @ R_pov.T @ O
    V = 2 * B_hat_sv.T @ C.T @ R_sv.T @ R_sv @ C @ A_hat_sv @ x_sv - 2 * B_hat_sv.T @ C.T @ R_sv.T @ R_pov @ C @ A_hat_pov @ x_pov - 2 * B_hat_sv.T @ C.T @ R_sv.T @ O
    H = x_pov.T @ A_hat_pov.T @ C.T @ R_pov.T @ R_pov @ C @ A_hat_pov @ x_pov - 2 * x_pov.T @ A_hat_pov.T @ C.T @ R_pov.T @ R_sv @ C @ A_hat_sv @ x_sv + x_sv.T @ A_hat_sv.T @ C.T @ R_sv.T @ R_sv @ C @ A_hat_sv @ x_sv + 2 * O.T @ R_pov @ C @ A_hat_pov @ x_pov - 2 * O.T @ R_sv @ C @ A_hat_sv @ x_sv + O.T @ O

    return P, Q, R, U, V, H
# ====================== Visualization Used Functions ======================
def MPrISM_dynamics(veh_ini_state, control_sequence, delta, T):
    """
    This function is to generate the MPrISM planned trajectory for visualization.

    :param veh_ini_state: the vehicle initial state.
    :param control_sequence: optimized control sequence.
    :param delta: time resolution.
    :param T: look-ahead steps.
    :return: future states in local and global coordinates system.
    """

    assert(control_sequence.size == (T*2))
    x_ini, initial_offset, initial_heading = veh_ini_state
    # Rotation and Offset matrix
    R_0 = np.array([[np.cos(initial_heading), -np.sin(initial_heading)], [np.sin(initial_heading), np.cos(initial_heading)]])
    R = np.kron(np.array([[1, 0], [0, 0]]), R_0) + np.kron(np.array([[0, 0], [0, 1]]), np.eye(2))  # TODO: Hard code here assume the state is 4-dimensional.
    O = np.array([[initial_offset[0]], [initial_offset[1]], [0], [0]])
    H = np.array([[0], [0], [0], [initial_heading]])

    # Use the initial velocity as the constant velocity
    v_til = x_ini[2, 0]
    if np.abs(v_til) < 5:
        if v_til >= 0:
            v_til = 5
        else:
            v_til = -5

    # State transition dynamics
    # x' = A^hat @ x + B_hat @ u, x is the initial state, u is the actions along the interval
    A, B = get_A(delta, v_til), get_B(delta, v_til)

    local_coordinates_state_list = [x_ini]
    global_state = (R @ x_ini + O + H)
    global_coordinate_state_list = [global_state]
    for i in range(T):
        # Local coordinates: [local x, local y, v, local_heading]
        new_local_state = A @ local_coordinates_state_list[-1] + B @ (control_sequence[i * 2:(i + 1) * 2, 0].reshape(-1, 1))
        local_coordinates_state_list.append(new_local_state)

        # new_heading = initial_heading + new_local_state[3, 0].item()  # new heading
        # new_R = np.array([[np.cos(new_heading), -np.sin(new_heading)], [np.sin(new_heading), np.cos(new_heading)]])
        # global_state_array = new_R @ (C @ new_local_state) + O
        # global_state = [global_state_array[0].item(), global_state_array[1].item(), new_heading]
        global_state = (R @ new_local_state + O + H)
        global_coordinate_state_list.append(global_state)
    local_coordinates_state_list, global_coordinate_state_list = [(val.T.tolist())[0] for val in local_coordinates_state_list], [(val.T.tolist())[0] for val in global_coordinate_state_list]
    local_state_array, global_state_array = np.array(local_coordinates_state_list).T, np.array(global_coordinate_state_list).T
    return local_state_array, global_state_array
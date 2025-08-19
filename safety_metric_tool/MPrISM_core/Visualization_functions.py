# This file develops the visualization tools for the safety metric evaluation project including figures, videos, etc.
# Author: Xintao Yan
# Date: 2/19/2021
# Affiliation: Michigan Traffic Lab (MTL)


import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
mpl.rcParams['font.size'] = 30
plt.rcParams["font.family"] = "Times New Roman"
plt.switch_backend('agg')
mpl.use('Agg')

plt.rcParams['animation.ffmpeg_path'] = 'C:/Program Files/ffmpeg-6.0-full_build/bin/ffmpeg.exe'
plt.rcParams["font.family"] = "Times New Roman"
import copy
import pandas as pd
import os
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

# Map info
# TODO: Generalize to more maps
lane_width, num_lanes = 4, 3
num_lane_strips = num_lanes + 1
highway_length, Exit_length = 1600, 1400
lane_discretize_resolution = 1000
VEH_LENGTH, VEH_WIDTH = 5, 2
VEH_APPROX_CIR_RADIUS = 2
# xlim = 600, 720
ylim = (-6, 14)

crash_risk_level = 99

# ====== Visualization tools =========
def rotate(x1, y1, x2, y2, angle=0):
    x = (x1 - x2)*np.cos(angle) - (y1 - y2)*np.sin(angle) + x2
    y = (x1 - x2)*np.sin(angle) + (y1 - y2)*np.cos(angle) + y2
    return [x,y]


def cal_box(x, y, length=5, width=2, angle=0):
    upper_left = rotate(x-0.5*length, y+0.5*width, x, y, angle=angle)
    lower_left = rotate(x-0.5*length, y-0.5*width, x, y, angle=angle)
    upper_right = rotate(x+0.5*length, y+0.5*width, x, y, angle=angle)
    lower_right = rotate(x+0.5*length, y-0.5*width, x, y, angle=angle)
    xs = [upper_left[0], upper_right[0], lower_right[0], lower_left[0], upper_left[0]]
    ys = [upper_left[1], upper_right[1], lower_right[1], lower_left[1], upper_left[1]]
    return xs, ys


def _rotate_scene_for_replay(one_episode, angle):
    """
    Rotate all trajectories for better visualization
    :param one_episode:
    :param angle: anti-clockwise in radians
    :return:
    """
    # Rotate traj for better visualization
    one_episode.loc[:, 'heading'] = one_episode.loc[:, 'heading'] + angle
    for row in one_episode.itertuples():
        rotate_x, rotate_y = rotate(row.x, row.y, 0., 0., angle=angle)
        one_episode.at[row.Index, 'x'] = rotate_x
        one_episode.at[row.Index, 'y'] = rotate_y

    return one_episode


def replay_one_simulation(one_episode, sim_id=None, metric=None, slow_version=True, save_video_flag=False, file_name=None, whole_traj_flag=False, color_dict=None,
                          evasive_traj_flag=False, MPrISM_planned_traj_flag=False, MPrISM_given_POV_id=None, plot_center_circles=False, plot_heading_lines=True, rotate_scene_angle=None):
    """This function generate video from the input vehicle dataframe.
    """
    dynamic_view = False
    cav_obs_range = 50

    if rotate_scene_angle is not None:
        one_episode = _rotate_scene_for_replay(one_episode, angle=rotate_scene_angle)

    fig = plt.figure(figsize=(16, 16))
    SV_initial_pos_x = one_episode.loc[one_episode["veh_id"] == "CAV", "x"].tolist()[0]
    SV_initial_pos_y = one_episode.loc[one_episode["veh_id"] == "CAV", "y"].tolist()[0]
    xlim = (SV_initial_pos_x - 100, SV_initial_pos_x + 100)
    ylim = (SV_initial_pos_y - 100, SV_initial_pos_y + 100)
    ax = plt.axes(xlim=xlim, ylim=ylim)
    ax.set_aspect('equal')
    lanes, veh_boxes, fills, center_circles, heading_lines = [], [], [], [], []

    # Create stright lanes
    for lane_strip_idx in range(num_lane_strips):
        lane_y = lane_strip_idx * lane_width - 0.5 * lane_width
        x_print = list(np.linspace(0, highway_length, lane_discretize_resolution))
        y_print = [lane_y] * lane_discretize_resolution

        lane = ax.plot([], [], 'C1')[0]
        lane.set_data(x_print, y_print)
        lanes.append(lane)

    # Create ramp position
    x_ramp_indicator = [Exit_length] * lane_discretize_resolution
    y_ramp_indicator = list(np.linspace(- 0.5 * lane_width, (num_lane_strips - 1) * lane_width - 0.5 * lane_width,
                                        lane_discretize_resolution))
    lane = ax.plot([], [], 'r')[0]
    lane.set_data(x_ramp_indicator, y_ramp_indicator)
    lanes.append(lane)

    avail_time = sorted(list(one_episode.time.unique()))
    start_time = avail_time[0]
    # frames = (data.Global_Time.max()-data.Global_Time.min()) / 100 * portion
    frames = len(avail_time)

    def animate(t):
        # remove the previous objects
        # ax.collections.clear()
        # ax.clear() # remove all
        ax.lines[:], veh_boxes[:], fills[:], center_circles[:], heading_lines[:] = lanes[:], [], [], [], []
        # Remove only rectangles and retain circles
        patches_list = []
        for patch_item in ax.patches:
            if isinstance(patch_item, plt.Polygon):
                continue
            else:
                patches_list.append(patch_item)
        ax.patches[:] = patches_list

        fill_flag = False
        t = t + start_time
        if t in avail_time:
            vehs_data_specific_moment = one_episode.groupby('time').get_group(t).reset_index(drop=True)
            cav_x = vehs_data_specific_moment.loc[vehs_data_specific_moment["veh_id"] == "CAV", "x"].item()
            cav_y = vehs_data_specific_moment.loc[vehs_data_specific_moment["veh_id"] == "CAV", "y"].item()
            xlim = (cav_x - cav_obs_range - 10, cav_x + cav_obs_range + 10)
            ylim = (cav_y - cav_obs_range - 10, cav_y + cav_obs_range + 10)
            if dynamic_view:
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
            for veh_id, x, y, heading in zip(list(vehs_data_specific_moment.veh_id), list(vehs_data_specific_moment.x),
                                             list(vehs_data_specific_moment.y),
                                             list(vehs_data_specific_moment.heading)):
                xs, ys = cal_box(x, y, length=VEH_LENGTH, angle=heading)
                if veh_id == "CAV":
                    plot_color = "red"  # color_dict[dangerous_level]
                    if metric is not None:
                        fill_flag = vehs_data_specific_moment[vehs_data_specific_moment["veh_id"] == "CAV"][
                            metric].item()
                        # fill_flag = True
                        # dangerous_level = vehs_data_specific_moment.loc[vehs_data_specific_moment["veh_id"] == "CAV", "mapped_MPrTTC"].item()
                    if whole_traj_flag or evasive_traj_flag:
                        fill_flag = False
                        dangerous_level = vehs_data_specific_moment.loc[vehs_data_specific_moment["veh_id"] == "CAV", "dangerous_level"].item()
                        SV_color = color_dict[dangerous_level]
                    if fill_flag is True:
                        fills.append(ax.add_patch(plt.Polygon(np.array([xs, ys]).T, color='red')))
                else:
                    plot_color = 'blue'
                    if MPrISM_planned_traj_flag and veh_id == MPrISM_given_POV_id:
                        fills.append(ax.add_patch(plt.Polygon(np.array([xs, ys]).T, color='blue')))
                # Plot vehicle box
                veh_box = ax.plot([], [], color=plot_color)[0]
                veh_box.set_data(xs, ys)
                veh_boxes.append(veh_box)
                # Plot center circles
                if plot_center_circles:
                    center_circles.append(ax.add_patch(plt.Circle((x, y), radius=VEH_APPROX_CIR_RADIUS, fc='none', ec=plot_color)))
                # Plot heading direction lines
                if plot_heading_lines:
                    line_x, line_y = [x - dist*np.cos(heading) for dist in np.linspace(0, 3, 10)], [y - dist*np.sin(heading) for dist in np.linspace(0, 3, 10)]
                    heading_line = ax.plot(line_x, line_y, color='k')[0]
                    heading_lines.append(heading_line)

            return tuple(veh_boxes) + tuple(lanes) + tuple(fills) + tuple(center_circles) + tuple(heading_lines)
        else:
            return tuple(lanes)

    line_ani = animation.FuncAnimation(fig, animate, frames=int(frames), interval=100, blit=True)
    Writer = animation.writers['ffmpeg']
    if slow_version:
        # fps = 8
        fps = 5
        speed = "Slow"
    else:
        # fps, speed = 15, "Normal"
        fps, speed = 10, "Normal"
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)

    if save_video_flag:
        assert (file_name is not None)
        line_ani.save(file_name + ".mp4", writer=writer)
        plt.close('all')
    plt.close('all')


def generate_evasive_video_df(trancate_traj, evasive_planned_traj, dangerous_level):
    """Swap the AV state information in the trancate traj using the evasive traj.
    
    :param trancate_traj: the trajectory df.
    :param evasive_traj: np.array (4*t+1) for the AV evasive traj
    :return: new df where the SV traj is swapped using the planned evasive traj.
    """
    assert (trancate_traj.time.unique().size == evasive_planned_traj.shape[1])
    new_df = copy.deepcopy(trancate_traj)
    time_list = list(trancate_traj.time.unique())
    for time in time_list:
        time_idx = time_list.index(time)
        one_moment_df = trancate_traj[trancate_traj["time"] == time]
        if "CAV" in one_moment_df.veh_id.values:
            new_df.loc[(new_df["time"] == time) & (new_df["veh_id"] == "CAV"), ["x", "y", "v", "heading"]] = evasive_planned_traj[:, time_idx]
            new_df.loc[(new_df["time"] == time) & (new_df["veh_id"] == "CAV"), ["dangerous_level"]] = dangerous_level
        else:
            one_row_list, columns_name = [time, "CAV"] + evasive_planned_traj[:, time_idx].tolist() + [dangerous_level], ["time", "veh_id", "x", "y", "v",
                                                                                                      "heading", "dangerous_level"]
            one_row = pd.DataFrame([one_row_list], columns=columns_name)
            new_df = new_df.append(one_row, ignore_index=True)
    new_df.sort_values(by=['time'], inplace=True)
    return new_df


def generate_discrete_color_dict(safe_state_idx=0, unavoidable_state_idx=4, default_dangerous_level=-1, default_dangerous_level_color="grey", crash_dangerous_level=99, crash_dangerous_level_color="red"):
    """
    Generate the mapping dict from state idx to color. state 0: safe->green, state unavoidable->red.

    :param safe_state_idx:
    :param unavoidable_state_idx:
    :return:
    """
    color_dict = {}
    dangerous_color = ["yellow", "darkorange", "darkviolet"]
    color_dict[safe_state_idx] = "green"
    color_dict[unavoidable_state_idx] = "red"
    color_dict[default_dangerous_level], color_dict[crash_dangerous_level] = default_dangerous_level_color, crash_dangerous_level_color
    dangerous_state_idx_list = range(safe_state_idx+1, unavoidable_state_idx, 1)
    for dangerous_state_idx in dangerous_state_idx_list:
        color_dict[dangerous_state_idx] = dangerous_color[dangerous_state_idx_list.index(dangerous_state_idx)]

    return color_dict


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


def plot_static_illustration_figure(one_episode, plot_box_freq, fill_flag=True, BV_fill_flag=False,
                                    alpha=0.2, crash_BV_color="blue", fig_size_factor=3, SV_center_point_plot_flag=True,
                                    save_fig_flag=False, file_name=None, given_interval=None, given_POV_id=None,
                                    plot_evasive_traj_flag=False, evasive_planned_traj=None, evasive_traj_color="blue",
                                    evasive_fill_flag=True, plot_evasive_box=True, plot_evasive_traj_line_flag=True,
                                    plot_evasive_traj_three_circles_flag=False, evasive_traj_three_circles=None,
                                    extend_traj_color=None,
                                    plot_three_circles_flag=False, plot_three_circles_original_CAV_flag=False,
                                    radius=None, center_point_distance=None):
    """
    one_episode: the trajectory episode (trancate trajectory)
    xlim: the xlim for the figure
    given_interval: the given interval that want to plot. if given interval is given, use this one
    given_POV_id: if use given interval, need to also give POV id.
    plot_box_freq: the frequency of plotting the boxes. 1: plot all time instances, 2: plot every
                    two time instance
    fill_flag: whether fill the box
    alpha: the transparency of the fill   
    evasive_planned_traj: the trajectory planned by the evasive planning algorithm
    evasive_traj_three_circles: [rear_circle(np.array 4*t), center_circle(np.array 4*t), front_circle(np.array 4*t)]
    """
    SV_initial_pos = one_episode.loc[one_episode["veh_id"] == "CAV", "x"].tolist()[0]
    xlim = (SV_initial_pos-30, SV_initial_pos+90)
    fig = plt.figure(figsize=((xlim[1] - xlim[0]) * fig_size_factor / 20, fig_size_factor))
    plt.xlim(xlim)
    plt.ylim(ylim)

    # Create stright lanes
    for lane_strip_idx in range(num_lane_strips):
        lane_y = lane_strip_idx * lane_width - 0.5 * lane_width
        x_print = list(np.linspace(0, highway_length, lane_discretize_resolution))
        y_print = [lane_y] * lane_discretize_resolution
        plt.plot(x_print, y_print, c="C1")

    # Find crash veh id (the BV closest to the SV)
    try:
        crash_BV_id = one_episode[(one_episode["mode"] == "Crash") & (one_episode["veh_id"] != "CAV")]["veh_id"].item()
    except:
        crash_BV_id = None
    avail_time = list(one_episode.time.unique()) if given_interval is None else given_interval

    first_moment_flag = True
    if given_interval:
        t_interval = given_interval[::plot_box_freq]
        if given_POV_id is not None:
            crash_BV_id = given_POV_id
        else:
            raise ValueError("No given POV id")
    else:
        t_interval = avail_time[::plot_box_freq]
    plot_all_box_veh_id_list = [crash_BV_id]  # "CAV", crash_BV_id ,
    # "2e0844b0-ac7f-466a-be1b-80bfb0e3de84"
    # not_plot_traj_line_veh_id_list = ["CAV"]
    plot_traj_line_veh_id_list = []
    for t in t_interval:
        vehs_data_specific_moment = one_episode.groupby('time').get_group(t).reset_index(drop=True)
        extend_traj_at_current_moment_flag = False  # Whether current moment is generated by extend trajectories
        if 'is_extend_traj' in vehs_data_specific_moment.columns:
            extend_traj_at_current_moment_flag = vehs_data_specific_moment.is_extend_traj.any()
        for veh_id, x, y, heading in zip(list(vehs_data_specific_moment.veh_id), list(vehs_data_specific_moment.x),
                                         list(vehs_data_specific_moment.y), list(vehs_data_specific_moment.heading)):
            # if veh_id not in plot_all_box_veh_id_list and not first_moment_flag:
            #     continue
            # if veh_id == "CAV" and not first_moment_flag:
            #     continue
            xs, ys = cal_box(x, y, length=VEH_LENGTH, width=VEH_WIDTH, angle=heading)

            # Plot vehicle boxes
            if veh_id == "CAV":
                plt.plot(xs, ys, c="r")
                if fill_flag:
                    ax = plt.gca()
                    ax.add_patch(patches.Rectangle((xs[3], ys[3]),
                                                   width=VEH_LENGTH, height=VEH_WIDTH, angle=math.degrees(heading),
                                                   fill=True, alpha=alpha, color='red'))

                if plot_three_circles_original_CAV_flag:
                    x_front, y_front = util_cal_circle_position(x, y, heading,
                                                                 center_point_distance=center_point_distance,
                                                                 circle_pos="front")
                    x_rear, y_rear = util_cal_circle_position(x, y, heading,
                                                               center_point_distance=center_point_distance,
                                                               circle_pos="rear")
                    rear, center, front = [x_rear, y_rear], [x, y], [x_front, y_front]
                    ax = plt.gca()
                    ax.add_patch(patches.Circle(rear, radius=radius, fc=None, ec="red")), ax.add_patch(
                        patches.Circle(center, radius=radius, fc=None, ec="red")), ax.add_patch(
                        patches.Circle(front, radius=radius, fc=None, ec="red"))
            elif veh_id == crash_BV_id:
                c = crash_BV_color if (not extend_traj_at_current_moment_flag or extend_traj_color is None) else extend_traj_color
                plt.plot(xs, ys, c=c)
                # print(veh_id)
                if BV_fill_flag:
                    ax = plt.gca()
                    ax.add_patch(patches.Rectangle((xs[3], ys[3]),
                                                   width=VEH_LENGTH, height=VEH_WIDTH, angle=math.degrees(heading),
                                                   fill=True, alpha=alpha, color=c))
            else:
                c = "k" if (not extend_traj_at_current_moment_flag or extend_traj_color is None) else extend_traj_color
                c = "k" if (
                            veh_id == "2e0844b0-ac7f-466a-be1b-80bfb0e3de84" and not
                extend_traj_at_current_moment_flag) else c
                plt.plot(xs, ys, c=c)
                if BV_fill_flag:
                    ax = plt.gca()
                    ax.add_patch(patches.Rectangle((xs[3], ys[3]),
                                                   width=VEH_LENGTH, height=VEH_WIDTH, angle=math.degrees(heading),
                                                   fill=True, alpha=alpha, color=c))

            # Plot three circles
            if plot_three_circles_flag and veh_id != "CAV":
                x_front, y_front = util_cal_circle_position(x, y, heading, center_point_distance=center_point_distance,
                                                             circle_pos="front")
                x_rear, y_rear = util_cal_circle_position(x, y, heading, center_point_distance=center_point_distance,
                                                           circle_pos="rear")
                rear, center, front = [x_rear, y_rear], [x, y], [x_front, y_front]
                ax = plt.gca()
                ax.add_patch(patches.Circle(rear, radius=radius, fc=c, ec=c, alpha=alpha)), ax.add_patch(
                    patches.Circle(center, radius=radius, fc=c, ec=c, alpha=alpha)), ax.add_patch(
                    patches.Circle(front, radius=radius, fc=c, ec=c, alpha=alpha))

            if first_moment_flag:
                one_veh_data = one_episode[(one_episode["time"] >= t_interval[0]) &
                                           (one_episode["time"] <= t_interval[-1]) & (one_episode["veh_id"] == veh_id)]
                color = "r" if veh_id == "CAV" else crash_BV_color if veh_id == crash_BV_id else "k"
                label = "SV" if veh_id == "CAV" else "POV" if veh_id == crash_BV_id else "BV"
                if veh_id != "CAV" and extend_traj_at_current_moment_flag: color = extend_traj_color

                # Plot the center point of the SV if needed
                if SV_center_point_plot_flag and veh_id == "CAV":
                    SV_x, SV_y = one_veh_data.iloc[0]["x"], one_veh_data.iloc[0]["y"]
                    plt.scatter(SV_x, SV_y, c=color)

                    # Plot trajectory line
                # if veh_id in not_plot_traj_line_veh_id_list: continue
                if veh_id not in plot_traj_line_veh_id_list: continue
                plt.plot(one_veh_data["x"], one_veh_data["y"], c=color)

        if first_moment_flag:
            first_moment_flag = False

    # Plot the evasive planned trajectory
    if plot_evasive_traj_flag:
        assert (evasive_planned_traj is not None)
        # Plot boxes
        if plot_evasive_box:
            for t in t_interval:
                t_idx = avail_time.index(t)
                x, y, heading = evasive_planned_traj[0, t_idx], evasive_planned_traj[1, t_idx], evasive_planned_traj[
                    3, t_idx]
                xs, ys = cal_box(x, y, length=VEH_LENGTH, width=VEH_WIDTH, angle=heading)

                # Plot vehicle boxes
                plt.plot(xs, ys, c=evasive_traj_color)
                if evasive_fill_flag:
                    ax = plt.gca()
                    ax.add_patch(patches.Rectangle((xs[3], ys[3]),
                                                   width=VEH_LENGTH, height=VEH_WIDTH, angle=math.degrees(heading),
                                                   fill=True, alpha=0.01, color="blue"))
        if plot_evasive_traj_line_flag:
            # Plot trajectory line
            plt.plot(evasive_planned_traj[0, :].tolist(), evasive_planned_traj[1, :].tolist(), c=evasive_traj_color)
    # Plot the 3 circles of the evasive planned traj
    if plot_evasive_traj_three_circles_flag:
        assert (evasive_traj_three_circles is not None)
        for t in t_interval:
            t_idx = avail_time.index(t)
            x_rear, y_rear, x_center, y_center, x_front, y_front = evasive_traj_three_circles[0][0, t_idx], \
                                                                   evasive_traj_three_circles[0][1, t_idx], \
                                                                   evasive_traj_three_circles[1][0, t_idx], \
                                                                   evasive_traj_three_circles[1][1, t_idx], \
                                                                   evasive_traj_three_circles[2][0, t_idx], \
                                                                   evasive_traj_three_circles[2][1, t_idx]
            rear, center, front = [x_rear, y_rear], [x_center, y_center], [x_front, y_front]
            ax = plt.gca()
            ax.add_patch(patches.Circle(rear, radius=radius, fc=evasive_traj_color, ec=evasive_traj_color,
                                        alpha=alpha)), ax.add_patch(
                patches.Circle(center, radius=radius, fc=evasive_traj_color, ec=evasive_traj_color,
                               alpha=alpha)), ax.add_patch(
                patches.Circle(front, radius=radius, fc=evasive_traj_color, ec=evasive_traj_color, alpha=alpha))

    # plt.xticks([])
    # plt.yticks([])
    # plt.legend()
    if save_fig_flag:
        assert (file_name is not None)
        plt.savefig(file_name+".svg", bbox_inches="tight")
        plt.savefig(file_name+".pdf", bbox_inches="tight")
        plt.savefig(file_name + ".png", dpi=300, bbox_inches="tight")

    plt.close('all')

def plot_time_region_with_metric_figure(one_episode, alpha=0.1, evaluated_dangerous_level_color_dict=None, save_fig_flag=False, file_name=None, metric_list=None, ylim=(0,2),
                                        plot_with_mapped_metric_flag=False, MPrISM_critical_points=None):
    """
    This function plots the dangerous level with metrics

    :param one_episode:
    :param metric_list:
    :return:
    """
    plt.figure(figsize=(16, 9))
    time_list, evaluated_dangerous_level_list = one_episode["time"].tolist(), one_episode["dangerous_level"].tolist()
    evaluated_dangerous_level_change_pos = [time for time in range(1, len(time_list)) if evaluated_dangerous_level_list[time] != evaluated_dangerous_level_list[time-1]]
    evaluated_dangerous_level_color = [evaluated_dangerous_level_color_dict[evaluated_dangerous_level_list[time]] for time in evaluated_dangerous_level_change_pos]
    plt.axvspan(time_list[0], evaluated_dangerous_level_change_pos[0], facecolor=evaluated_dangerous_level_color_dict[time_list[0]], alpha=alpha)
    for i in range(len(evaluated_dangerous_level_change_pos)):
        start_time, color = evaluated_dangerous_level_change_pos[i], evaluated_dangerous_level_color[i]
        try: end_time = evaluated_dangerous_level_change_pos[i+1]
        except: end_time = start_time+1
        plt.axvspan(start_time, end_time, facecolor=color, alpha=alpha)

    # Plot vertical line on the last moment
    plt.axvline(time_list[-1], color="k", linestyle="--")

    for safety_metric in metric_list:
        if safety_metric.split("_")[0] == "MPrISM":
            MPrISM_value_list = one_episode["MPrTTC"].tolist()
            plt.plot(time_list, MPrISM_value_list, '--s', linewidth=2, markersize=8, label="MPrTTC")
            if plot_with_mapped_metric_flag:
                mapped_MPrISM_value_list = one_episode["mapped_MPrTTC"].tolist()
                color_for_each_point = [evaluated_dangerous_level_color_dict[val] for val in mapped_MPrISM_value_list]
                plt.scatter(time_list, MPrISM_value_list, marker='s', s=100, color=color_for_each_point, zorder=20)
                if MPrISM_critical_points is not None:
                    for critical_point in MPrISM_critical_points: plt.axhline(critical_point, color="k", linestyle="-.")

        if safety_metric == "TTC":
            TTC_value_list = one_episode["TTC"].tolist()
            plt.plot(time_list, TTC_value_list, '--v', linewidth=2, markersize=8, label="TTC")

    plt.ylim(ylim)
    plt.xlabel("Time")
    plt.ylabel("TTC (s)")
    plt.legend(fontsize=30, loc="lower left")

    if save_fig_flag:
        assert (file_name is not None)
        plt.savefig(file_name+".svg", bbox_inches="tight")
        plt.savefig(file_name+".pdf", bbox_inches="tight")
        plt.savefig(file_name + ".png", dpi=300, bbox_inches="tight")

    plt.close('all')

def plot_time_dangerous_level_figure(evaluated_dangerous_level_df, alpha=0.1, evaluated_dangerous_level_color_dict=None, save_fig_flag=False,
                                     file_name=None):
    time_list, evaluated_dangerous_level_list = evaluated_dangerous_level_df["time"].tolist(), evaluated_dangerous_level_df["dangerous_level"].tolist()
    evaluated_dangerous_level_change_pos = [time for time in range(1, len(time_list)) if time_list[time] != time_list[time-1]]
    evaluated_dangerous_level_color = [evaluated_dangerous_level_color_dict[evaluated_dangerous_level_list[time-1]] for time in evaluated_dangerous_level_change_pos]
    plt.axvspan(time_list[0], evaluated_dangerous_level_change_pos[0], facecolor=evaluated_dangerous_level_color[0], alpha=alpha)
    for i in range(len(evaluated_dangerous_level_change_pos)):
        start_time, color = evaluated_dangerous_level_change_pos[i], evaluated_dangerous_level_color[i]
        try: end_time = evaluated_dangerous_level_change_pos[i+1]
        except: end_time = start_time+1
        plt.axvspan(start_time, end_time, facecolor=color, alpha=alpha)

    # Plot vertical line on the last moment
    plt.axvline(time_list[-1], color="k", linestyle="--")

    plt.ylim(0, 1)
    plt.xlabel("Time Idx")
    plt.ylabel("TTC (s)")
    plt.legend(fontsize=30)

    if save_fig_flag:
        assert (file_name is not None)
        plt.savefig(file_name+".svg", bbox_inches="tight")
        plt.savefig(file_name+".pdf", bbox_inches="tight")
        plt.savefig(file_name + ".png", dpi=300, bbox_inches="tight")

    plt.close('all')


def plot_time_axis_with_colored_dangerous_level(one_episode, dangerous_level_col_name, time_steps=1, line_width=8, evaluated_dangerous_level_color_dict=None, save_fig_flag=False,
                                                file_name=None, figsize=(16,9), alpha=0.8, activate_binary_color_flag=False):
    """
    This function plots the time axis figure with color which indicates the dangerous level (either the ground-truth or the mapped safety metric)

    :param one_episode:
    :param dangerous_level_col_name:
    :param evaluated_dangerous_level_color_dict:
    :param save_fig_flag:
    :param file_name:
    :param activate_binary_color_flag: if True, then the color will be red if activate (for metric)/ unavoidable (for post-trip).
    :return:
    """
    mpl.rcParams['font.size'] = 60
    plt.rcParams["font.family"] = "Times New Roman"

    fig1 = plt.figure(facecolor='white', figsize=figsize)
    ax1 = plt.axes(frameon=False)

    ax1.set_frame_on(False)
    ax1.get_xaxis().tick_bottom()

    time_list, evaluated_dangerous_level_list = one_episode["time"].tolist(), one_episode[dangerous_level_col_name].tolist()
    evaluated_dangerous_level_change_pos = [time for time in range(1, len(time_list)) if evaluated_dangerous_level_list[time] != evaluated_dangerous_level_list[time - 1]]
    if activate_binary_color_flag:  # Fixme: quick fix to plot binary plot using 0/1/2/3/unavoidable data.
        # For post-trip, show only unavoidable
        if dangerous_level_col_name == "dangerous_level" or dangerous_level_col_name == "mapped_PCM":
            evaluated_dangerous_level_change_pos = [time for time in range(1, len(time_list)) if (evaluated_dangerous_level_list[time] != evaluated_dangerous_level_list[time - 1] and (evaluated_dangerous_level_list[time] == 4 or evaluated_dangerous_level_list[time-1] == 4))]
        else:
            evaluated_dangerous_level_change_pos = [time for time in range(1, len(time_list)) if evaluated_dangerous_level_list[time] != evaluated_dangerous_level_list[time - 1] and (evaluated_dangerous_level_list[time] == 0 or evaluated_dangerous_level_list[time-1] == 0)]
    # print(dangerous_level_col_name, evaluated_dangerous_level_change_pos, evaluated_dangerous_level_list)
    evaluated_dangerous_level_color = [evaluated_dangerous_level_color_dict[evaluated_dangerous_level_list[time]] for time in evaluated_dangerous_level_change_pos]
    ini_plot_right_pos = evaluated_dangerous_level_change_pos[0]*time_steps if len(evaluated_dangerous_level_change_pos) != 0 else time_list[-1]
    plt.axvspan(time_list[0]*time_steps, ini_plot_right_pos, facecolor=evaluated_dangerous_level_color_dict[time_list[0]], alpha=alpha)
    for i in range(len(evaluated_dangerous_level_change_pos)):
        start_time_idx, color = evaluated_dangerous_level_change_pos[i], evaluated_dangerous_level_color[i]
        if activate_binary_color_flag:  # Fixme: quick fix to plot binary plot using 0/1/2/3/unavoidable data.
            # For post-trip, show only unavoidable
            if dangerous_level_col_name == "dangerous_level":
                color = "red" if color == "red" else "green"
            # For metric, show only activate
            else:
                color = "red" if color != "green" else "green"
        try:
            end_time_idx = evaluated_dangerous_level_change_pos[i + 1]
        except:
            end_time_idx = time_list[-1]
        plt.axvspan(start_time_idx*time_steps, end_time_idx*time_steps, facecolor=color, alpha=alpha)

    last_moment_crash_flag = True if evaluated_dangerous_level_list[-1] == crash_risk_level else False  # If the last moment have a collision, then don't show that moment.
    plt.xlim(time_list[0] * time_steps, time_list[-1] * time_steps)  # Fixme: Omit the last moment.
    # if last_moment_crash_flag: plt.xlim(time_list[0]*time_steps, time_list[-1]*time_steps)
    # else: plt.xlim(time_list[0]*time_steps, (time_list[-1] + 1)*time_steps)
    plt.ylim(0, 1)
    ax1.axes.get_yaxis().set_visible(False)
    plt.xlabel("Time")

    if save_fig_flag:
        assert (file_name is not None)
        plt.savefig(file_name+".svg", bbox_inches="tight")
        plt.savefig(file_name+".pdf", bbox_inches="tight")
        plt.savefig(file_name + ".png", dpi=300, bbox_inches="tight")

    plt.close('all')


# ====== MPrISM Visualization tools =========
def plot_time_metric_figure(data_df=None, metric_col_name_list=["MPrTTC"], save_fig_flag=False, file_name=None, ymax=2, t_reference_idx=0, SIM_FREQ=15,
                            plot_last_moment_flag=False):
    mpl.rcParams['font.size'] = 30
    plt.rcParams["font.family"] = "Times New Roman"
    if data_df is not None:
        for metric_col_name in metric_col_name_list:
            time_list, metric_value_list = data_df["time"].tolist(), data_df[metric_col_name].tolist()
            time_list, metric_value_list = time_list[-t_reference_idx:], metric_value_list[-t_reference_idx:]
            time_list = [val / SIM_FREQ for val in time_list]
            if metric_col_name == "PCM_activate" or metric_col_name == "RSS_activate":
                metric_value_list = [ymax if val is True else 1e5 for val in metric_value_list]
            # last_moment_crash_flag = True if metric_value_list[-1] == 0 else False
            # if last_moment_crash_flag: plot_time_list, plot_metric_value_list = time_list[:-1], metric_value_list[:-1]
            # else: plot_time_list, plot_metric_value_list = time_list, metric_value_list
            if not plot_last_moment_flag: plot_time_list, plot_metric_value_list = time_list[:-1], metric_value_list[:-1]
            else: plot_time_list, plot_metric_value_list = time_list, metric_value_list
            if metric_col_name == "MPrTTC":
                marker_line_type, linewidth, markersize, color, label = '--s', 1.5, 5, "C0", "MPrTTC"
            if metric_col_name == "TTC":
                marker_line_type, linewidth, markersize, color, label = '--v', 1.5, 5, "C1", "TTC"
            if metric_col_name == "RSS_activate":
                marker_line_type, linewidth, markersize, color, label = '--.', 1.5, 10, "C2", "RSS"
            if metric_col_name == "PCM_activate":
                marker_line_type, linewidth, markersize, color, label = '--.', 1.5, 10, "C2", "PCM"
            plt.plot(plot_time_list, plot_metric_value_list, marker_line_type, linewidth=linewidth, markersize=markersize, label=label, color=color)

    # Plot vertical line on the last moment
    plt.axvline(time_list[-1], color="k", linestyle="--")
    plt.ylim(0, ymax)
    plt.xlabel("Time (s)")
    plt.ylabel("TTC (s)")
    # plt.legend(loc="lower left", fontsize=20)
    # plt.legend(loc="lower right", fontsize=20)
    plt.legend(loc="best", fontsize=20)
    if save_fig_flag:
        assert (file_name is not None)
        plt.savefig(file_name+".svg", bbox_inches="tight")
        plt.savefig(file_name+".pdf", bbox_inches="tight")
        plt.savefig(file_name + ".png", dpi=300, bbox_inches="tight")

    plt.close('all')


def generate_MPrISM_planned_traj_video_df(truncate_traj, SV_traj, POV_traj, POV_id, start_time, end_time, T):
    """Swap the SV and POV state information in the truncate traj using the MPrISM planned traj.

    :param truncate_traj: the trajectory df.
    :param SV_traj: np.array (4*t+1) for the AV planned traj
    :param POV_traj: np.array (4*t+1) for the POV evasive traj
    :param POV_id: POV id
    :return: new df where the SV and POV traj is swapped using the planned evasive traj.
    """
    new_df = copy.deepcopy(truncate_traj)

    desired_time_list = range(start_time, end_time+1, 1)
    sim_exist_time_list = list(truncate_traj.time.unique())

    for time in desired_time_list:
        time_idx = desired_time_list.index(time)
        if time in sim_exist_time_list:
            new_df.loc[(new_df["time"] == time) & (new_df["veh_id"] == "CAV"), ["x", "y", "v", "heading"]] = SV_traj[:, time_idx]
            new_df.loc[(new_df["time"] == time) & (new_df["veh_id"] == POV_id), ["x", "y", "v", "heading"]] = POV_traj[:, time_idx]
        else:  # Situation that the simulation end earlier than the estimated collision.
            SV_one_row_list, columns_name = [time, "CAV"] + SV_traj[:, time_idx].tolist(), ["time", "veh_id", "x", "y", "v", "heading"]
            POV_one_row_list, columns_name = [time, POV_id] + POV_traj[:, time_idx].tolist(), ["time", "veh_id", "x", "y", "v", "heading"]

            SV_one_row, POV_one_row = pd.DataFrame([SV_one_row_list], columns=columns_name), pd.DataFrame([POV_one_row_list], columns=columns_name)
            new_df = new_df.append(SV_one_row, ignore_index=True)
            new_df = new_df.append(POV_one_row, ignore_index=True)
    new_df.sort_values(by=['time'], inplace=True)
    return new_df
# ======================================================


# ====== Data analysis tools ==========
def _get_dangerous_level_pdf(all_episodes, dangerous_level_list, col_name=None, crash_dangerous_level=99, unavoidable_dangerous_level=4):
    """
    This function is used to get the pdf of dangerous level (either the ground-truth dangerous level or the mapped safety metric dangerous level e.g., mapped_MPrTTC).

    :param all_episodes: df including all metrics' mapped and ground truth dangerous level
    :param dangerous_level_list: the list including all dangerous level. E.g., [0, 1, 2, 3, 4].
    :param col_name: the name of the dangerous level want to fetch. E.g., "dangerous_level" for the ground truth and "mapped_MPrTTC" for the MPrISM...
    :param crash_dangerous_level: transform the crash dangerous level to the unavoidable dangerous level when calculating the pdf.
    :return: the dangerous level pdf.
    """

    # Get dangerous level
    eval_dangerous_level_list = all_episodes[col_name].tolist()
    # Transfer the crash dangerous level to the unavoidable dangerous level
    eval_dangerous_level_list = [val if val != crash_dangerous_level else unavoidable_dangerous_level for val in eval_dangerous_level_list]
    # Get the frequency and pdf
    eval_dangerous_level_list_freq = [eval_dangerous_level_list.count(val) for val in dangerous_level_list]
    print("Item name: {0}, sum of moments: {1}, value: {2}".format(col_name, sum(eval_dangerous_level_list_freq), eval_dangerous_level_list_freq))
    eval_dangerous_level_list_pdf = [val / sum(eval_dangerous_level_list_freq) for val in eval_dangerous_level_list_freq]

    return eval_dangerous_level_list_pdf


def plot_dangerous_level_dist_bar(all_episodes, dangerous_level_list, plot_item_list=None, crash_dangerous_level=99, unavoidable_dangerous_level=4, figsize=(16, 9),
                                  save_flag=False, fig_save_address=None):
    dangerous_level_list_pdf_list = []
    for item in plot_item_list:
        dangerous_level_list_pdf = _get_dangerous_level_pdf(all_episodes, dangerous_level_list, col_name=item,
                                        crash_dangerous_level=crash_dangerous_level, unavoidable_dangerous_level=unavoidable_dangerous_level)

        each_item_fig_save_address = os.path.join(fig_save_address, item)
        _plot_dangerous_level_dist_bar(dangerous_level_list_pdf, dangerous_level_list, figsize=figsize, save_flag=save_flag, fig_save_address=each_item_fig_save_address)
        dangerous_level_list_pdf_list.append(dangerous_level_list_pdf)

    # Plot comparison figure if more than one item.
    if len(plot_item_list) > 1:
        _plot_multiple_dangerous_level_dist_bar(dangerous_level_list_pdf_list, plot_item_list, figsize=figsize, save_flag=save_flag, fig_save_address=fig_save_address)


def _plot_multiple_dangerous_level_dist_bar(dangerous_level_list_pdf_list, plot_item_list, figsize=(16, 9), save_flag=False, fig_save_address=None):
    mpl.rcParams['font.size'] = 30
    plt.rcParams["font.family"] = "Times New Roman"

    plt.figure(figsize=figsize)
    legend_list = []
    for item in plot_item_list:
        if item == "dangerous_level": legend_list.append("Post-trip")
        if item == "mapped_MPrTTC": legend_list.append("MPrISM")
        if item == "mapped_PCM": legend_list.append("PCM")
        if item == "mapped_TTC": legend_list.append("TTC")
    df = pd.DataFrame(np.array(dangerous_level_list_pdf_list).T, columns=legend_list, index=["Safe", "1", "2", "3", "4"])
    df.plot.bar(logy=True)

    plt.xlabel("Risk Level")
    plt.ylabel("Percentage")
    plt.ylim(1e-3, 2)
    plt.legend(fontsize=20)

    if save_flag:
        assert (fig_save_address is not None)
        compare_fig_save_address = os.path.join(fig_save_address, "compare")
        plt.savefig(compare_fig_save_address + ".svg", bbox_inches="tight")
        plt.savefig(compare_fig_save_address + ".pdf", bbox_inches="tight")
        plt.savefig(compare_fig_save_address + ".png", dpi=300, bbox_inches="tight")


def _plot_dangerous_level_dist_bar(dangerous_level_list_pdf, dangerous_level_list,figsize=(16, 9), save_flag=False, fig_save_address=None):
    mpl.rcParams['font.size'] = 60
    plt.rcParams["font.family"] = "Times New Roman"


    fig, ax = plt.subplots(figsize=figsize)
    rects1 = ax.bar(dangerous_level_list, height=dangerous_level_list_pdf, log=True, tick_label=["Safe", "1", "2", "3", "4"])
    plt.xlabel("Risk Level")
    plt.ylabel("Percentage")
    plt.ylim(1e-3, 2)

    def autolabel(rects, text_color="k"):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            if height != 0:
                ax.annotate('{}%'.format(round(100 * height, 3)),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', color=text_color, fontsize=40)

    autolabel(rects1, text_color="k")

    if save_flag:
        assert (fig_save_address is not None)
        plt.savefig(fig_save_address + ".svg", bbox_inches="tight")
        plt.savefig(fig_save_address + ".pdf", bbox_inches="tight")
        plt.savefig(fig_save_address + ".png", dpi=300, bbox_inches="tight")
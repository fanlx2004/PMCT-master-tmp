from mtlsp.simulator import Simulator
from envs.nade import NADE
from controller.treesearchnadecontroller import TreeSearchNADEBackgroundController
from conf import conf
import warnings
warnings.filterwarnings("ignore") 
from multiprocessing import current_process
import numpy as np


'''
    These functions are used in deduction from a given initial state to the end of the scenario. 
    Attention: These functions are only available under the three-lane autonomous driving system settings.
'''

def deduction_for_MP(p):

    lane_id_0 = "0to1_"
    AV_info = []
    BV_info = {}
    AV_place = np.random.randint(350, 450)
    AV_info = [AV_place, p[1], lane_id_0 + str(p[0])]
    
    AV_state = [
        float(AV_place),
        42.0 + p[0] * 4.0,
        p[1],
        0.0,
        90.0,
        0.0
    ]
    BV_list = []
    for j in range(2, 14, 2):
        if p[j] is None or p[j+1] is None:
            BV_list.append(None)
        else:
            if j < 6:
                y = 42.0 + p[0] * 4.0
            elif 6 <= j and j < 10:
                y = 46.0 + p[0] * 4.0
            else:
                y = 38.0 + p[0] * 4.0
            BV_list.append([
                float(AV_place + p[j]),
                y,
                p[j+1],
                90.0,
                0.0
            ])

    if p[2] == None and p[3] == None:
        BV_info["Lead"] = None
    else:
        BV_info["Lead"] = [AV_place + p[2], p[3], lane_id_0 + str(p[0])]
    if p[4] == None and p[5] == None:
        BV_info["Foll"] = None
    else:
        BV_info["Foll"] = [AV_place + p[4], p[5], lane_id_0 + str(p[0])]
    if p[0] == 0:
        if p[6] == None and p[7] == None:
            BV_info["LeftLead"] = None
        else:
            BV_info["LeftLead"] = [AV_place + p[6], p[7], lane_id_0 + "1"]
        if p[8] == None and p[9] == None:
            BV_info["LeftFoll"] = None
        else:
            BV_info["LeftFoll"] = [AV_place + p[8], p[9], lane_id_0 + "1"]
        BV_info["RightLead"] = None
        BV_info["RightFoll"] = None
    elif p[0] == 1:
        if p[6] == None and p[7] == None:
            BV_info["LeftLead"] = None
        else:
            BV_info["LeftLead"] = [AV_place + p[6], p[7], lane_id_0 + "2"]
        if p[8] == None and p[9] == None:
            BV_info["LeftFoll"] = None
        else:
            BV_info["LeftFoll"] = [AV_place + p[8], p[9], lane_id_0 + "2"]
        if p[10] == None and p[11] == None:
            BV_info["RightLead"] = None
        else:
            BV_info["RightLead"] = [AV_place + p[10], p[11], lane_id_0 + "0"]
        if p[12] == None and p[13] == None:
            BV_info["RightFoll"] = None
        else:
            BV_info["RightFoll"] = [AV_place + p[12], p[13], lane_id_0 + "0"]
    else:
        BV_info["LeftLead"] = None
        BV_info["LeftFoll"] = None
        if p[10] == None and p[11] == None:
            BV_info["RightLead"] = None
        else:
            BV_info["RightLead"] = [AV_place + p[10], p[11], lane_id_0 + "1"]
        if p[12] == None and p[13] == None:
            BV_info["RightFoll"] = None
        else:
            BV_info["RightFoll"] = [AV_place + p[12], p[13], lane_id_0 + "1"]
    
    conf.experiment_config["mode"] = "NDE"
    conf.simulation_config["epsilon_setting"] = "fixed"
    
    
    env = NADE(BVController=TreeSearchNADEBackgroundController, cav_model=conf.experiment_config["AV_model"])
    sumo_net_file_path = './maps/road.net.xml'
    sumo_config_file_path = './maps/environment.sumocfg'

    sim = Simulator(
            sumo_net_file_path=sumo_net_file_path,
            sumo_config_file_path=sumo_config_file_path,
            num_tries=round(conf.K * conf.delta_t / conf.simulation_step), # 50
            step_size=conf.simulation_step,
            action_step_size=conf.simulation_step,
            lc_duration=1,
            track_cav=conf.simulation_config["gui_flag"],
            sublane_flag=True,
            gui_flag=conf.simulation_config["gui_flag"],
            # output=["fcd"],
            output=[]
    )
    sim.bind_env(env)
    data_dict = sim.sample_run(AV_info=AV_info, BV_info=BV_info)
    n = len(data_dict)
    return [conf.simulation_step*n, [AV_state, BV_list]]


def deduction(p=None, get_time=False, all_state_flag=False):
    if p is None:
        AV_info = None
        BV_info = None
    else :
        lane_id_0 = "0to1_"
        AV_info = []
        BV_info = {}
        AV_place = np.random.randint(350, 450)
        AV_info = [AV_place, p[1], lane_id_0 + str(p[0])]
        if p[2] == None and p[3] == None:
            BV_info["Lead"] = None
        else:
            BV_info["Lead"] = [AV_place + p[2], p[3], lane_id_0 + str(p[0])]
        if p[4] == None and p[5] == None:
            BV_info["Foll"] = None
        else:
            BV_info["Foll"] = [AV_place + p[4], p[5], lane_id_0 + str(p[0])]
        if p[0] == 0:
            if p[6] == None and p[7] == None:
                BV_info["LeftLead"] = None
            else:
                BV_info["LeftLead"] = [AV_place + p[6], p[7], lane_id_0 + "1"]
            if p[8] == None and p[9] == None:
                BV_info["LeftFoll"] = None
            else:
                BV_info["LeftFoll"] = [AV_place + p[8], p[9], lane_id_0 + "1"]
            BV_info["RightLead"] = None
            BV_info["RightFoll"] = None
        elif p[0] == 1:
            if p[6] == None and p[7] == None:
                BV_info["LeftLead"] = None
            else:
                BV_info["LeftLead"] = [AV_place + p[6], p[7], lane_id_0 + "2"]
            if p[8] == None and p[9] == None:
                BV_info["LeftFoll"] = None
            else:
                BV_info["LeftFoll"] = [AV_place + p[8], p[9], lane_id_0 + "2"]
            if p[10] == None and p[11] == None:
                BV_info["RightLead"] = None
            else:
                BV_info["RightLead"] = [AV_place + p[10], p[11], lane_id_0 + "0"]
            if p[12] == None and p[13] == None:
                BV_info["RightFoll"] = None
            else:
                BV_info["RightFoll"] = [AV_place + p[12], p[13], lane_id_0 + "0"]
        else:
            BV_info["LeftLead"] = None
            BV_info["LeftFoll"] = None
            if p[10] == None and p[11] == None:
                BV_info["RightLead"] = None
            else:
                BV_info["RightLead"] = [AV_place + p[10], p[11], lane_id_0 + "1"]
            if p[12] == None and p[13] == None:
                BV_info["RightFoll"] = None
            else:
                BV_info["RightFoll"] = [AV_place + p[12], p[13], lane_id_0 + "1"]
    
    #conf.experiment_config["mode"] = "D2RL"
    #conf.simulation_config["epsilon_setting"] = "drl"
    conf.experiment_config["mode"] = "NDE"
    conf.simulation_config["epsilon_setting"] = "fixed"
    data_dict = {}
    sumo_net_file_path = './maps/road.net.xml'
    sumo_config_file_path = './maps/environment.sumocfg'
    #d2rl_agent_path = "./checkpoints/2lane_400m/model.pt"
    #conf.discriminator_agent = conf.load_discriminator_agent(checkpoint_path=d2rl_agent_path)
    if p is None:
        env = NADE(BVController=TreeSearchNADEBackgroundController, cav_model=conf.experiment_config["AV_model"], scenario_total_length=60)
        sim = Simulator(
            sumo_net_file_path=sumo_net_file_path,
            sumo_config_file_path=sumo_config_file_path,
            num_tries=600, 
            step_size=conf.simulation_step,
            action_step_size=conf.simulation_step,
            lc_duration=1,
            track_cav=conf.simulation_config["gui_flag"],
            sublane_flag=True,
            gui_flag=conf.simulation_config["gui_flag"],
            output=[]
        )
    else:
        env = NADE(BVController=TreeSearchNADEBackgroundController, cav_model=conf.experiment_config["AV_model"])
        sim = Simulator(
            sumo_net_file_path=sumo_net_file_path,
            sumo_config_file_path=sumo_config_file_path,
            num_tries=round(conf.K * conf.delta_t / conf.simulation_step), # 50
            step_size=conf.simulation_step,
            action_step_size=conf.simulation_step,
            lc_duration=1,
            track_cav=conf.simulation_config["gui_flag"],
            sublane_flag=True,
            gui_flag=conf.simulation_config["gui_flag"],
            output=[]
        )
    
    sim.bind_env(env)
    
    data_dict = sim.sample_run(AV_info=AV_info, BV_info=BV_info)
    
    if get_time:
        time = len(data_dict) * conf.simulation_step
        return time
            
    r = (len(data_dict)) // round(conf.delta_t/conf.simulation_step) + 1 
    if p is None:
        point_list = []
    else:
        point_list = [p]
    

    if all_state_flag:

        for key, value in data_dict.items():
            point_0 = value
            point = [point_0["Ego"]["lane_index"], point_0["Ego"]["velocity"]]
            if point_0["Lead"] != None:
                point.append(point_0["Lead"]["position"][0] - point_0["Ego"]["position"][0])
                point.append(point_0["Lead"]["velocity"])
            else:
                point.append(None)
                point.append(None)
            if point_0["Foll"] != None:
                point.append(point_0["Foll"]["position"][0] - point_0["Ego"]["position"][0])
                point.append(point_0["Foll"]["velocity"])
            else:
                point.append(None)
                point.append(None)
            if point_0["LeftLead"] != None:
                point.append(point_0["LeftLead"]["position"][0] - point_0["Ego"]["position"][0])
                point.append(point_0["LeftLead"]["velocity"])
            else:
                point.append(None)
                point.append(None)
            if point_0["LeftFoll"] != None:
                point.append(point_0["LeftFoll"]["position"][0] - point_0["Ego"]["position"][0])
                point.append(point_0["LeftFoll"]["velocity"])
            else:
                point.append(None)
                point.append(None)
            if point_0["RightLead"] != None:
                point.append(point_0["RightLead"]["position"][0] - point_0["Ego"]["position"][0])
                point.append(point_0["RightLead"]["velocity"])
            else:
                point.append(None)
                point.append(None)
            if point_0["RightFoll"] != None:
                point.append(point_0["RightFoll"]["position"][0] - point_0["Ego"]["position"][0])
                point.append(point_0["RightFoll"]["velocity"])
            else:
                point.append(None)
                point.append(None)
            point_list.append(point)
    else:

        for i in range(1, r):
            point_0 = data_dict[str(round(conf.delta_t/conf.simulation_step)*i-1)]
            point = [point_0["Ego"]["lane_index"], point_0["Ego"]["velocity"]]
            if point_0["Lead"] != None:
                point.append(point_0["Lead"]["position"][0] - point_0["Ego"]["position"][0])
                point.append(point_0["Lead"]["velocity"])
            else:
                point.append(None)
                point.append(None)
            if point_0["Foll"] != None:
                point.append(point_0["Foll"]["position"][0] - point_0["Ego"]["position"][0])
                point.append(point_0["Foll"]["velocity"])
            else:
                point.append(None)
                point.append(None)
            if point_0["LeftLead"] != None:
                point.append(point_0["LeftLead"]["position"][0] - point_0["Ego"]["position"][0])
                point.append(point_0["LeftLead"]["velocity"])
            else:
                point.append(None)
                point.append(None)
            if point_0["LeftFoll"] != None:
                point.append(point_0["LeftFoll"]["position"][0] - point_0["Ego"]["position"][0])
                point.append(point_0["LeftFoll"]["velocity"])
            else:
                point.append(None)
                point.append(None)
            if point_0["RightLead"] != None:
                point.append(point_0["RightLead"]["position"][0] - point_0["Ego"]["position"][0])
                point.append(point_0["RightLead"]["velocity"])
            else:
                point.append(None)
                point.append(None)
            if point_0["RightFoll"] != None:
                point.append(point_0["RightFoll"]["position"][0] - point_0["Ego"]["position"][0])
                point.append(point_0["RightFoll"]["velocity"])
            else:
                point.append(None)
                point.append(None)
            point_list.append(point)

    return point_list
                 

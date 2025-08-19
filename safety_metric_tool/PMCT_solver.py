from network_structure.Q_net import DRLAgent
import torch
import numpy as np
from conf import conf

def PMCT_solver(point_list):
    '''
        This function is used to calculate the PMCT of the input states.
        the function supports the input of a list of states or a single state.
        Arg:
            point_list: a list of states (or a single state), each state is a list of 14 elements.
            [AV_lane_id, AV_vx, BV_Lead_distance, BV_Lead_vx, BV_Foll_distance, BV_Foll_vx,
            BV_LeftLead_distance, BV_LeftLead_vx, BV_LeftFoll_distance, BV_LeftFoll_vx,
            BV_RightLead_distance, BV_RightLead_vx, BV_RightFoll_distance, BV_RightFoll_vx]
        Return:
            PMCT_list: a list of PMCT values (or a single value), each value is a float.
    '''
    agent = DRLAgent()
    agent.q_net.load_state_dict(torch.load(conf.final_net_params_path))
    PMCT_list = (agent.get_q_value(point_list) - 1) * conf.delta_t
    return PMCT_list.tolist()
    
    

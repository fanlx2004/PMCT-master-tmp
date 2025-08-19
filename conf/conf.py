from math import exp
from .defaultconf import *
import torch
import numpy as np

STATE_DIM = 14   # Number of state dimensions
ENCODED_DIM = 128   # Dimension of the encoding layer
K = 10      # Number of steps           
LR = 1e-4   # Learning rate for the optimizer        
BATCH_SIZE = 256  # Batch size for training
BUFFER_SIZE = 10000 # Size of the replay buffer
epsilons = np.ones(K+1) * 0.05 # Failure probability for each subset
beta1 = 1e-3 # Significance level for the verification
sample_per_adding = 300 # Number of samples to add to the buffer in each iteration
training_net_params_path = "./param_save/net_params_undertraining.pth"  # Path to save the network parameters
failure_data_path = "./result/failure_data"  # Path to the pre-collected failure data used in verification
update_times_per_epoch = 10  # Number of training iterations per epoch
verification_interval = 10  # Interval for verification during training
loss_save_interval = 25  # Interval for saving the loss during training
lb = np.array([20, 6, 20, -115, 20, 6, 20, -115, 20, 6, 20, -115, 20], dtype=float) # Lower bound for the state space
ub = np.array([40, 115, 40, -6, 40, 115, 40, -6, 40, 115, 40, -6, 40], dtype=float) # Upper bound for the state space
delta_t = 0.5 # Time step for the set partitioning
simulation_step = 0.1 # Simulation step size
final_net_params_path = "./param_save/net_params_final.pth" # Path to save the final network parameters




weight_threshold = 0
epsilon_value = 0.99

simulation_config["map"] = "3Lane" # simulation map definition
d2rl_agent_path = "./checkpoints/model.pt" # the pytorch checkpoint of the D2RL agent, not used in our code

simulation_config["epsilon_type"] = "continious" # define whether the d2rl agent will output continious/discrete adversarial action probability
experiment_config["AV_model"] = "IDM" # Tested AV models
simulation_config["speed_mode"] = "high_speed" # the speed profile of the vehicles in the simulation
simulation_config["gui_flag"] = False # whether to show the simulation in GUI

discriminator_agent = None
experiment_config["root_folder"] = "./data_analysis/raw_data" # the folder to save the simulation data
experiment_config["episode_num"] = 1232

# tree search-based maneuver challenge calculation configuration
treesearch_config["search_depth"] = 1
treesearch_config["surrogate_model"] = "AVI" # "AVI" "surrogate"
treesearch_config["offline_leaf_evaluation"] = False
treesearch_config["offline_discount_factor"] = 1
treesearch_config["treesearch_discount_factor"] = 1

simulation_config["initialization_rejection_sampling_flag"] = False
experiment_config["log_mode"] = "crash" # "all" "crash"
traffic_flow_config["BV"] = True
Surrogate_LANE_CHANGE_MAX_BRAKING_IMPOSED = 8  # [m/s2]

# D2RL-based agent definition
class torch_discriminator_agent:
    def __init__(self, checkpoint_path):
        if not checkpoint_path:
            checkpoint_path = "./model.pt"
        #print("Loading checkpoint", checkpoint_path)
        self.model = torch.jit.load(checkpoint_path)
        self.model.eval()

    def compute_action(self, observation):
        lb = 0.001
        ub = 0.999
        obs = torch.reshape(torch.tensor(observation), (1,len(observation)))
        out = self.model({"obs":obs},[torch.tensor([0.0])],torch.tensor([1]))
        if simulation_config["epsilon_type"] == "discrete":
            action = torch.argmax(out[0][0])
        else:
            action = np.clip((float(out[0][0][0])+1)*(ub-lb)/2 + lb, lb, ub)
        return action

# load pytorch checkpoints into the D2RL agent
def load_discriminator_agent(mode="torch", checkpoint_path=d2rl_agent_path):
    
    if mode == "torch":
        discriminator_agent = torch_discriminator_agent(checkpoint_path)
    else:
        raise NotImplementedError("unsupported mode in discriminator agent load!")
    return discriminator_agent
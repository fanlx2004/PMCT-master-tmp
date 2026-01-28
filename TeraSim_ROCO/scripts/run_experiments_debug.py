import argparse
import random
import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm import tqdm
from terasim.logger.infoextractor import InfoExtractor
from terasim.simulator import Simulator
from datetime import datetime
from terasim_nde_nade.envs import NADE, NADEWithAV
from terasim_nde_nade.vehicle import NDEVehicleFactory
from terasim_nde_nade.vru import NDEVulnerableRoadUserFactory
import time
# Import resolve_config_paths function
from terasim_service.utils.base import resolve_config_paths

# Add packages directory to sys path if needed
# sys.path.append(str(Path(__file__).resolve().parent.parent))

logger.remove()
def main(config_path, id=0):
    start_time = time.time()
    config = OmegaConf.load(config_path)
    # Convert OmegaConf to dict for path resolution
    config_dict = OmegaConf.to_container(config, resolve=True)
    # Resolve all paths in config
    config_dict = resolve_config_paths(config_dict, config_path)

    # Convert back to OmegaConf for attribute access
    config = OmegaConf.create(config_dict)
    current_time = datetime.now()
    time_str3 = current_time.strftime("%Y%m%d_%H%M%S") + '_' + str(id)
    base_dir = Path('data/terasim_data/temp/' + time_str3)
    base_dir.mkdir(parents=True, exist_ok=True)
    env = NADE( # NADEWithAV or NADE
        # av_cfg = config.environment.parameters.AV_cfg,
        vehicle_factory=NDEVehicleFactory(cfg=config.environment.parameters),
        vru_factory=NDEVulnerableRoadUserFactory(cfg=config.environment.parameters),
        info_extractor=InfoExtractor,
        log_flag=True,
        log_dir=base_dir,
        warmup_time_lb=config.environment.parameters.warmup_time_lb,
        warmup_time_ub=config.environment.parameters.warmup_time_ub,
        run_time=config.environment.parameters.run_time,
        configuration=config.environment.parameters,
        # av_debug_control=True, # Enable debug control for AV, will use SUMO
    )

    # Paths already resolved in config
    sumo_net_file = config.input.sumo_net_file
    sumo_config_file = config.input.sumo_config_file
    # sumo_additional_file = config.input.sumo_additional_file
    sumo_additional_file = "./vTypeDistributions.add.xml"

    sim = Simulator(
        sumo_net_file_path=sumo_net_file,
        sumo_config_file_path=sumo_config_file,
        #sumo_additional_file_path=sumo_additional_file,
        num_tries=10,
        gui_flag=config.simulator.parameters.gui_flag,
        realtime_flag=config.simulator.parameters.realtime_flag,
        output_path=base_dir,
        sumo_output_file_types=["collision"],
        traffic_scale=(
            config.simulator.parameters.traffic_scale
            if hasattr(config.simulator.parameters, "traffic_scale")
            else 1
        ),
        additional_sumo_args=[
            "--device.bluelight.explicit",
            "true",
        ],
    )
    sim.bind_env(env)

    terasim_logger = logger.bind(name="terasim_nde_nade")
    terasim_logger.info(f"terasim_nde_nade: Experiment started")

    sim.run()
    end_time = time.time()
    print("total time:", end_time - start_time)


if __name__ == "__main__":

    config_path = Path('examples/scenarios/Mcity_safety_assessment.yaml')
    for j in range(10000):
        print('***********************')
        print('Data: ', j)
        print('***********************')
        main(str(config_path))
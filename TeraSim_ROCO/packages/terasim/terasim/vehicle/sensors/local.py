import traci.constants as tc
from addict import Dict

from terasim import utils
from terasim.agent.agent_sensor import AgentSensor
from terasim.overlay import traci
from terasim.simulator import Simulator


class LocalSensor(AgentSensor):
    """
    LocalSensor is a basic sensor that subscribe to some SUMO variables of a vehicle.

    A LocalSensor will maintain a observation, which is a nested dictionary observation.time_stamp
    observation: a dictionary{
        'Ego': {'veh_id': vehicle ID, 'speed': vehicle velocity [m/s], 'position': tuple of X,Y coordinates [m], 'heading': vehicle angle [degree], 'lane_index': lane index of vehicle, 'distance': 0 [m], 'acceleration': m/s^2},
        'Lead'
        'Foll'
        'LeftLead'
        'RightLead'
        'LeftFoll'
        'RightFoll'
    }
    """
    DEFAULT_PARAMS = dict(
        cache=True,  
        obs_range=120.0,  
    )
    def __init__(self, name="local", **params):
        """Initialize the local sensor for the vehicle.

        Args:
            name (str, optional): The name of the sensor. Defaults to "local".
            params (dict, optional): The parameters of the sensor.
        """
        super().__init__(name, **params)

    def fetch(self):
        """Fetch the vehicle information.

        Returns:
            dict: The vehicle information.
        """
        common = dict(vehID=self._agent.id, obs_range=self._params.obs_range)

        return Dict(
            Ego=LocalSensor.get_ego_vehicle_info(veh_id=self._agent.id),
            Lead=utils.get_leading_vehicle(**common),
            LeftLead=utils.get_neighboring_leading_vehicle(dir="left", **common),
            RightLead=utils.get_neighboring_leading_vehicle(dir="right", **common),
            Foll=utils.get_following_vehicle(**common),
            LeftFoll=utils.get_neighboring_following_vehicle(dir="left", **common),
            RightFoll=utils.get_neighboring_following_vehicle(dir="right", **common),
        )

    @staticmethod
    def get_ego_vehicle_info(veh_id, distance=0.0):
        """Modify the vehicle information into a standard form.

        Args:
            veh_id (str, optional): Vehicle ID. Defaults to None.
            distance (float, optional): Distance from the ego vehicle [m]. Defaults to 0.0.

        Returns:
            dict: Standard form of vehicle information.
        """
        veh_info = Dict(
            veh_id=veh_id,
            velocity=traci.vehicle.getSpeed(veh_id),
            position=traci.vehicle.getPosition(veh_id),
            heading=traci.vehicle.getAngle(veh_id),
            acceleration=traci.vehicle.getAcceleration(veh_id),
            could_drive_adjacent_lane_left=Simulator.get_vehicle_lane_adjacent(veh_id, 1),
            could_drive_adjacent_lane_right=Simulator.get_vehicle_lane_adjacent(veh_id, -1),
            lateral_speed=traci.vehicle.getLateralSpeed(veh_id),
            lateral_offset=traci.vehicle.getLateralLanePosition(veh_id),
            yaw_rate=yawrate_function(veh_id),
        )
        return veh_info


def yawrate_function(vehID):

    current_time = traci.simulation.getTime()
    current_angle = traci.vehicle.getAngle(vehID)
    dt = traci.simulation.getDeltaT()

    last_update_time_str = traci.vehicle.getParameter(vehID, "last_update_time")
    prev_angle_str = traci.vehicle.getParameter(vehID, "last_angle")

    if last_update_time_str == "":
        traci.vehicle.setParameter(vehID, "last_update_time", str(current_time))
        traci.vehicle.setParameter(vehID, "last_angle", str(current_angle))
        return 0.0
    
    last_update_time = float(last_update_time_str)
    prev_angle = float(prev_angle_str)

    if current_time > last_update_time:

        delta_angle = current_angle - prev_angle

        if delta_angle > 180: delta_angle -= 360
        elif delta_angle < -180: delta_angle += 360
        
        yaw_rate = delta_angle / dt
        
        traci.vehicle.setParameter(vehID, "last_angle", str(current_angle))
        traci.vehicle.setParameter(vehID, "last_update_time", str(current_time))
 
        traci.vehicle.setParameter(vehID, "current_yaw_rate", str(yaw_rate))
    else:
        cached_yaw_rate = traci.vehicle.getParameter(vehID, "current_yaw_rate")
        yaw_rate = float(cached_yaw_rate) if cached_yaw_rate != "" else 0.0

    return yaw_rate


class LocalNODESensor(AgentSensor):
    DEFAULT_PARAMS = dict(
        cache=True, 
        obs_range=120.0,  
    )
    def __init__(self, name="local_node", **params):
        """Initialize the local sensor for the vehicle.

        Args:
            name (str, optional): The name of the sensor. Defaults to "local".
            params (dict, optional): The parameters of the sensor.
        """
        super().__init__(name, **params)

    def fetch(self):
        """Fetch the vehicle information.

        Returns:
            dict: The vehicle information.
        """
        common = dict(vehID=self._agent.id, obs_range=self._params.obs_range)
        # collect all other vehicles and compute distances to ego
        veh_list = traci.vehicle.getIDList()
        ego_pos = traci.vehicle.getPosition(self._agent.id)
        dists = []
        for vid in veh_list:
            if vid == self._agent.id:
                continue
            try:
                pos = traci.vehicle.getPosition(vid)
            except Exception:
                continue
            dx = pos[0] - ego_pos[0]
            dy = pos[1] - ego_pos[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < 120.0:
                dists.append((dist, vid))

        # sort by distance and pick up to 6 nearest, pad with None if fewer
        dists.sort(key=lambda x: x[0])
        nearest_six_ids = [vid for _, vid in dists[:6]]
        nearest_six_dist = [dis for dis, _ in dists[:6]]
        while len(nearest_six_ids) < 6:
            nearest_six_ids.append(None)
            nearest_six_dist.append(None)

        obs_list = []
        for i in range(6):
            if nearest_six_ids[i] != None:
                obs_list.append(LocalNODESensor.get_other_vehicle_info(veh_id=nearest_six_ids[i], distance=nearest_six_dist[i]))
            else:
                obs_list.append(None)
        
        return Dict(
            Ego=LocalNODESensor.get_ego_vehicle_info(veh_id=self._agent.id),
            Lead=obs_list[0],
            LeftLead=obs_list[1],
            RightLead=obs_list[2],
            Foll=obs_list[3],
            LeftFoll=obs_list[4],
            RightFoll=obs_list[5],
        )

    @staticmethod
    def get_ego_vehicle_info(veh_id, distance=0.0):
        """Modify the vehicle information into a standard form.

        Args:
            veh_id (str, optional): Vehicle ID. Defaults to None.
            distance (float, optional): Distance from the ego vehicle [m]. Defaults to 0.0.

        Returns:
            dict: Standard form of vehicle information.
        """
        veh_info = Dict(
            veh_id=veh_id,
            velocity=traci.vehicle.getSpeed(veh_id),
            position=traci.vehicle.getPosition(veh_id),
            heading=traci.vehicle.getAngle(veh_id),
            acceleration=traci.vehicle.getAcceleration(veh_id),
            could_drive_adjacent_lane_left=Simulator.get_vehicle_lane_adjacent(veh_id, 1),
            could_drive_adjacent_lane_right=Simulator.get_vehicle_lane_adjacent(veh_id, -1),
            lateral_speed=traci.vehicle.getLateralSpeed(veh_id),
            lateral_offset=traci.vehicle.getLateralLanePosition(veh_id),
            yaw_rate=yawrate_function(veh_id),
        )
        return veh_info
    
    def get_other_vehicle_info(veh_id, distance=0.0):
        """Modify the vehicle information into a standard form.

        Args:
            veh_id (str, optional): Vehicle ID. Defaults to None.
            distance (float, optional): Distance from the ego vehicle [m]. Defaults to 0.0.

        Returns:
            dict: Standard form of vehicle information.
        """
        veh_info = Dict(
            veh_id=veh_id,
            velocity=traci.vehicle.getSpeed(veh_id),
            distance = distance,
            position=traci.vehicle.getPosition(veh_id),
            heading=traci.vehicle.getAngle(veh_id),
            acceleration=traci.vehicle.getAcceleration(veh_id),
            lateral_speed=traci.vehicle.getLateralSpeed(veh_id),
            yaw_rate=yawrate_function(veh_id),
        )
        return veh_info
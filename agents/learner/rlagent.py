import time
from collections import deque
from typing import List, Tuple
import os

import carla
import numpy as np

from utils.config import Config, Action
from utils.utils import degrees_to_radians, velocity_kmh, velocity_ms, distance_actors
from utils.carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions
from utils.logger import log_debug, log_info
from benchmark.environment.world import ClientWorld
from assets.occupancy_grid import OccupancyGrid
from path_planner.hybridastar import HybridAStar
from benchmark.risk.risk_aware_path import PathPlanner
from ped_path_predictor.autobots.autobot_wrapper import AutoBotWrapperNew


# defines current car intention and reward
class RLAgent(object):

    def __init__(self, client: carla.Client, client_world: ClientWorld):
        self.client_world = client_world
        self.occupancy_grid = OccupancyGrid()

        # all of the below are only relevant for a single episode
        self.episode_observations: List[Tuple[float, float, float, float, float, float]] = []
        self.episode_rewards: List[float] = []
        self.episode_actions: List[Action] = []
        self.episode_ego_vehicle_speeds: List[float] = []
        self.episode_controls: List[carla.VehicleControl] = []
        # debugging
        self.episode_ego_vehicle_pedestrian_distance: List[float] = []

        # required for reconstructing the already driven path which is then drawn on the car intention image
        # angle is omitted as it is not required for drawing a line with cv2
        self.episode_ego_vehicle_past_trajectory: List[Tuple[float, float]] = []
        # the planned path for the ego vehicle at the current time step
        self.step_ego_vehicle_future_trajectory: List[Tuple[float, float, float]] = []
        # the past trajectory of the pedestrian in this episode
        self.episode_pedestrian_past_trajectory: List[Tuple[float, float]] = []
        # the predicted path for the pedestrian at the current time step (if any)
        self.step_pedestrian_future_trajectory: List[Tuple[float, float]] = []
        # car intention images of each step in this episode
        self.episode_birdview_car_intentions: List[np.ndarray] = []
        # track pedestrian visibility
        self.episode_pedestrian_visibility: List[bool] = []

        # episode flags
        self.episode_counter: int = -1
        self.is_terminal_state: bool = False
        self.has_ego_vehicle_reached_goal: bool = False
        self.is_ego_vehicle_in_collision: bool = False
        self.is_ego_vehicle_in_nearmiss: bool = False
        self.is_incoming_car_observable: bool = False

        obstacle = []
        
        self.grid_cost = np.ones((110, 310)) * 1000.0
        # Road Network
        self.grid_cost[7:13, 13:] = 1.0
        self.grid_cost[97:103, 13:] = 1.0
        self.grid_cost[7:, 7:13] = 1.0
        # Sidewalk Network
        self.grid_cost[4:7, 4:] = 50.0
        self.grid_cost[:, 4:7] = 50.0
        self.grid_cost[13:16, 13:] = 50.0
        self.grid_cost[94:97, 13:] = 50.0
        self.grid_cost[103:106, 13:] = 50.0
        self.grid_cost[13:16, 16:94] = 50.0

        self.min_x = -10
        self.max_x = 100
        self.min_y = -10
        self.max_y = 300
        self.path_planner = HybridAStar(
            self.min_x, self.max_x, self.min_y, self.max_y, obstacle, Config.Carla.EGO_VEHICLE_LENGTH
        )

        self.risk_cmp = np.zeros((110, 310))
        # Road Network
        self.risk_cmp[7:13, 13:] = 1.0
        self.risk_cmp[97:103, 13:] = 1.0
        self.risk_cmp[7:, 7:13] = 1.0
        # Sidewalk Network
        sidewalk_cost = 50.0
        self.risk_cmp[4:7, 4:] = sidewalk_cost
        self.risk_cmp[:, 4:7] = sidewalk_cost
        self.risk_cmp[13:16, 13:] = sidewalk_cost
        self.risk_cmp[94:97, 13:] = sidewalk_cost
        self.risk_cmp[103:106, 13:] = sidewalk_cost
        self.risk_cmp[13:16, 16:94] = sidewalk_cost

        # only used by HyLEAR
        if Config.RISK_AWARE_PATH:
            self.risk_path_planner = PathPlanner()

        # load pedestrian path prediction model
        if Config.PREDICT_PEDESTRIAN_PATH:
            self.pedestrian_path_predictor = AutoBotWrapperNew(
                model_file_path=os.path.join(os.getcwd(), "ped_path_predictor", "autobots", "autobot_ego.pth")
            )
            self.pedestrian_path_predictor.model.eval()

        # used for producing car intention images 
        self.birdview_car_intention_producer = BirdViewProducer(
            client,  # carla.client
            target_size=PixelDimensions(
                width=int(Config.Carla.CAR_INTENTION_IMAGE_WIDHT), 
                height=int(Config.Carla.CAR_INTENTION_IMAGE_HEIGHT)),
            # ~50m cropping 
            pixels_per_meter=(int(Config.Carla.CAR_INTENTION_IMAGE_WIDHT) + int(Config.Carla.CAR_INTENTION_IMAGE_HEIGHT))/2/50,
            crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
            all_parked_vehicle_transforms=self.client_world.all_parked_vehicle_transforms
        )
        

    # ============================================================================================== #
    #                                         SETUP METHODS                                          #
    # ============================================================================================== #
    def initialize_episode(self, episode_counter: int, scenario: Tuple):
        # update scene information
        self.scenario_id = scenario[0]
        # client world is initialized first, so these values are up-to-date at this point and refer to the current episode
        self.pedestrian_velocity = self.client_world.pedestrian_velocity
        self.pedestrian_crossing_distance = self.client_world.pedestrian_crossing_distance
        self.episode_ego_vehicle_goal_position = scenario[2]
        self.episode_ego_vehicle_start_position = scenario[3]

        # reset episode counters & flags
        self.episode_counter = episode_counter
        self.is_terminal_state = False
        self.is_incoming_car_observable = False
        self.has_ego_vehicle_reached_goal = False
        self.is_ego_vehicle_in_collision = False

        # reset episode buffers
        self.episode_observations.clear()
        self.episode_rewards.clear()
        self.episode_actions.clear()
        self.episode_ego_vehicle_speeds.clear()
        self.episode_controls.clear()
        self.episode_ego_vehicle_pedestrian_distance.clear()

        self.step_ego_vehicle_future_trajectory.clear()
        self.step_pedestrian_future_trajectory.clear()

        self.episode_ego_vehicle_past_trajectory.clear()
        self.episode_pedestrian_past_trajectory.clear()
        self.episode_birdview_car_intentions.clear()
        self.episode_pedestrian_visibility.clear()

        # first dummy reward for timestep t = -1 (previous reward of initial state)
        self.episode_rewards.append(0.0)
        # default action: maintain for timestep t = -1
        self.episode_actions.append(Action.MAINTAIN)


    def finalize_episode(self, episode_counter: int, non_conclusive: bool, step_counter: int):
        return
    

    # ============================================================================================== #
    #                                          CORE METHODS                                          #
    # ============================================================================================== #
    def get_current_observation(self, step_counter: int, skipped_step: bool = False):  
        # create observation components
        ego_vehicle_position = [
            self.client_world.ego_vehicle.get_location().x, self.client_world.ego_vehicle.get_location().y
        ]
        # in m/s
        ego_vehicle_velocity = velocity_ms(self.client_world.ego_vehicle)
        ego_vehicle_angle_degrees = self.client_world.ego_vehicle.get_transform().rotation.yaw

        # pedestrian
        if self.client_world.is_pedestrian_observable:
            pedestrian_position = [
                self.client_world.pedestrian.get_location().x, self.client_world.pedestrian.get_location().y
            ]
            self.episode_pedestrian_past_trajectory.append(pedestrian_position)
        else:
            pedestrian_position = [None, None]

        # reached goal?
        ego_vehicle_goal_distance = np.linalg.norm(
            [ego_vehicle_position[0] - self.episode_ego_vehicle_goal_position[0], 
             ego_vehicle_position[1] - self.episode_ego_vehicle_goal_position[1]]
        )
        self.has_ego_vehicle_reached_goal = ego_vehicle_goal_distance <= Config.Carla.GOAL_TOLERANCE
        # in collision?
        self.is_ego_vehicle_in_collision = \
            self.client_world.is_ego_vehicle_in_collision or \
            self.in_rectangle(
                ego_vehicle_position[0], ego_vehicle_position[1], 
                ego_vehicle_angle_degrees, 
                pedestrian_position[0], pedestrian_position[1],
                front_margin=0.0, side_margin=0.0, back_margin=0.0
            )
        # check for terminal state
        self.is_terminal_state = self.has_ego_vehicle_reached_goal or self.is_ego_vehicle_in_collision
        # nearmiss
        self.is_ego_vehicle_in_nearmiss = self.in_rectangle(
                ego_vehicle_position[0], ego_vehicle_position[1], 
                ego_vehicle_angle_degrees, 
                pedestrian_position[0], pedestrian_position[1],
                front_margin=1.5, side_margin=0.5, back_margin=0.5
        )

        # do not remember step if it was skipped (as agent did not operate upon it)
        if skipped_step: return {
            "nearmiss": self.is_ego_vehicle_in_nearmiss, 
            "collision": self.is_ego_vehicle_in_collision,
            "goal": self.has_ego_vehicle_reached_goal,
            "terminal": self.is_terminal_state
        }

        self.episode_observations.append(
            (*ego_vehicle_position, ego_vehicle_velocity, ego_vehicle_angle_degrees, *pedestrian_position)
        )
        self.episode_ego_vehicle_speeds.append(ego_vehicle_velocity)
        self.episode_ego_vehicle_past_trajectory.append(ego_vehicle_position)
        self.episode_pedestrian_visibility.append(self.client_world.is_pedestrian_observable)
        self.episode_ego_vehicle_pedestrian_distance.append(
            distance_actors(self.client_world.ego_vehicle, self.client_world.pedestrian)
        )
        # sanity checks
        if len(self.episode_observations) != step_counter:
            raise ValueError(
                f"Invalid numer of episode observations: "
                f"Expected {step_counter}, got {len(self.episode_observations)} instead."
            )
        if len(self.episode_ego_vehicle_past_trajectory) != step_counter:
            raise ValueError(
                f"Invalid length of ego-vehicle's past trajectory: "
                f"Expected {step_counter}, got {len(self.episode_ego_vehicle_past_trajectory)} instead."
            )     
        if len(self.episode_ego_vehicle_speeds) != step_counter:
            raise ValueError(
                f"Invalid numer of episode ego vehicle speeds: "
                f"Expected {step_counter + 1}, got {len(self.episode_ego_vehicle_speeds)} instead."
            )   


    def get_birdview_car_intention(
            self, 
            ego_vehicle_transform: carla.Transform, 
            pedestrian_transform: carla.Transform, 
            episode_ego_vehicle_past_trajectory: List[Tuple[float, float]],
            step_ego_vehicle_future_trajectory: List[Tuple[float, float, float]],
            step_counter: int
    ) -> np.ndarray:
        
        if np.shape(step_ego_vehicle_future_trajectory)[1] != 3:
            raise ValueError(
                f"Invalid dimensions of ego vehicle's future trajectory: "
                f"Expected (n, 3), got {np.shape(step_ego_vehicle_future_trajectory)}."
            )

        # returned result is np.ndarray with ones and zeros of shape (8, height, width)
        birdview_car_intention = self.birdview_car_intention_producer.produce(
            ego_vehicle_transform,
            pedestrian_transform,
            episode_ego_vehicle_past_trajectory.copy(),
            # angle is omitted as it is not required for drawing a line with cv2
            [waypoint[:-1] for waypoint in step_ego_vehicle_future_trajectory.copy()],
            self.scenario_id, # scenario id: required for loading parked cars
            self.step_pedestrian_future_trajectory # required for drawing pedestrian path on birdview image
        )

        # produces np.ndarray of shape (height, width, 3)
        birdview_car_intention = self.birdview_car_intention_producer.as_rgb(birdview_car_intention)
        self.episode_birdview_car_intentions.append(birdview_car_intention)
   
        return birdview_car_intention


    def get_reward_despot(self, step_counter: int):
        # episode actions
        if not isinstance(self.episode_actions, list):
            raise TypeError(
                f"Invalid episode actions buffer type: Expected 'list', got '{type(self.episode_actions)}'."
            )
        if not isinstance(self.episode_actions[-1], Action):
            raise TypeError(
                f"Invalid previous action type: Expected '{Action}', got '{type(self.episode_actions[-1])}'."
            )
        # + 1 because of first dummy action (maintain) and current action is needed for calculating reward
        if len(self.episode_actions) != step_counter + 1:
            raise ValueError(
                f"Invalid number of episode actions: Expected {step_counter + 1}, got {len(self.episode_actions)}."
            )
        
        # episode car speeds
        if not isinstance(self.episode_ego_vehicle_speeds, list):
            raise TypeError(
                f"Invalid episode car speeds buffer type: Expected 'list', got '{type(self.episode_ego_vehicle_speeds)}'."
            )
        if not isinstance(self.episode_ego_vehicle_speeds[-1], float):
            raise TypeError(
                f"Invalid previous car speed type: Expected '{float}', got '{type(self.episode_ego_vehicle_speeds[-1])}'."
            )
        # because self.get_current_observation() must be called before self.get_reward()
        if len(self.episode_ego_vehicle_speeds) != step_counter:
            raise ValueError(
                f"Invalid numer of episode ego vehicle speeds: "
                f"Expected {step_counter}, got {len(self.episode_ego_vehicle_speeds)} instead."
            )    
        
        # episode carla vehicle controls
        if not isinstance(self.episode_controls, list):
            raise TypeError(
                f"Invalid episode vehicle controls buffer type: Expected 'list', got '{type(self.episode_controls)}'."
            )
        if not isinstance(self.episode_controls[-1], type(carla.VehicleControl())):
            raise TypeError(
                f"Invalid previous vehicle control type: "
                f"Expected '{carla.VehicleControl()}', got '{type(self.episode_controls[-1])}'."
            )      
        # because control will be set before calling reward function
        if len(self.episode_controls) != step_counter:
            raise ValueError(
                f"Invalid numer of episode vehicle control objects: "
                f"Expected {step_counter}, got {len(self.episode_controls)} instead."
            )
        
        # because of first dummy reward (0.0) 
        if len(self.episode_rewards) != step_counter:
            raise ValueError(
                f"Invalid number of episode rewards: Expected {step_counter}, got {len(self.episode_rewards)}."
            )

        current_observation = self.episode_observations[-1]
        ego_vehicle_speed = current_observation[2]

        goal = hit = speeding = braking = False
        reward = 0
        goal_reward = 1
        # IS-DESPOT's veloicty scaled crash penalty
        hit_penalty = speeding_penalty = braking_penalty = \
            -1 * pow(0.5 + ego_vehicle_speed / Config.Carla.MAX_SPEED, 1.4) - 0.2

        # has ego vehicle reached goal?
        ego_vehicle_goal_distance = np.linalg.norm(
            [current_observation[0] - self.episode_ego_vehicle_goal_position[0], 
             current_observation[1] - self.episode_ego_vehicle_goal_position[1]]
        )
        if ego_vehicle_goal_distance <= Config.Carla.GOAL_TOLERANCE:
            reward += goal_reward
            goal = True

        # collision and nearmiss check
        hit = self.in_rectangle(
            current_observation[0], current_observation[1], 
            current_observation[3], 
            current_observation[4], current_observation[5],
            front_margin=1.5, side_margin=0.5, back_margin=0.5
        )
        hit = self.client_world.is_ego_vehicle_in_collision or (hit and ego_vehicle_speed > 0.01)
        if hit: 
            reward -= hit_penalty
            goal = False

        # acceleration rate = 5km/h per second of simulation
        # one simulation step = 50ms, thus 1s/20 = 50ms
        if self.episode_actions[-1] is Action.ACCELERATE and ego_vehicle_speed >= Config.Carla.MAX_SPEED:
            reward -= speeding_penalty
            speeding = True

        # "heavily" penalize braking if you are already standing still
        if self.episode_actions[-1] is Action.DECELERATE and ego_vehicle_speed <= 0.01:
            reward -= braking_penalty
            braking = True

        # smoothnes control
        if self.episode_actions[-1] is Action.DECELERATE or self.episode_actions[-1] is Action.ACCELERATE:
            reward -= 0.01

        # penalize low velocities (required for IS-DESPOT to work...)
        reward += (0.5 * (ego_vehicle_speed - Config.Carla.MAX_SPEED) / Config.Carla.MAX_SPEED) / 100.0

        self.is_terminal_state = hit or goal #or speeding or braking
        self.episode_rewards.append(reward)

        step_summary = {
            "reward": reward,
            "goal": goal,
            "collision": hit,
            "nearmiss": hit,
            "terminal": self.is_terminal_state,
            "ped_observable": self.client_world.is_pedestrian_observable
        }
        log_debug(step_summary)
        return step_summary
    

    def get_reward_nils(self, step_counter: int):
        # episode actions
        if not isinstance(self.episode_actions, list):
            raise TypeError(
                f"Invalid episode actions buffer type: Expected 'list', got '{type(self.episode_actions)}'."
            )
        if not isinstance(self.episode_actions[-1], Action):
            raise TypeError(
                f"Invalid previous action type: Expected '{Action}', got '{type(self.episode_actions[-1])}'."
            )
        # + 1 because of first dummy action (maintain) and current action is needed for calculating reward
        if len(self.episode_actions) != step_counter + 1:
            raise ValueError(
                f"Invalid number of episode actions: Expected {step_counter + 1}, got {len(self.episode_actions)}."
            )
        
        # episode car speeds
        if not isinstance(self.episode_ego_vehicle_speeds, list):
            raise TypeError(
                f"Invalid episode car speeds buffer type: Expected 'list', got '{type(self.episode_ego_vehicle_speeds)}'."
            )
        if not isinstance(self.episode_ego_vehicle_speeds[-1], float):
            raise TypeError(
                f"Invalid previous car speed type: Expected '{float}', got '{type(self.episode_ego_vehicle_speeds[-1])}'."
            )
        # because self.get_current_observation() must be called before self.get_reward()
        if len(self.episode_ego_vehicle_speeds) != step_counter:
            raise ValueError(
                f"Invalid numer of episode ego vehicle speeds: "
                f"Expected {step_counter}, got {len(self.episode_ego_vehicle_speeds)} instead."
            )    
        
        # episode carla vehicle controls
        if not isinstance(self.episode_controls, list):
            raise TypeError(
                f"Invalid episode vehicle controls buffer type: Expected 'list', got '{type(self.episode_controls)}'."
            )
        if not isinstance(self.episode_controls[-1], type(carla.VehicleControl())):
            raise TypeError(
                f"Invalid previous vehicle control type: "
                f"Expected '{carla.VehicleControl()}', got '{type(self.episode_controls[-1])}'."
            )      
        # because control will be set before calling reward function
        if len(self.episode_controls) != step_counter:
            raise ValueError(
                f"Invalid numer of episode vehicle control objects: "
                f"Expected {step_counter}, got {len(self.episode_controls)} instead."
            )
        
        # because of first dummy reward (0.0) 
        if len(self.episode_rewards) != step_counter:
            raise ValueError(
                f"Invalid number of episode rewards: Expected {step_counter}, got {len(self.episode_rewards)}."
            )

        hit_penalty = 50
        goal_reward = 25
        nearmiss_penalty = 50
        too_fast = 10
        braking_penalty = 1

        reward = 0
        goal = False
        current_observation = self.episode_observations[-1]
        ego_vehicle_speed = current_observation[2] * 3.6

        ego_vehicle_goal_distance = np.linalg.norm(
            [current_observation[0] - self.episode_ego_vehicle_goal_position[0], 
             current_observation[1] - self.episode_ego_vehicle_goal_position[1]]
        )

        # penalize distance to pedestrian linearly
        if self.client_world.is_pedestrian_observable and ego_vehicle_speed > 1.0:
            ego_vehicle_pedestrian_distance = np.linalg.norm(
                [current_observation[0] - current_observation[4], 
                current_observation[1] - current_observation[5]]
            )
            if ego_vehicle_pedestrian_distance < 3.0:
                reward = -0.1 - (3.0 - ego_vehicle_pedestrian_distance)

        # except for first state
        if len(self.episode_actions) > 1:
            # "heavily" penalize braking if you are already standing still
            if self.episode_actions[-1] is not Action.ACCELERATE and self.episode_ego_vehicle_speeds[-1] < 0.28:
                reward -= braking_penalty

        # limit maximum ego vehicle speed to 50 km/h == 13.88 m/s
        if self.episode_actions[-1] is Action.ACCELERATE and self.episode_ego_vehicle_speeds[-1] > Config.Carla.MAX_SPEED:
            reward -= too_fast

        if ego_vehicle_goal_distance <= Config.Carla.GOAL_TOLERANCE:
            reward += goal_reward
            goal = True

        hit = self.in_rectangle(
            current_observation[0], current_observation[1], 
            current_observation[3], 
            current_observation[4], current_observation[5],
            front_margin=0, side_margin=0, back_margin=0
        )
        hit = self.client_world.is_ego_vehicle_in_collision or hit #carla can be buggy sometimes
        
        if hit:
            reward -= hit_penalty
            goal = False

        too_close = self.in_rectangle(
            current_observation[0], current_observation[1], 
            current_observation[3], 
            current_observation[4], current_observation[5],
            front_margin=2, side_margin=0.5, back_margin=0.5
        )
        
        if too_close and not hit:
            reward -= nearmiss_penalty

        nearmiss = self.in_rectangle(
            current_observation[0], current_observation[1], 
            current_observation[3], 
            current_observation[4], current_observation[5],
            front_margin=1.5, side_margin=0.5, back_margin=0.5
        )

        # penalize low velocities (required for IS-DESPOT to work...)
        reward += (0.5 * (ego_vehicle_speed - 50.0) / 50.0) / 100.0

        if not goal: reward -= 0.1 # for not reaching the goal
        # Normalize reward
        reward = reward / 2000.0 # reward scaling for gradients

        self.is_terminal_state = hit or goal
        self.episode_rewards.append(reward)
        step_summary = {
            "reward": reward,
            "goal": goal,
            "collision": hit,
            "nearmiss": nearmiss,
            "terminal": self.is_terminal_state,
            "ped_observable": self.client_world.is_pedestrian_observable
        }
        log_debug(step_summary)
        return step_summary


    def get_reward_akash(self, step_counter: int):
        # episode actions
        if not isinstance(self.episode_actions, list):
            raise TypeError(
                f"Invalid episode actions buffer type: Expected 'list', got '{type(self.episode_actions)}'."
            )
        if not isinstance(self.episode_actions[-1], Action):
            raise TypeError(
                f"Invalid previous action type: Expected '{Action}', got '{type(self.episode_actions[-1])}'."
            )
        # + 1 because of first dummy action (maintain) and current action is needed for calculating reward
        if len(self.episode_actions) != step_counter + 1:
            raise ValueError(
                f"Invalid number of episode actions: Expected {step_counter + 1}, got {len(self.episode_actions)}."
            )
        
        # episode car speeds
        if not isinstance(self.episode_ego_vehicle_speeds, list):
            raise TypeError(
                f"Invalid episode car speeds buffer type: Expected 'list', got '{type(self.episode_ego_vehicle_speeds)}'."
            )
        if not isinstance(self.episode_ego_vehicle_speeds[-1], float):
            raise TypeError(
                f"Invalid previous car speed type: Expected '{float}', got '{type(self.episode_ego_vehicle_speeds[-1])}'."
            )
        # because self.get_current_observation() must be called before self.get_reward()
        if len(self.episode_ego_vehicle_speeds) != step_counter:
            raise ValueError(
                f"Invalid numer of episode ego vehicle speeds: "
                f"Expected {step_counter}, got {len(self.episode_ego_vehicle_speeds)} instead."
            )    
        
        # episode carla vehicle controls
        if not isinstance(self.episode_controls, list):
            raise TypeError(
                f"Invalid episode vehicle controls buffer type: Expected 'list', got '{type(self.episode_controls)}'."
            )
        if not isinstance(self.episode_controls[-1], type(carla.VehicleControl())):
            raise TypeError(
                f"Invalid previous vehicle control type: "
                f"Expected '{carla.VehicleControl()}', got '{type(self.episode_controls[-1])}'."
            )      
        # because control will be set before calling reward function
        if len(self.episode_controls) != step_counter:
            raise ValueError(
                f"Invalid numer of episode vehicle control objects: "
                f"Expected {step_counter}, got {len(self.episode_controls)} instead."
            )
        
        # because of first dummy reward (0.0) 
        if len(self.episode_rewards) != step_counter:
            raise ValueError(
                f"Invalid number of episode rewards: Expected {step_counter}, got {len(self.episode_rewards)}."
            )
        
        max_speed = 50  # in kmph
        hit_penalty = 100
        near_miss_penalty = 10
        goal_reward = 200
        braking_penalty = 1
        over_speeding_penalty = 10

        goal = False
        reward = 0
        current_observation = self.episode_observations[-1]
        ego_vehicle_speed = current_observation[2] * 3.6

        ego_vehicle_goal_distance = np.linalg.norm(
            [current_observation[0] - self.episode_ego_vehicle_goal_position[0], 
             current_observation[1] - self.episode_ego_vehicle_goal_position[1]]
        )

        if ego_vehicle_speed > 1.0:
            if ego_vehicle_speed <= 20:
                ped_hit = self.in_rectangle(
                    current_observation[0], current_observation[1], 
                    current_observation[3], 
                    current_observation[4], current_observation[5],
                    front_margin=1, side_margin=0.5
                )
            else:
                ped_hit = self.in_rectangle(
                    current_observation[0], current_observation[1], 
                    current_observation[3], 
                    current_observation[4], current_observation[5],
                    front_margin=2, side_margin=0.5)
            if ped_hit:
                # scale penalty by impact speed
                scaling = self.linmap(0, max_speed, 0, 1, min(ego_vehicle_speed, max_speed))  # in kmph
                # Different penalties for near miss and actual collisision.
                if self.is_ego_vehicle_in_collision:
                    collision_reward = hit_penalty * (scaling + 0.1)
                else:
                    collision_reward = near_miss_penalty * (scaling + 0.1)
                reward -= collision_reward

        #TODO: How are these numbers calculated?
        # reward = -goal_dist / 1000
        reward -= pow(ego_vehicle_goal_distance / 4935.0, 0.8) * 1.2

        # All grid positions of incoming_car in player rectangle
        # Cost of collision with obstacles
        grid = self.grid_cost.copy()
        exo_vehicle_positions = []
        # scenarios with parked or incoming cars
        if self.scenario_id in [3, 7, 8, 10, 11, 12]:
            # each of the above scenarios has at least one parked car
            for parked_car_index in range(len(self.client_world.parked_exo_vehicles)):
                exo_vehicle_positions.append(
                    (self.client_world.parked_exo_vehicles[parked_car_index].get_location().x, 
                     self.client_world.parked_exo_vehicles[parked_car_index].get_location().y)
                )
            # scenarios with incoming car
            if self.scenario_id in [10, 11, 12]:
                exo_vehicle_positions.append(
                    (self.client_world.incoming_exo_vehicle.get_location().x, 
                     self.client_world.incoming_exo_vehicle.get_location().y)
                )
            for (car_x, car_y) in exo_vehicle_positions:
                # calculate hitbox of exo ego vehicle
                xmin = round(car_x - Config.Carla.EGO_VEHICLE_WIDTH / 2)
                xmax = round(car_x + Config.Carla.EGO_VEHICLE_WIDTH / 2)
                ymin = round(car_y - Config.Carla.EGO_VEHICLE_LENGTH / 2)
                ymax = round(car_y + Config.Carla.EGO_VEHICLE_LENGTH / 2)
                # all grid positions of incoming car
                for x in range(xmin, xmax):
                    for y in range(ymin, ymax):
                        grid[round(x), round(y)] = 100

        # cost of occupying road/non-road tile
        # Penalizing for hitting an obstacle
        location = [min(round(current_observation[0] - self.min_x), self.grid_cost.shape[0] - 1),
                    min(round(current_observation[1] - self.min_y), self.grid_cost.shape[1] - 1)]
        obstacle_cost = grid[location[0], location[1]]
        if obstacle_cost <= 100:
            reward -= (obstacle_cost / 20.0)
        elif obstacle_cost <= 150:
            reward -= (obstacle_cost / 15.0)
        elif obstacle_cost <= 200:
            reward -= (obstacle_cost / 10.0)
        else:
            reward -= (obstacle_cost / 0.22)

        # except for first state
        if len(self.episode_actions) > 1:
            # "heavily" penalize braking if you are already standing still
            if self.episode_actions[-1] is not Action.ACCELERATE and self.episode_ego_vehicle_speeds[-1] < 0.28:
                reward -= braking_penalty

        # limit maximum ego vehicle speed to 50 km/h == 13.88 m/s
        if self.episode_actions[-1] is Action.ACCELERATE and self.episode_ego_vehicle_speeds[-1] > Config.Carla.MAX_SPEED:
            reward -= over_speeding_penalty
        
        # if at least two actions have already been executed
        if len(self.episode_actions) > 1:
            if self.episode_actions[-1] is not Action.MAINTAIN:
                # penalize indecisive behaviour (if the last two actions are different)
                if self.episode_actions[-2] is not self.episode_actions[-1]:
                    reward -= 0.05

        reward -= pow(abs(self.episode_controls[-1].steer), 1.3) / 2.0

        if ego_vehicle_goal_distance < 3:
            reward += goal_reward
            goal = True

        # Normalize reward
        reward = reward / 1000.0

        collision = self.is_ego_vehicle_in_collision or obstacle_cost > 50.0

        self.is_terminal_state = collision or goal

        nearmiss = self.in_rectangle(
            current_observation[0], current_observation[1], 
            current_observation[3], 
            current_observation[4], current_observation[5],
            front_margin=1.5, side_margin=0.5, back_margin=0.5)
        
        self.episode_rewards.append(reward)

        step_summary = {
            "reward": reward,
            "goal": goal,
            "collision": collision,
            "nearmiss": nearmiss,
            "terminal": self.is_terminal_state,
            "ped_observable": self.client_world.is_pedestrian_observable
        }
        log_debug(step_summary)
        return step_summary


    # ============================================================================================== #
    #                                       PATH FINDING METHODS                                     #
    # ============================================================================================== #
    # path predicition with hard coding incoming car path prediction
    def get_path_simple(self, start, end):
        obstacles = self.client_world.get_obstacles()

        self.vehicle = self.client_world.ego_vehicle
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        end = self.episode_ego_vehicle_goal_position
        
        if Config.RISK_AWARE_PATH:
            updated_risk_cmp = np.copy(self.risk_cmp)
            for pos in obstacles:
                pos = (round(pos[0]), round(pos[1]))
                updated_risk_cmp[pos[0] + 10, pos[1] + 10] = 10000
            
        if Config.PREDICT_PEDESTRIAN_PATH and \
           len(self.episode_pedestrian_past_trajectory) >= 15 and \
           self.client_world.is_pedestrian_observable:
                self.step_pedestrian_future_trajectory = self.pedestrian_path_predictor.get_single_prediction(
                    np.array(self.episode_pedestrian_past_trajectory[-15:]), 
                    np.array(self.episode_ego_vehicle_past_trajectory[-15:])
                )
                mode_pos = self.episode_pedestrian_past_trajectory[-1]
                for node in self.step_pedestrian_future_trajectory:
                    world_x = float(mode_pos[0]) + float(node[0])
                    world_y = float(mode_pos[1]) + float(node[1])
                    grid_x = round(world_x - self.path_planner.min_x)
                    grid_y = round(world_y - self.path_planner.min_y)
                    grid_x = min(grid_x, self.grid_cost.shape[0] - 1)
                    grid_y = min(grid_y, self.grid_cost.shape[1] - 1)
                    self.grid_cost[grid_x, grid_y] += 10000

        if self.scenario_id == 11:
            self.grid_cost[9:16, 13:] = 10000
            self.risk_cmp[10:13, 13:] = 10000
            x = round(start[0])
            y = round(start[1])
            # hard coding incoming car path prediction
            obstacles.append((x, y - 1))
            obstacles.append((x, y - 2))
            obstacles.append((x, y - 3))
            obstacles.append((x, y - 4))
            obstacles.append((x, y - 5))
            # all grid locations occupied by car added to obstacles
            for i in [-1, 0, 1]:
                for j in [-2, -1, 0, 1, 2]:
                    obstacles.append((x + i, y + j))

        if self.scenario_id in [1, 10] and \
           self.client_world.pedestrian.get_location().y > start[1] and \
           start[0] >= 2.5:
            end = (end[0], start[1] - 6, end[2])

        if Config.RISK_AWARE_PATH:
            path, risk = self.risk_path_planner.find_path_with_risk(
                start, end, self.grid_cost, obstacles, velocity_kmh(self.vehicle), start[2], updated_risk_cmp, True, self.scenario_id
            )
        else:
            path = self.find_path(start, end, self.grid_cost, obstacles)
            risk = []
            
        # reset costmap
        self.grid_cost = np.ones((110, 310)) * 1000.0
        # Road Network
        self.grid_cost[7:13, 13:] = 1.0
        self.grid_cost[97:103, 13:] = 1.0
        self.grid_cost[7:, 7:13] = 1.0
        # Sidewalk Network
        self.grid_cost[4:7, 4:] = 50.0
        self.grid_cost[:, 4:7] = 50.0
        self.grid_cost[13:16, 13:] = 50.0
        self.grid_cost[94:97, 13:] = 50.0
        self.grid_cost[103:106, 13:] = 50.0
        self.grid_cost[13:16, 16:94] = 50.0

        return (path, risk)


    # wrapper function for HybridA* path planner
    # defines different starting points depending on scenario
    def find_path(self, start, end, costmap, obstacles):
        checkpoint = (92, 14, -90)
        # path computation is done in one go
        if self.scenario_id != 9 or start[1] <= checkpoint[1]:
            t = time.time()
            paths = self.path_planner.find_path(start, end, costmap, obstacles)
            if len(paths):
                path = paths[0]
            else:
                path = []
            path.reverse()
        else:
            # path computation is split into two parts: start -> checkpoint -> end
            path_segemnt_1 = self.path_planner.find_path(start, checkpoint, costmap, obstacles)[0]
            path_segemnt_2 = self.path_planner.find_path(checkpoint, end, costmap, obstacles)[0]
            path_segemnt_2.reverse()
            path_segemnt_1.reverse()
            path = path_segemnt_1[:-1] + path_segemnt_2[1:]

        return path
    

    # ============================================================================================== #
    #                                          UTIL METHODS                                          #
    # ============================================================================================== #
    # define hitbox of the car; used for reward calculation
    def in_rectangle(self, x, y, degrees, ped_x, ped_y, front_margin=1.5, side_margin=0.5, back_margin=0.5):
        # pedestrian position is none if the pedestrian is not observable,
        # i.e. outside a circle with a 50m radius centered around the agent
        if ped_x is None or ped_y is None:
            return False
        radians = degrees_to_radians(degrees)

        # TOP LEFT VERTEX:
        top_left_x = x + ((side_margin + Config.Carla.EGO_VEHICLE_WIDTH / 2) * np.sin(radians)) + \
                         ((front_margin + Config.Carla.EGO_VEHICLE_LENGTH / 2) * np.cos(radians))
        top_left_y = y - ((side_margin + Config.Carla.EGO_VEHICLE_WIDTH / 2) * np.cos(radians)) + \
                         ((front_margin + Config.Carla.EGO_VEHICLE_LENGTH / 2) * np.sin(radians))

        # TOP RIGHT VERTEX:
        top_right_x = x - ((side_margin + Config.Carla.EGO_VEHICLE_WIDTH / 2) * np.sin(radians)) + \
                          ((front_margin + Config.Carla.EGO_VEHICLE_LENGTH / 2) * np.cos(radians))
        top_right_y = y + ((side_margin + Config.Carla.EGO_VEHICLE_WIDTH / 2) * np.cos(radians)) + \
                          ((front_margin + Config.Carla.EGO_VEHICLE_LENGTH / 2) * np.sin(radians))

        # BOTTOM LEFT VERTEX:
        bot_left_x = x + ((side_margin + Config.Carla.EGO_VEHICLE_WIDTH / 2) * np.sin(radians)) - \
                         ((back_margin + Config.Carla.EGO_VEHICLE_LENGTH / 2) * np.cos(radians))
        bot_left_y = y - ((side_margin + Config.Carla.EGO_VEHICLE_WIDTH / 2) * np.cos(radians)) - \
                         ((back_margin + Config.Carla.EGO_VEHICLE_LENGTH / 2) * np.sin(radians))
        
        # BOTTOM RIGHT VERTEX:
        bot_right_x = x - ((side_margin + Config.Carla.EGO_VEHICLE_WIDTH / 2) * np.sin(radians)) - \
                          ((back_margin + Config.Carla.EGO_VEHICLE_LENGTH / 2) * np.cos(radians))
        bot_right_y = y + ((side_margin + Config.Carla.EGO_VEHICLE_WIDTH / 2) * np.cos(radians)) - \
                          ((back_margin + Config.Carla.EGO_VEHICLE_LENGTH / 2) * np.sin(radians))

        ab = [top_right_x - top_left_x, top_right_y - top_left_y]
        am = [ped_x - top_left_x, ped_y - top_left_y]
        bc = [bot_right_x - top_right_x, bot_right_y - top_right_y]
        bm = [ped_x - top_right_x, ped_y - top_right_y]

        is_in_rectangle = 0 <= np.dot(ab, am) <= np.dot(ab, ab) and 0 <= np.dot(bc, bm) <= np.dot(bc, bc)

        log_debug(
            f"ego vehicle corner positions:\n"
            f"\t- center: ({x:.2f},{y:.2f})\n"
            f"\t- theta: {degrees:.2f}\n"
            f"\t- front-margin: {front_margin:.2f}, side-margin: {side_margin:.2f}, back-margin: {back_margin:.2f}\n"
            f"\t- top-left & top-right: ({top_left_x:.2f},{top_left_y:.2f}) & ({top_right_x:.2f},{top_right_y:.2f})\n"
            f"\t- bottom-left & bottom-right: ({bot_left_x:.2f},{bot_left_y:.2f}) & ({bot_right_x:.2f},{bot_right_y:.2f})\n"
            f"pedestrian position:\n"
            f"\t- (x,y): ({ped_x:.2f},{ped_y:.2f})\n"
            f"is_in_rectangle: {'TRUE' if is_in_rectangle else 'FALSE'}"
        )

        return is_in_rectangle


    def linmap(self, a, b, c, d, x):
        return (x - a) / (b - a) * (d - c) + c
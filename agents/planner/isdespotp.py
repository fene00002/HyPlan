from multiprocessing import Process
from typing import List, Tuple

import carla
import psutil

from agents.learner.rlagent import RLAgent
from utils.config import Config, Action
from utils.connector import DespotBridge, Connection
from utils.utils import run_despot
from utils.logger import log_debug, log_info


class ISDespotP(RLAgent):
    def __init__(self, client: carla.Client, client_world):
        super(ISDespotP, self).__init__(client, client_world)

        self.episode_despot_policies: List[Tuple[float, float, float]] = []

        # IS-DESPOT C++ process
        self.despot_process = None
        self.start_despot_process()

        # general control connection
        self.despot_connection = DespotBridge()
        self.establish_control_connection()


    # ================================================================================================= #
    #                                        IS-DESPOT SETUP METHODS                                    #
    # ================================================================================================= #
    def start_despot_process(self):
        if self.despot_process is not None:
            raise ValueError("Existing despot process must first be terminated before new invocation.")
        log_info(f"starting {Config.Despot.VARIANT} C++ process...")

        self.despot_process = Process(target=run_despot)      
        self.despot_process.start()


    def kill_despot_process(self):
        if self.despot_process == None:
            raise ValueError("No despot process to terminate.")
        if not isinstance(self.despot_process, Process):
            raise TypeError("Not a process.")
        log_info(f"killing {Config.Despot.VARIANT} C++ process...")

        parent_process = psutil.Process(self.despot_process.pid)
        for child_process in parent_process.children(recursive=True):
            log_info(f"killing C++ process {child_process.pid}")
            child_process.kill()
        parent_process.kill()
        log_info(f"killing python process {parent_process.pid}")
        self.despot_process = None


    def establish_control_connection(self):
        if self.despot_connection is None:
            raise TypeError("IS-DESPOT C++ process interface has not been initialized.")
        if not isinstance(self.despot_connection, DespotBridge):
            raise TypeError(f"Invalid IS-DESPOT C++ prcess interface: " 
                            f"Expected '{DespotBridge}', got '{type(self.despot_connection)}'.")
        log_info(f"setting up TCP connection for communication with {Config.Despot.VARIANT}...")
        
        # general control connection
        self.despot_connection.establish_connection(Connection.DESPOT)


    def close_connections(self):
        if self.despot_connection is None:
            raise TypeError("IS-DESPOT C++ process interface has not been initialized.")
        if not isinstance(self.despot_connection, DespotBridge):
            raise TypeError(f"Invalid IS-DESPOT C++ prcess interface: " 
                            f"Expected '{DespotBridge}', got {type(self.despot_connection)}.")

        self.despot_connection.close_connections()
        self.despot_connection = None


    # ================================================================================================= #
    #                                       IS-DESPOT CORE METHODS                                      #
    # ================================================================================================= #
    def get_speed_action(self, step_counter: int):
        if not isinstance(self.episode_rewards, list):
            raise TypeError(
                f"Invalid episode rewards buffer type: Expected 'list', got '{type(self.episode_rewards)}'."
            )
        # at this point we do not yet have calculated the reward of the current observation
        # because we first need IS-DESPOT's action to do so
        if len(self.episode_rewards) != step_counter:
            raise ValueError(
                f"Invalid number of episode rewards: Expected {step_counter}, got {len(self.episode_rewards)}."
            )
        if not isinstance(self.episode_rewards[-1], float):
            raise TypeError(
                f"Invalid previous reward type: Expected '{float}', got '{type(self.episode_rewards[-1])}'."
            )
        
        if not isinstance(self.episode_observations, list):
            raise TypeError(
                f"Invalid episode observations buffer type: Expected 'list', got '{type(self.episode_observations)}'."
            )
        if len(self.episode_observations) != step_counter:
            raise ValueError(
                f"Invalid number of episode observations: Expected {step_counter}, got {len(self.episode_observations)}."
            )
        if not isinstance(self.episode_observations[-1], tuple):
            raise TypeError(
                f"Invalid previous observation type: Expected '{tuple}', got '{type(self.episode_observations[-1])}'."
            )
            
        # reward of previous observation set by super(ISDespotP, self).get_reward()
        # or dummy reward in case of first step of episode
        previous_reward = self.episode_rewards[-1]
        
        # current observation set by super(ISDespotP, self).get_observation()
        current_observation = self.episode_observations[-1]

        self.despot_connection.send_observation(
            terminal=self.is_terminal_state,
            # IS-DESPOT needs the previous reward because it is required by HyLEAP's NN
            reward=previous_reward,
            car_position=[current_observation[0], current_observation[1]],
            car_speed=current_observation[2],
            angle=current_observation[3], 
            car_path=self.step_ego_vehicle_future_trajectory,
            pedestrian_visibility=self.client_world.is_pedestrian_observable,
            pedestrian_position=[current_observation[4], current_observation[5]],
            pedestrian_path=None if len(self.step_pedestrian_future_trajectory) == 0 else self.step_pedestrian_future_trajectory
        )
        
        # query IS-DESPOT for action
        return self.despot_connection.receive_despot_simulation_result()


    def run_step(self, step_counter: int):
        self.vehicle = self.client_world.ego_vehicle
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        end = self.episode_ego_vehicle_goal_position

        (self.step_ego_vehicle_future_trajectory, risk) = super(ISDespotP, self).get_path_simple(start, end)

        agent_vehicle_control = carla.VehicleControl(
            throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False, manual_gear_shift=False, gear=0
        )

        # buggy path planner returning empty path means we skip this step and just maintain current velocity
        if not len(self.step_ego_vehicle_future_trajectory):
            # empty path means no path that is collision-free, thus decellerating is the only save action
            # agent_vehicle_control.brake = 0.6
            # obtain partial step summary without remembering observation corresponding to this step
            step_summary = super(ISDespotP, self).get_current_observation(step_counter, skipped_step=True)
            step_summary["skipped_step"] = True
            step_summary["control"] = agent_vehicle_control
            return step_summary

        agent_vehicle_control.steer = \
            (self.step_ego_vehicle_future_trajectory[2][2] - start[2]) / Config.Carla.MAX_STEERING_ANGLE
        
        # only update episode buffers, when we do not skip this step
        super(ISDespotP, self).get_current_observation(step_counter)

        # best speed action for the given path
        despot_action, despot_value, despot_policy = self.get_speed_action(step_counter)

        # translate received action into CARLA car control commands
        if despot_action is Action.DECELERATE:
            agent_vehicle_control.brake = 0.6
        elif despot_action is Action.ACCELERATE:
            agent_vehicle_control.throttle = 0.6

        # remember episode
        self.episode_actions.append(despot_action)
        self.episode_controls.append(agent_vehicle_control)
        self.episode_despot_policies.append(despot_policy)

        step_summary = super(ISDespotP, self).get_reward_despot(step_counter)

        step_summary["skipped_step"] = False
        step_summary["control"] = agent_vehicle_control
        step_summary["action"] = despot_action

        return step_summary 


    def initialize_episode(self, episode_counter: int, scenario: Tuple):
        self.episode_despot_policies.clear()
        return super().initialize_episode(episode_counter, scenario)
    

    def finalize_episode(self, episode_counter: int, non_conclusive: bool, step_counter: int):
        # this function is only intended for clean-up at the moment, 
        # i.e. if we have to synchronize Python & C++ after forcefully terminating an episode early
        if not non_conclusive: return
    
        previous_observation = self.episode_observations[-1]
        # terminate episode in C++ by sending the previous observation again but with terminal flag
        self.despot_connection.send_observation(
            terminal=True,
            reward=self.episode_rewards[-1],
            car_position=[previous_observation[0], previous_observation[1]],
            car_speed=previous_observation[2],
            angle=previous_observation[3], 
            car_path=self.step_ego_vehicle_future_trajectory,
            pedestrian_visibility=self.client_world.is_pedestrian_observable,
            pedestrian_position=[previous_observation[4], previous_observation[5]],
            abort=True # terminate episode in IS-DESPOT without running another planning step
        )
            
        # receive & throw away IS-DESPOT's simulation results
        _, _, _ = self.despot_connection.receive_despot_simulation_result()

        # IS-DESPOT is now waiting for new episode to start
        return super().finalize_episode(episode_counter, non_conclusive, step_counter)
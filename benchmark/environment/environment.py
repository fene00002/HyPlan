import random

import carla
import pygame
import numpy as np
import torch

from benchmark.environment.hud import HUD
from benchmark.environment.world import ClientWorld
from benchmark.environment.scenario import Scenario
from utils.config import Config, Action
from utils.logger import log_debug, log_info


class CarlaCTS02(object):

    def __init__(self):
        super(CarlaCTS02, self).__init__()

        # connect to carla simulator
        self.client = carla.Client(Config.Carla.HOST.value, Config.Carla.PORT)
        self.client.set_timeout(100.0 if Config.Carla.REMOTE else 10.0) # seconds
        log_info(f"CARLA client version: {self.client.get_client_version()}")
        log_info(f"CARLA server version: {self.client.get_server_version()}")

        # the actual CARLA simulator handle
        self.server_world = self.client.get_world()

        # the map used for the benchmark
        self.client.load_world('Town01_Opt')
        log_info(f"current map: {self.server_world.get_map().name}")

        # define simulation settings
        settings = self.server_world.get_settings()
        settings.fixed_delta_seconds = Config.Carla.SIMULATION_STEP
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
        settings.synchronous_mode = Config.Carla.SYNCHRONONOUS
        self.server_world.apply_settings(settings)

        # used for rendering scene simulation in CARLA
        self.display = None
        pygame.display.init()
        pygame.font.init()
        self.clock = pygame.time.Clock()

        # create client side control of the simulation
        self.client_world = ClientWorld(self.server_world)

        # make runs reproducible: https://pytorch.org/docs/stable/notes/randomness.html
        # random is used in scenario.py and world.py
        random.seed(Config.SEED)
        # model weight init
        torch.manual_seed(Config.SEED)
        # any operation invloving numpy
        np.random.seed(Config.SEED)

        # agent for the entire duration of script execution
        self.agent = None
        # whether to train, validate or test agent
        self.mode = None
        self.train_episodes = None
        self.test_episodes = None
        self.number_of_episode = None
        # define the current scene/episode
        self.scenario_id = None
        self.pedestrian_velocity = None
        self.pedestrian_crossing_distance = None
        # handle for creating scenes/episodes
        self.scene_generator = Scenario(self.server_world)
        # the carla.VehicleControl object set by the agent for selecting a velocity
        self.ego_vehicle_control = None
    
        self.get_all_parked_vehicle_transforms_across_all_scenarios()
        self.client_world.set_parked_vehicle_transforms_across_all_scenarios(self.all_parked_vehicle_transforms)


    # ============================================================================================== #
    #                                         SETUP METHODS                                          #
    # ============================================================================================== #
    def set_agent(self, agent):
        self.agent = agent


    # define episodes over which a given algorithm is to be trained 
    def prepare_train_episodes(self):
        log_info("preparing train data...")
        self.mode = "TRAINING"
        episodes = list()
        log_info(f"scenarios in use: {Config.TRAIN_SCENARIOS}")

        # number of episode depends on the step size of the range of allowed pedestrian speeds & distance
        speed_step = 0.1
        ped_speeds = np.arange(
            Config.TRAIN_PED_SPEED_RANGE[0], Config.TRAIN_PED_SPEED_RANGE[1] + 0.1, speed_step
        ).round(2)
        distance_step = 1
        ped_distances = np.arange(
            Config.TRAIN_PED_DIST_RANGE[0], Config.TRAIN_PED_DIST_RANGE[1] + 1, distance_step
        ).round(2)
        
        log_info(
            f"pedestrian speeds in use (m/s): {ped_speeds[0]:.2f}-{ped_speeds[-1]:.2f} (in steps of {speed_step})\n"
            f"pedestrian distances in use (m): {ped_distances[0]:.2f}-{ped_distances[-1]:.2f} (in steps of {distance_step})"
        )
        
        # for each epoch
        for epoch in range(Config.EPOCHS):
            # create episodes for each scenario
            for scenario in Config.TRAIN_SCENARIOS:
                    # generate episodes of a given scenario with differing pedestrian speeds and distances
                    for speed in ped_speeds:
                        for distance in ped_distances:
                            episodes.append((scenario, speed, distance))

        self.number_of_episode = len(episodes)
        # randomly shufflöe episodes to prevent a NN from forgetting the first few scenarios
        random.shuffle(episodes)
        self.train_episodes = iter(episodes[Config.INITIAL_EPISODE-1:])


    # define episodes over which a given algorithm is to be evaluated 
    def prepare_validation_episodes(self):
        log_info("preparing validation data...")
        self.mode = "TESTING"
        episodes = list()
        log_info(f"scenarios in use: {Config.VAL_SCENARIOS}")

        # number of episode depends on the step size of the range of allowed pedestrian speeds & distance
        speed_step = 0.1
        ped_speeds = np.concatenate((
            np.arange(Config.VAL_PED_SPEED_RANGE[0][0], Config.VAL_PED_SPEED_RANGE[0][1] + 0.1, speed_step),
            np.arange(Config.VAL_PED_SPEED_RANGE[1][0], Config.VAL_PED_SPEED_RANGE[1][1] + 0.1, speed_step)
        ))  
        distance_step = 1
        ped_distances = np.arange(
            Config.VAL_PED_DIST_RANGE[0], Config.VAL_PED_DIST_RANGE[1] + 1, distance_step
        )  
        log_info(
            f"pedestrian speeds in use (m/s): {Config.VAL_PED_SPEED_RANGE[0][0]:.2f}-{Config.VAL_PED_SPEED_RANGE[0][1]:.2f} "
            f"and {Config.VAL_PED_SPEED_RANGE[1][0]:.2f}-{Config.VAL_PED_SPEED_RANGE[1][1]:.2f}  (in steps of {speed_step}) \n"
            f"pedestrian distances in use (m): {ped_distances[0]:.2f}-{ped_distances[-1]:.2f} (in steps of {distance_step})"
        )

        # create episodes for each scenario
        for scenario in Config.VAL_SCENARIOS:
            # generate episodes of a given scenario with differing pedestrian speeds and distances
            for speed in ped_speeds:
                for distance in ped_distances:
                    episodes.append((scenario, speed, distance))

        self.number_of_episode = len(episodes)
        self.test_episodes = iter(episodes)


    # define episodes over which a given algorithm is to be evaluated 
    def prepare_test_episodes(self):
        log_info("preparing test data...")
        self.mode = "TESTING"
        episodes = list()
        log_info(f"scenarios in use: {Config.TEST_SCENARIOS}")

        # number of episode depends on the step size of the range of allowed pedestrian speeds & distance
        ped_speed_step = 0.1
        ped_speeds = np.arange(Config.TEST_PED_SPEED_RANGE[0], Config.TEST_PED_SPEED_RANGE[1] + 0.1, ped_speed_step)
        ped_speeds.round(decimals=2)
        ped_distance_step = 1
        ped_distances = np.arange(Config.TEST_PED_DIST_RANGE[0], Config.TEST_PED_DIST_RANGE[1] + 1, ped_distance_step)
        ped_distances.round(decimals=2)

        # scenarios 11, 12 with car
        car_speed_step = 0.1
        car_speeds = np.arange(Config.TEST_CAR_SPEED_RANGE[0], Config.TEST_CAR_SPEED_RANGE[1] + 0.1, car_speed_step)
        car_speeds.round(decimals=2)
        log_info(
            f"pedestrian speeds in use (m/s): {ped_speeds[0]:.2f}-{ped_speeds[-1]:.2f} (in steps of {ped_speed_step}, total: {len(ped_speeds)})\n"
            f"pedestrian distances in use (m): {ped_distances[0]:.2f}-{ped_distances[-1]:.2f} (in steps of {ped_distance_step}, total: {len(ped_distances)})\n"
            f"car speeds in use (m/s): {car_speeds[0]:.2f}-{car_speeds[-1]:.2f} (in steps of {car_speed_step}, total: {len(car_speeds)})"
        )
    
        # create episodes for each scenario
        for scenario in Config.TEST_SCENARIOS:
            if scenario in ['11', '12']:
                for speed in car_speeds:
                    episodes.append((scenario, speed, 0))
            else:
                # generate episodes of a given scenario with differing pedestrian speeds and distances
                for speed in ped_speeds:
                    for distance in ped_distances:
                        episodes.append((scenario, speed, distance))
                        
        self.number_of_episode = len(episodes)
        self.test_episodes = iter(episodes[Config.INITIAL_EPISODE-1:])


    # TODO: refactor
    # get all parked cars across all scenarios
    # we do this to avoid having to infer the cv2.polygon during car intention image generation
    # for parked cars each time we change episode or increment steps
    # -> this saves time and allows for parallelization through worker processes
    def get_all_parked_vehicle_transforms_across_all_scenarios(self):
        self.all_parked_vehicle_transforms = {}
        for index, function in enumerate([function for function in dir(self.scene_generator) if not function.startswith("__")], 1):
            # single parked car scenarios
            if index in [3, 7, 8]:
                _, obstacles, _, _ = getattr(self.scene_generator, function)()
                # index=0 at dim=0 is walker; index=0 at dim=1 is blueprint
                parked_car_transform = obstacles[1][1] # gets the transform of parked car
                self.all_parked_vehicle_transforms.update({index:[parked_car_transform]})
            # scenarios with an incoming and a single parked car
            if index in [11, 12]:
                _, obstacles, _, _ = getattr(self.scene_generator, function)()
                # index=0 at dim=0 is walker; index=1 at dim=0 is incoming car
                parked_car_transform = obstacles[1][1]
                if index == 11:
                    self.all_parked_vehicle_transforms.update({index:[parked_car_transform]})
                if index == 12:
                    parked_car_transforms = []
                    initial_y = parked_car_transform.location.y
                    for parked_car_index in range(12):
                        parked_car_transform = carla.Transform(
                            carla.Location(
                                x=parked_car_transform.location.x,
                                y=initial_y + parked_car_index*7,
                                z=parked_car_transform.location.z
                            ),
                            carla.Rotation(pitch=parked_car_transform.rotation.pitch,
                                            yaw=parked_car_transform.rotation.yaw,
                                            roll=parked_car_transform.rotation.roll
                            )
                        )
                        parked_car_transforms.append(parked_car_transform)
                    self.all_parked_vehicle_transforms.update({index:parked_car_transforms})
        if len(self.all_parked_vehicle_transforms) != 5:
            raise ValueError(
                f"Invalid number of scenarios with parked cars: Expected 5, got {len(self.all_parked_vehicle_transforms)}"
            )
        

    # ============================================================================================== #
    #                                          CORE METHODS                                          #
    # ============================================================================================== #
    def next_episode(self):
        if self.mode == "TRAINING":
            return next(self.train_episodes)
        elif self.mode == "TESTING":
            return next(self.test_episodes)


    # reset world for each new episode
    def initialize_episode(self, episode_counter: int):
        
        self.scenario_id, self.pedestrian_velocity, self.pedestrian_crossing_distance = self.next_episode()
        log_info(
            f"SCENARIO [{self.scenario_id}]: (PEDESTRIAN VELOCITY: {self.pedestrian_velocity:.2f} & "
            f"PEDESTRIAN CROSSING DISTANCE: {self.pedestrian_crossing_distance:.2f})"
        )
        func = 'self.scene_generator.scenario' + self.scenario_id
        scenario = eval(func + '()')
        self.client_world.initialize_episode(
            episode_counter, scenario, self.pedestrian_velocity, self.pedestrian_crossing_distance
        )
        self.agent.initialize_episode(episode_counter, scenario)


    def finalize_episode(self, episode_counter: int, non_conclusive: bool, step_counter: int):
        self.client_world.finalize_episode(episode_counter, step_counter)
        self.agent.finalize_episode(episode_counter, non_conclusive, step_counter)


    def step(self, episode_counter: int, step_counter: int):
        step_summary = self.agent.run_step(step_counter)
        step_summary["scenario"] = self.scenario_id
        step_summary["ped_speed"] = float(self.pedestrian_velocity)
        step_summary["ped_distance"] = int(self.pedestrian_crossing_distance)

        # update control 
        self.ego_vehicle_control = step_summary["control"]

        # sanity check for congruent control & action
        if not step_summary["skipped_step"]:
            if self.ego_vehicle_control.throttle > 0.0: action = Action.ACCELERATE
            elif self.ego_vehicle_control.brake > 0.0: action = Action.DECELERATE
            else: action = Action.MAINTAIN
            agent_action = step_summary["action"]
            if agent_action is not action: raise ValueError(
                f"Incongruent environment action: {action} and agent action: {agent_action}"
            )

        # apply ego vehicle control
        self.client_world.ego_vehicle.apply_control(self.ego_vehicle_control)

        # when recording pedestrian data, we have to stop all movement of the ego vehicle
        # when the goal has been reached, so that the pedestrian can cross the road, allowing us to record the crossing
        if Config.RECORD_PEDESTRIAN_DATA and step_summary["goal"]:
            self.client_world.ego_vehicle.set_location(
                carla.Location(x=self.client_world.scenario[2][0], y=self.client_world.scenario[2][1], z=0.1)
            )

        # apply control of all exo agents (pedestrian & vehicles) in the current scene
        self.client_world.tick(self.clock)

        # this ticks the server for a single simulation step
        if Config.Carla.SYNCHRONONOUS: self.server_world.tick()
        else: raise ValueError("Invalid CARLA server simulation mode: Only synchrononous simulation supported.")     

        '''
        if Config.DISPLAY:
            # display the image in an OpenCV display window
            cv2.namedWindow('ImageWindowName', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('ImageWindowName', self.client_world.render_rgb_camera_with_bounding_boxes())
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
        '''
        return step_summary


    # ============================================================================================== #
    #                                         DEBUG METHODS                                          #
    # ============================================================================================== #
    def render(self):
        # create display
        if self.display is None:
            self.display = pygame.display.set_mode(
                (Config.Carla.DISPLAY_SCREEN_WIDTH, Config.Carla.DISPLAY_SCREEN_HEIGHT), 
                pygame.HWSURFACE | pygame.DOUBLEBUF
            )
            self.display.fill((0, 0, 0))
            pygame.display.flip()
        # pass display to rgb camera sensor to populate display with rgb image of current scene simulation step
        self.client_world.render(self.display)
        pygame.display.flip()

        

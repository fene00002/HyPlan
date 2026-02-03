import random
import queue
import re
import math

import cv2
import carla
import numpy as np
import pygame

from benchmark.environment.hud import HUD
from utils.client_bounding_boxes import ClientSideBoundingBoxes
from utils.config import Config
from utils.utils import distance_actors, distance_locations, build_projection_matrix, get_image_point, point_in_canvas, velocity_kmh
from utils.logger import log_debug, log_info


class ClientWorld(object):
    def __init__(self, server_world):
        
        # benchmark
        self.server_world = server_world
        self.map = self.server_world.get_map()
        # interface for displaying user information
        self.hud = HUD()
        self.server_world.on_tick(self.hud.on_world_tick)
        
        # actors in the benchmark
        self.ego_vehicle = None
        self.pedestrian = None
        self.incoming_exo_vehicle = None
        self.parked_exo_vehicles = []
        # also count as actors
        self.rgb_camera = None
        self.collision_sensor = None

        # collision sensor flag
        self.is_ego_vehicle_in_collision = False

        # define a scenario
        self.scenario = None
        self.pedestrian_velocity = None
        self.pedestrian_crossing_distance = None
        self.is_pedestrian_observable = False

        # set weather
        self.weather_presets = self.find_weather_presets()
        self.weather_index = 0
        [self.next_weather() for _ in range(2)]

        # unload unwanted map layers
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]
        #if Config.Carla.REMOTE:
            #self.server_world.unload_map_layer(carla.MapLayer.StreetLights)

        # create a queue to store and retrieve the rgb camera sensor data
        self.rgb_camera_image_queue = queue.Queue()
        # save auxiliary data structure for computing bounding boxed during simulation loop (saves execution time)
        # used for drawing bounding boxes of actors in the scene
        self.edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]
        # calculate the camera projection matrix to project from 3D -> 2D
        self.projection_matrix = build_projection_matrix(
            Config.Carla.DISPLAY_SCREEN_WIDTH, Config.Carla.DISPLAY_SCREEN_HEIGHT, Config.Carla.DISPLAY_FOV
        )
        self.projection_matrix_behind = build_projection_matrix(
            Config.Carla.DISPLAY_SCREEN_WIDTH, 
            Config.Carla.DISPLAY_SCREEN_HEIGHT, 
            Config.Carla.DISPLAY_FOV, 
            is_behind_camera=True
        )

    # ============================================================================================== #
    #                                         SETUP METHODS                                          #
    # ============================================================================================== #
    def find_weather_presets(self):
        rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
        name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
        presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
        return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


    def next_weather(self, reverse=False):
        self.weather_index += -1 if reverse else 1
        self.weather_index %= len(self.weather_presets)
        preset = self.weather_presets[self.weather_index]
        self.server_world.set_weather(preset[0])
    

    def set_parked_vehicle_transforms_across_all_scenarios(self, parked_car_transforms: list):
        self.all_parked_vehicle_transforms = parked_car_transforms


    # wrapper function to ensure that actor has actually been spawned
    def try_spawn_actor(self, actor_blueprint: carla.ActorBlueprint, actor_spwan_point: carla.Location) -> carla.Actor:
        actor = None
        while actor is None:
            actor = self.server_world.try_spawn_actor(
                actor_blueprint, actor_spwan_point
            )
        return actor
    
    # ============================================================================================== #
    #                                          CORE METHODS                                          #
    # ============================================================================================== #
    # called by the collision sensor if ego vehicle hits something
    def collision(self, event):
        # calculate collision intensitiy
        intensity = math.sqrt(event.normal_impulse.x**2 + event.normal_impulse.y**2 + event.normal_impulse.z**2)
        self.is_ego_vehicle_in_collision = True


    # called at the beginning of every scene/episode
    # spawsn all relevant actors (ego/exo vehicle & pedestrian)
    def initialize_episode(self, episode_counter, scenario, pedestrian_velocity, pedestrian_crossing_distance):
        if self.ego_vehicle is not None:
            raise ValueError("Ego vehicle has not been destroyed properly.")
        if self.rgb_camera is not None:
            raise TypeError("RGB camera sensor has not been destroyed properly.")
        if self.collision_sensor is not None:
            raise TypeError("Collision sensor has not been destroyed properly.")
        if not self.map.get_spawn_points(): 
            raise ValueError("There are no spawn points available.")
             
        # information uniquely describing scenario
        self.scenario = scenario
        self.pedestrian_velocity = pedestrian_velocity
        self.pedestrian_crossing_distance = pedestrian_crossing_distance
        
        # ego vehicle spawn point transform
        ego_vehicle_spawn_point = carla.Transform(
            carla.Location(x=scenario[3][0], y=scenario[3][1], z=0.1), # z needs to be positive
            carla.Rotation(pitch=0.0, yaw=scenario[3][2], roll=0.0)
        )
        # ego vehicle blueprint
        blueprint_library = self.server_world.get_blueprint_library()
        ego_vehicle_blueprint = random.choice(blueprint_library.filter(Config.Carla.FILTER))
        ego_vehicle_blueprint.set_attribute('role_name', Config.Carla.AGENT_VEHICLE_ROLENAME)
        ego_vehicle_blueprint.set_attribute(
            'color', ego_vehicle_blueprint.get_attribute('color').recommended_values[1]
        )
        self.ego_vehicle = self.try_spawn_actor(ego_vehicle_blueprint, ego_vehicle_spawn_point)
        # better collision detection
        physics_control = self.ego_vehicle.get_physics_control()
        physics_control.use_sweep_wheel_collision = True
        self.ego_vehicle.apply_physics_control(physics_control)

        # check for correct ego vehicle dimensions
        # TODO: idk why x-axis corresponds to length and y-axis to width of ego vehicle
        if abs(self.ego_vehicle.bounding_box.extent.x * 2 - Config.Carla.EGO_VEHICLE_LENGTH) > 1e-3:
            raise ValueError(
                f"Invalid ego vehicle width: Expected {Config.Carla.EGO_VEHICLE_LENGTH}, "
                f"got {(self.ego_vehicle.bounding_box.extent.x * 2):.4f}"
            )
        if abs(self.ego_vehicle.bounding_box.extent.y * 2 - Config.Carla.EGO_VEHICLE_WIDTH) > 1e-3:
            raise ValueError(
                f"Invalid ego vehicle length: Expected {Config.Carla.EGO_VEHICLE_WIDTH}, "
                f"got {(self.ego_vehicle.bounding_box.extent.y * 2):.4f}"
            )   
        # initial tick for this scenario
        # required to prevent inconsistent ego vehicle locations & speeds
        self.server_world.tick()
        
        # determine when ego vehicle becomes responsive
        # required to prevent "dud" steps, where the ego vehicle is not increasing its speed
        # even when receiving correctly specified carla.VehicleControl objects,
        # which in turn falsifies the training memory of RL agents
        ego_vehicle_starting_position = self.ego_vehicle.get_location()
        travelled_distance = 0.0
        while travelled_distance < 0.1:
            self.ego_vehicle.apply_control(
                carla.VehicleControl(
                    throttle=0.6, steer=0.0, brake=0.0, hand_brake=False, reverse=False, manual_gear_shift=False, gear=0
                )
            )
            self.server_world.tick()
            travelled_distance = distance_locations(ego_vehicle_starting_position, self.ego_vehicle.get_location())
        # reset ego vehicle velocity
        ego_vehicle_velocity = velocity_kmh(self.ego_vehicle)
        while ego_vehicle_velocity > 0.0:
            self.ego_vehicle.apply_control(
                carla.VehicleControl(
                    throttle=0.0, steer=0.0, brake=1.0, hand_brake=False, reverse=False, manual_gear_shift=False, gear=0
                )
            )            
            self.server_world.tick()
            ego_vehicle_velocity = velocity_kmh(self.ego_vehicle)
        # reset to starting position
        self.ego_vehicle.set_location(ego_vehicle_starting_position)

        # set up other agents (pedestrian and exo vehicles)
        obstacles = self.scenario[1]
        # single pedestrian scenarios; pedestrian immediately walks away from ego vehicle
        if self.scenario[0] in [1, 2, 4, 5]:
            self.pedestrian = self.try_spawn_actor(obstacles[0][0], obstacles[0][1])
            self.pedestrian.apply_control(carla.WalkerControl(carla.Vector3D(0, self.pedestrian_velocity, 0), 1))
       
       # single pedestrian scenario; pedestrian immediately walks towards ego vehicle
        if self.scenario[0] == 6:
            self.pedestrian = self.try_spawn_actor(obstacles[0][0], obstacles[0][1])
            self.pedestrian.apply_control(carla.WalkerControl(carla.Vector3D(0, -self.pedestrian_velocity, 0), 1))
       
        # single pedestrian scenario with a single parked vehicle
        # only let pedestrian start walking once ego vehicle is sufficiently close
        if self.scenario[0] in [3, 7, 8]:
            self.pedestrian = self.try_spawn_actor(obstacles[0][0], obstacles[0][1])
            self.parked_exo_vehicles.append(self.server_world.try_spawn_actor(obstacles[1][0], obstacles[1][1]))

        # single pedestrian scenario
        # only let pedestrian start walking once ego vehicle is sufficiently close
        if self.scenario[0] == 9:
            self.pedestrian = self.try_spawn_actor(obstacles[0][0], obstacles[0][1])
            self.pedestrian.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), 1))

        # single pedestrian scenario with a single incoming vehicle; pedestrian immediately walks towards ego vehicle
        if self.scenario[0] == 10:
            self.pedestrian = self.try_spawn_actor(obstacles[0][0], obstacles[0][1])
            self.pedestrian.apply_control(carla.WalkerControl(carla.Vector3D(0, -self.pedestrian_velocity, 0), 1))
            self.incoming_exo_vehicle = self.server_world.try_spawn_actor(obstacles[1][0], obstacles[1][1])

        # scenario with 12 parked and one incoming vehicle
        if self.scenario[0] == 11:
            self.incoming_exo_vehicle = self.try_spawn_actor(obstacles[0][0], obstacles[1][1])
            exo_vehicle_spawn_point = obstacles[1][1]
            for _ in range(12):
                parked_exo_vehicle = self.try_spawn_actor(obstacles[1][0], exo_vehicle_spawn_point)
                self.parked_exo_vehicles.append(parked_exo_vehicle)
                exo_vehicle_spawn_point.location.y += 7

        # scenario with single pedestrian, parked and incoming vehicle
        if self.scenario[0] == 12:
            self.pedestrian = self.try_spawn_actor(obstacles[0][0], obstacles[0][1])
            self.incoming_exo_vehicle = self.try_spawn_actor(obstacles[1][0], obstacles[1][1])
            parked_exo_vehicle = self.try_spawn_actor(obstacles[2][0], obstacles[2][1])
            self.parked_exo_vehicles.append(parked_exo_vehicle)

        # default: ped not visible
        self.is_pedestrian_observable = False
        if not hasattr(self.pedestrian, "bounding_box"):
            raise AttributeError("Invalid pedestrian specification: Missing bounding box dimensions.")
        # check for correct exo pedestrian dimensions
        if abs(self.pedestrian.bounding_box.extent.x * 2 - Config.Carla.EXO_PEDESTRIAN_LENGTH) > 1e-3:
            raise ValueError(
                f"Invalid exo pedestrian width: Expected {Config.Carla.EXO_PEDESTRIAN_LENGTH}, "
                f"got {(self.pedestrian.bounding_box.extent.x * 2):.4f}"
            )
        if abs(self.pedestrian.bounding_box.extent.y * 2 - Config.Carla.EXO_PEDESTRIAN_WIDTH) > 1e-3:
            raise ValueError(
                f"Invalid exo pedestrian length: Expected {Config.Carla.EXO_PEDESTRIAN_WIDTH}, "
                f"got {(self.pedestrian.bounding_box.extent.y * 2):.4f}"
            )  

        # create camera sensor (for rendering scene simulation in CARLA if Config.DISPLAY is enabled)
        # car intention image generation has nothing to do with this (look into utils/carla_birdeye_view for this)
        camera_blueprint = blueprint_library.find('sensor.camera.rgb')
        camera_blueprint.set_attribute('image_size_x', str(Config.Carla.DISPLAY_SCREEN_WIDTH))
        camera_blueprint.set_attribute('image_size_y', str(Config.Carla.DISPLAY_SCREEN_HEIGHT))
        camera_transform = carla.Transform(
            carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)
        )
        self.rgb_camera = self.server_world.spawn_actor(
            camera_blueprint, 
            camera_transform, 
            attach_to=self.ego_vehicle, 
            attachment_type=carla.AttachmentType.SpringArm 
        )
        # needed to correctly draw bounding boxes
        calibration = np.identity(3)
        calibration[0, 2] = Config.Carla.DISPLAY_SCREEN_WIDTH / 2.0
        calibration[1, 2] = Config.Carla.DISPLAY_SCREEN_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = \
            Config.Carla.DISPLAY_SCREEN_WIDTH / (2.0 * np.tan(Config.Carla.DISPLAY_SCREEN_HEIGHT * np.pi / 360.0))
        self.rgb_camera.calibration = calibration

        # publish images into the queue as soon as they become available
        self.rgb_camera.listen(self.rgb_camera_image_queue.put)

        # spawn collision sensor and attach to ego vehicle
        self.collision_sensor = self.server_world.spawn_actor(
            blueprint_library.find('sensor.other.collision'), carla.Transform(), attach_to=self.ego_vehicle
        )
        self.collision_sensor.listen(self.collision)
        # default: ego vehicle not in collision
        self.is_ego_vehicle_in_collision = False

        # publish walker and sensor changes
        self.server_world.tick()


    def finalize_episode(self, episode_counter: int, step_counter: int):
        # destroy actors
        if self.ego_vehicle is not None:
            self.ego_vehicle.destroy()
            self.ego_vehicle = None
        if self.pedestrian is not None:
            self.pedestrian.destroy()
            self.pedestrian = None
        if self.incoming_exo_vehicle is not None:
            self.incoming_exo_vehicle.destroy()
            self.incoming_exo_vehicle = None
        for vehicle in self.parked_exo_vehicles:
            vehicle.destroy()
        self.parked_exo_vehicles.clear()
        # destroy camera sensor
        if self.rgb_camera is not None:
            self.rgb_camera.destroy()
            self.rgb_camera = None
        # empty queue
        while not self.rgb_camera_image_queue.empty(): self.rgb_camera_image_queue.get()
        # destroy collision sensor
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
            self.collision_sensor = None


    # TODO: refactor
    # returns all obstacles in the range of the car's sensors
    # position of obstacles is not accurate (float -> int, i.e. discretization of positions)
    def get_obstacles(self):
        # list of obstacles
        obstacles = list()
        # pedestrian is not visible to the agent as default
        is_pedestrian_observable = False

        # has the ego vehicle already passed the pedestrian?
        if self.scenario[0] == 6:
            if self.pedestrian.get_location().y > self.ego_vehicle.get_location().y:
                is_pedestrian_observable = True
        elif self.pedestrian.get_location().y < self.ego_vehicle.get_location().y:
            is_pedestrian_observable = True
        # make pedestrian invisible if it is behind the ego vehicle
        if not is_pedestrian_observable:
            self.is_pedestrian_observable = False

        # euclidean distance of pedestrian less than 50m to ego-vehicle?
        if distance_actors(self.ego_vehicle, self.pedestrian) <= 50.0 and is_pedestrian_observable:
            
            # pedestrian moved past parked vehicle (and is now on street)
            if self.scenario[0] == 3 and self.pedestrian.get_location().x >= self.parked_exo_vehicles[0].get_location().x:
                obstacles.append((int(self.pedestrian.get_location().x), int(self.pedestrian.get_location().y)))
                self.is_pedestrian_observable = True
            
            # pedestrian moved past parked vehicle (and is now on street)
            if self.scenario[0] in [7, 8] and self.pedestrian.get_location().x <= self.parked_exo_vehicles[0].get_location().x:
                obstacles.append((int(self.pedestrian.get_location().x), int(self.pedestrian.get_location().y)))
                self.is_pedestrian_observable = True
           
            # any pedestrian is recognized as an obstacle
            if self.scenario[0] in [1, 2, 4, 5, 6, 9, 10]:
                obstacles.append((int(self.pedestrian.get_location().x), int(self.pedestrian.get_location().y)))
                self.is_pedestrian_observable = True
       
        # scenarios with exo vehicles
        if self.scenario[0] in [3, 7, 8, 10]:
            if self.scenario[0] != 10: exo_vehicle_location = self.parked_exo_vehicles[0].get_location()
            else: exo_vehicle_location = self.incoming_exo_vehicle.get_location()
            # exo vehilce closer than 50m?
            if distance_locations(self.ego_vehicle.get_location(), exo_vehicle_location) <= 50.0:
                xmin = math.ceil(exo_vehicle_location.x - Config.Carla.EGO_VEHICLE_WIDTH / 2)
                xmax = math.ceil(exo_vehicle_location.x + Config.Carla.EGO_VEHICLE_WIDTH / 2)
                ymin = math.ceil(exo_vehicle_location.y - Config.Carla.EGO_VEHICLE_LENGTH / 2)
                ymax = math.ceil(exo_vehicle_location.y + Config.Carla.EGO_VEHICLE_LENGTH / 2)
                # discretize hitbox by iterating over length and width of the car in steps of 1
                # each step is counted as an obstacle
                for x in range(xmin, xmax + 1):
                    for y in range(ymin, ymax + 1):
                        obstacles.append((int(x), int(y)))

        # scenario without pedestrian
        if self.scenario[0] == 11:
            # np pedestrian in scenario = defaults to not observable
            self.is_pedestrian_observable = False
            # add incoming car as obstacle
            incoming_vehicle_location = self.incoming_exo_vehicle.get_location()
            if distance_locations(self.ego_vehicle.get_location(), incoming_vehicle_location) <= 50.0:
                xmin = math.ceil(incoming_vehicle_location.x - Config.Carla.EGO_VEHICLE_WIDTH / 2)
                xmax = math.ceil(incoming_vehicle_location.x + Config.Carla.EGO_VEHICLE_WIDTH / 2) 
                ymin = math.ceil(incoming_vehicle_location.y - Config.Carla.EGO_VEHICLE_LENGTH / 2)
                ymax = math.ceil(incoming_vehicle_location.y + Config.Carla.EGO_VEHICLE_LENGTH / 2)
                for x in range(xmin, xmax + 1):
                    for y in range(ymin, ymax + 1):
                        obstacles.append((int(x), int(y)))

            # add parked cars as obstacles
            for vehicle in self.parked_exo_vehicles:
                vehicle_location = vehicle.get_location()
                if distance_locations(self.ego_vehicle.get_location(), vehicle_location) <= 50.0:
                    xmin = math.ceil(vehicle_location.x - Config.Carla.EGO_VEHICLE_WIDTH / 2)
                    xmax = math.ceil(vehicle_location.x + Config.Carla.EGO_VEHICLE_WIDTH / 2) 
                    ymin = math.ceil(vehicle_location.y - Config.Carla.EGO_VEHICLE_LENGTH / 2)
                    ymax = math.ceil(vehicle_location.y + Config.Carla.EGO_VEHICLE_LENGTH / 2)
                    for x in range(xmin, xmax + 1):
                        for y in range(ymin, ymax + 1):
                            obstacles.append((int(x), int(y)))

        # scenario without pedestrian
        if self.scenario[0] == 12:
            # np pedestrian in scenario = defaults to not observable
            self.is_pedestrian_observable = False
            # add parked car as obstacle
            parked_car = self.parked_exo_vehicles[0]
            px, py = round(parked_car.get_location().x), round(parked_car.get_location().y)
            for i in [-1, 0, 1]:
                for j in [-2, -1, 0, 1, 2]:
                    obstacles.append((px + i, py + j))

            # ad incoming car as obstacle
            incoming_vehicle_location = self.incoming_exo_vehicle.get_location()
            if distance_locations(self.ego_vehicle.get_location(), incoming_vehicle_location) <= 50.0:
                for i in [-1, 0, 1]:
                    for j in [-2, -1, 0, 1, 2]:
                        obstacles.append((incoming_vehicle_location.x + i, incoming_vehicle_location.y + j))

        # if we are recording pedestrian data, we always consider the pedestrian as observable
        if Config.RECORD_PEDESTRIAN_DATA: self.is_pedestrian_observable = True
        
        return obstacles
    

    # manual definitions of when the pedestrian/incoming vehilce should start walking/driving
    # and when they should stop, i.e. at which x/y-coordinates
    # in other words: benchmark = scenario.py + world.py (next_scene() & tick())
    def tick(self, clock):
        # update interface
        self.hud.tick(self, clock)

        # determine crossing distance of pedestrian
        # i.e. how close does the ego vehicle have to be to the pedestrian for it to start crossing the street
        if abs(self.ego_vehicle.get_location().y - self.pedestrian.get_location().y) < self.pedestrian_crossing_distance:

            # cross from left to right
            if self.scenario[0] in [1, 2, 3]:
                self.pedestrian.apply_control(carla.WalkerControl(carla.Vector3D(self.pedestrian_velocity, 0, 0), 1))
                # stop walking
                if self.scenario[0] in [1, 3] and self.pedestrian.get_location().x > 6.5:
                    self.pedestrian.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), 1))
                # stop walking
                if self.scenario[0] == 2 and self.pedestrian.get_location().x > 97.0:
                    self.pedestrian.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), 1))

            # cross from right to left
            elif self.scenario[0] in [4, 5, 7, 8, 6, 10]:
                self.pedestrian.apply_control(carla.WalkerControl(carla.Vector3D(-self.pedestrian_velocity, 0, 0), 1))
                # stop walking
                if self.scenario[0] in [4, 7, 8] and self.pedestrian.get_location().x < -6.5:
                    self.pedestrian.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), 1))
                # stop walking
                if self.scenario[0] in [5, 6] and self.pedestrian.get_location().x < 83.0:
                    self.pedestrian.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), 1))

            elif self.scenario[0] == 9:
                self.pedestrian.apply_control(carla.WalkerControl(carla.Vector3D(0, self.pedestrian_velocity, 0), 1))

        # regardless of distance to pedestrian
        if self.scenario[0] == 10:
            # stop walking
            if self.pedestrian.get_location().x < -4.5:
                self.pedestrian.apply_control(carla.WalkerControl(carla.Vector3D(0, 0, 0), 1))
                
            # if incoming vehicle has travelled a sufficiently large distance on the road OR
            exo_vehicle_location = self.incoming_exo_vehicle.get_location().y > 250
            # if distance between pedstrian and incoming vehicle is sufficiently small AND
            pedestrian_exo_vehicle_distance = \
                0 < (self.pedestrian.get_location().y - self.incoming_exo_vehicle.get_location().y) < 5
            # pedestrian has travelled sufficiently far (across the street)
            pedestrian_location = self.pedestrian.get_location().x > -4.4
            if exo_vehicle_location or (pedestrian_exo_vehicle_distance and pedestrian_location):
                # stop incoming vehicle
                self.incoming_exo_vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
            else:
                # otherwise incoming vehicle drives at 32.4 km/h
                self.incoming_exo_vehicle.set_target_velocity(carla.Vector3D(0, 9, 0))

        # no pedestrian scenario
        if self.scenario[0] == 11:
            # stop incoming vehicle if it is sufficiently close th the agent vehicle
            if self.incoming_exo_vehicle.get_location().y - self.ego_vehicle.get_location().y < 10:
                self.incoming_exo_vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
            else:
                # drive
                self.incoming_exo_vehicle.set_target_velocity(carla.Vector3D(0, -self.pedestrian_velocity, 0))

        # no pedestrian scenario
        if self.scenario[0] == 12:
            # incoming vehicle will always continue driving straight
            self.incoming_exo_vehicle.set_target_velocity(carla.Vector3D(0, self.pedestrian_velocity, 0))


    # ============================================================================================== #
    #                                         DEBUG METHODS                                          #
    # ============================================================================================== #
    def render(self, display):
        # draw rgb image
        rgb_camera_image = self.rgb_camera_image_queue.get()
        rgb_camera_image = np.reshape(rgb_camera_image.raw_data, (rgb_camera_image.height, rgb_camera_image.width, 4))
        rgb_camera_image = rgb_camera_image[:, :, :3]
        rgb_camera_image = rgb_camera_image[:, :, ::-1]
        display.blit(pygame.surfarray.make_surface(rgb_camera_image.swapaxes(0, 1)), (0, 0))
        
        #vehicles = self.server_world.get_actors().filter('vehicle.*')
        # draw bounding boxes
        #bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(vehicles, self.rgb_camera)
        #log_info(f"bounding_boxes: {bounding_boxes}")
        #ClientSideBoundingBoxes.draw_bounding_boxes(display, bounding_boxes)

        # pass display to hud to draw on
        #self.hud.render(display)


    def render_rgb_camera_with_bounding_boxes(self) -> np.ndarray:
        rgb_camera_image = self.rgb_camera_image_queue.get()
        # reshape the raw data into an RGB array
        rgb_camera_image = np.reshape(
            np.copy(rgb_camera_image.raw_data), (rgb_camera_image.height, rgb_camera_image.width, 4)
        )

        # get the world to camera matrix
        world_2_camera = np.array(self.rgb_camera.get_transform().get_inverse_matrix())

        # get all actors (ego and exo vehicles and walkers)
        for actor_snapshot in self.server_world.get_snapshot():
            actor = self.server_world.get_actor(actor_snapshot.id)
            if actor is not None:
                # for all vehicles and walkers
                if "vehicle" in actor.type_id or "walker" in actor.type_id:
                    # get bounding box vertices based on current actor location
                    vertices = [v for v in actor.bounding_box.get_world_vertices(actor.get_transform())]

                    if "walker" in actor.type_id:
                        log_debug(f"pedestrian bounding_box center: {actor.bounding_box.location}")

                    # connect vertices by drawing all connecting lines as defined in self.edges (fixed order)
                    for edge in self.edges:
                        ray0 = vertices[edge[0]] - self.rgb_camera.get_transform().location
                        ray0 = np.array((ray0.x, ray0.y, ray0.z))
                        ray1 = vertices[edge[1]] - self.rgb_camera.get_transform().location
                        ray1 = np.array((ray1.x, ray1.y, ray1.z))

                        cam_forward_vec = self.rgb_camera.get_transform().get_forward_vector()
                        cam_forward_vec = np.array((cam_forward_vec.x, cam_forward_vec.y, cam_forward_vec.z))

                        # project CARLA world coordinates to camera coordinates
                        if cam_forward_vec.dot(ray0) > 0:
                            p1 = get_image_point(vertices[edge[0]], self.projection_matrix, world_2_camera)
                        # vertex is behind the camera
                        else:
                            p1 = get_image_point(vertices[edge[0]], self.projection_matrix_behind, world_2_camera)

                        if cam_forward_vec.dot(ray1) > 0:
                            p2 = get_image_point(vertices[edge[1]], self.projection_matrix, world_2_camera)
                        else:
                            p2 = get_image_point(vertices[edge[1]], self.projection_matrix_behind, world_2_camera)

                        # sanity check: are the retrieved points inside the relevant canvas?
                        p1_in_canvas = point_in_canvas(p1, Config.Carla.DISPLAY_SCREEN_HEIGHT, Config.Carla.DISPLAY_SCREEN_WIDTH)
                        p2_in_canvas = point_in_canvas(p2, Config.Carla.DISPLAY_SCREEN_HEIGHT, Config.Carla.DISPLAY_SCREEN_WIDTH)

                        # if not: skip edge
                        if not p1_in_canvas and not p2_in_canvas: continue
                        cv2.line(rgb_camera_image, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p2[1])), (255,0,0, 255), 1)

        return rgb_camera_image  
    
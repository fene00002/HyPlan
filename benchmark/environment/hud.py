import datetime

import pygame
import carla

from utils.config import Config
from utils.utils import velocity_kmh, distance_actors


class HUD(object):
    def __init__(self):
        self.display_dimensions = (Config.Carla.DISPLAY_SCREEN_WIDTH, Config.Carla.DISPLAY_SCREEN_HEIGHT)

        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self.server_clock = pygame.time.Clock()
        
        self.font = pygame.font.Font(pygame.font.get_default_font(), 20)
        self.info_text = []


    def on_world_tick(self, timestamp):
        self.server_clock.tick()
        self.server_fps = self.server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds


    # display information on agent, nearby cars and pedestrians on tick (each frame?)
    def tick(self, client_world, clock):
        # ego vehicle state
        ego_vehicle_transform = client_world.ego_vehicle.get_transform()
        ego_vehicle_control = client_world.ego_vehicle.get_control()

        # information to be displayed
        self.info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Velocity: % 15.0f km/h' % (velocity_kmh(client_world.ego_vehicle)),
            "Angle: % 15.0f degrees" % (ego_vehicle_transform.rotation.yaw),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (ego_vehicle_transform.location.x, ego_vehicle_transform.location.y))
        ]
        if isinstance(ego_vehicle_control, carla.VehicleControl):
            self.info_text += [
                ('Throttle:', ego_vehicle_control.throttle, 0.0, 1.0),
                ('Steer:', ego_vehicle_control.steer, -1.0, 1.0),
                ('Brake:', ego_vehicle_control.brake, 0.0, 1.0),
                ('Reverse:', ego_vehicle_control.reverse),
                ('Hand brake:', ego_vehicle_control.hand_brake),
                ('Manual:', ego_vehicle_control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(ego_vehicle_control.gear, ego_vehicle_control.gear)
            ]

        pedestrian = client_world.pedestrian
        # display pedestrian information
        if pedestrian is not None:
            pedestrian_distance = distance_actors(pedestrian, client_world.ego_vehicle)
            pedestrian_transform = pedestrian.get_transform()
            self.info_text += [
                '',
                'Pedestrian [Distance: % 3.2f, Location: (%5.2f, %5.2f)]' % \
                (pedestrian_distance, pedestrian_transform.location.x, pedestrian_transform.location.y)
            ]

        incoming_vehicle = client_world.incoming_exo_vehicle
        # display incoming vehicle information
        if incoming_vehicle is not None:
            incoming_vehicle_distance = distance_actors(incoming_vehicle, client_world.ego_vehicle)
            incoming_vehicle_transform = incoming_vehicle.get_transform()
            self.info_text += [
                '',
                'Incoming Vehicle [Distance: % 3.2f, Location: (%5.2f, %5.2f)]' % \
                (incoming_vehicle_distance, incoming_vehicle_transform.location.x, incoming_vehicle_transform.location.y)
            ]



    def render(self, display):
        info_surface = pygame.Surface((220, self.display_dimensions[1]))
        info_surface.set_alpha(100)
        display.blit(info_surface, (0, 0))
        v_offset = 4
        bar_h_offset = 100
        bar_width = 106
        for item in self.info_text:
            if v_offset + 18 > self.display_dimensions[1]:
                break
            if isinstance(item, list):
                if len(item) > 1:
                    points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                    pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                item = None
                v_offset += 18
            elif isinstance(item, tuple):
                if isinstance(item[1], bool):
                    rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                    pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                else:
                    rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                    pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                    f = (item[1] - item[2]) / (item[3] - item[2])
                    if item[2] < 0.0:
                        rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                    else:
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                    pygame.draw.rect(display, (255, 255, 255), rect)
                item = item[0]
            if item:  # At this point has to be a str.
                surface = self.font.render(item, True, (255, 255, 255))
                display.blit(surface, (8, v_offset))
            v_offset += 18
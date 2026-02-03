from enum import Enum
from datetime import datetime


class StringEnum(Enum):
     def __str__(self):
          return str(self.value)

class Address(StringEnum):
     LOCAL = "172.31.144.1"
     REMOTE = "127.0.0.1"

class Agent(StringEnum):
     IS_DESPOT = "IS-DESPOT"
     HYP_DESPOT = "HyP-DESPOT"
     A2C = "A2C"
     PPO = "PPO"
     HyLEAP = "HyLEAP"
     HyLEAR = "HyLEAR"
     HyPLAN = "HyPLAN"
     LEADER = "LEADER"

class DespotVariant(StringEnum):
     IS_DESPOT = Agent.IS_DESPOT
     HYP_DESPOT = Agent.HYP_DESPOT

PLANNING_AGENTS = [Agent.IS_DESPOT, Agent.HYP_DESPOT]
LEARNING_AGENTS = [Agent.A2C]
HYBRID_AGENTS = [Agent.HyLEAP, Agent.HyLEAR, Agent.HyPLAN, Agent.LEADER]

class Mode(StringEnum):
     TRAIN = "train"
     VAL = "validate"
     TEST = "test"

class Action(Enum):
     DECELERATE: int = 0
     MAINTAIN: int = 1
     ACCELERATE: int = 2


class Config:

     class Carla:
          NUM_PEDESTRIANS: int = 1
          NUM_ACTIONS: int = 3
          HOST: Address = None
          PORT: int = None
          REMOTE: bool = None

          SIMULATION_STEP = 0.05 # each scene simulation step is 50 milliseconds
          SENSOR_SIMULATION_STEP = '0.5'
          SYNCHRONONOUS = True # wait for client to tick the server before initiating new simulation step

          # scene simulation rendering in CARLA
          DISPLAY_FOV = 90
          DISPLAY_SCREEN_WIDTH = 1280
          DISPLAY_SCREEN_HEIGHT = 720

          # for training NNs
          CAR_INTENTION_IMAGE_WIDHT: str = None
          CAR_INTENTION_IMAGE_HEIGHT: str = None

          GOAL_TOLERANCE = 3.0
          MAX_SPEED = 50 / 60 / 60 * 1000 # 50 km/h = (50/60)/60 = 50/3600 m/s
          VELOCITY_STEP = 5 / 60 / 60 * 1000
          MAX_STEERING_ANGLE = 70 # in degrees
          
          FILTER = 'vehicle.audi.tt'
          AGENT_VEHICLE_ROLENAME = 'agent_vehicle'
          PARKED_VEHICLE_ROLENAME = 'parked_vehicle'
          INCOMING_VEHICLE_ROLENAME = 'incoming_vehicle'

          EGO_VEHICLE_LENGTH = 4.182
          EGO_VEHICLE_WIDTH = 1.994
          NEARMISS_FRONT_MARGIN = 1.5
          NEARMISS_SIDE_MARGIN = NEARMISS_BACK_MARGIN = 0.5
          EXO_PEDESTRIAN_LENGTH = 0.375
          EXO_PEDESTRIAN_WIDTH = 0.375


     class Despot:
          VARIANT: DespotVariant = None 
          PORT: int = None
          TRACKING: bool = None
          OBSERVATION_SIZE: int = 6 # Car: (x, y, angle, speed), Pedestrian: (x, y)

          FAVOR_ACCELERATE: bool = None
          CORRECT_VELOCITY: bool = None 
          CORRECT_TIMING: bool = None
          IMPROVED_HEURISTIC: bool = None
          AGGRESSIVE_BELIEF_UPDATES: bool = None
          MINIMAL_NOISE: bool = None
          NO_IMPORTANCE_SAMPLING: bool = None
          NO_NORMALIZATION: bool = None
          TIMEOUT: float = None
          TIME_PER_PLANNING_STEP: float = None
          NOISE: float = None
          MAX_SEARCH_DEPTH: int = None
          DISCOUNT: float = None
          PARTICLE_NUMBER: int = None
          GAP: float = None 
          MAX_POLICY_SIM_LEN: int = None
          PRUNING_CONSTANT: float = None

          CPU_MULTITHREADING: bool = None
          NUM_THREADS: int = None
          

     class A2C:
          DROPOUT: bool = None
          HIDDEN_LAYER_SIZE: int = None
          STANDARDIZE_RETURN: bool = None
          STANDARDIZE_ADVANTAGE: bool = None
          CLIP_GRADIENT: bool = None
          GAE_LAMBDA: bool = None
          LOSS_CLIPPING_COEFFICIENT: float = None
          CLIP_CRITIC_LOSS: bool = None
          CRITIC_LOSS_COEFFICIENT: float = None
          ENTROPY_COEFFICIENT: float = 0.01 # from PPO paper
          MODEL_ARCHITECTURE: str = None
          PPO: bool = None

          LEARNING_RATE = 0.0003 # from PPO paper
          MOMENTUM = 0.9 # from PPO paper
          DISCOUNT = 0.99


     class HyLEAP:
          HACKY: bool = None
          DECOUPLE: bool = None
          OBSERVATION_SIZE = 8 # Car: (x, y, angle, speed), Pedestrian: (x, y), reward, previous action

          LSTM_STATE_SIZE = 256
          LEARNING_RATE = 1e-4
          DECAY = 0.99
          MOMENTUM = 0.0
          EPSILON = 0.1
          L2_DECAY = 0.0005


     class HyPLAN:
          NUM_FORWARD_PASSES: bool = None
          NO_VERTICAL_PRUNING: bool = None
          UNCERTAINTY_ANALYSIS_DIR: str = "uncertainty_analysis"
          MIN_MAX_VARIANCE_FILENAME: str = "min_max_variance_observed_during_training.txt"


     class HyLEAR:
          num_steps = 3000000
          start_steps = 25000
          num_eval_steps = 3000
          update_interval = 4
          target_update_interval = 3000
          log_interval = 10
          eval_interval = 10000
          batch_size = 128
          lr = 0.00005
          buffer_capacity = 60000
          gamma = 0.99
          multi_step = 1
          target_entropy_ratio = 0.6
          use_per = True
          dueling_net = True


     class Leader:
          ATTENTION_SAMPLING: bool = None
          REPLAY_MIN: int = 5
          REPLAY_MAX: int = 100000
          BATCH_SIZE: int = 2
          LSTM_STATE_SIZE: int = 1024
          MAX_FEATURE_LEN: int = 181
          NUM_FEATURES: int = 3
          MAX_PEDESTRIANS: int = 1
          MAX_TRAJECTORY_LENGTH: int = None
          LEARNING_RATE: int = 1e-4
          HANDCRAFTED_ATT: bool = False
          ATTENTION_SIZE: int = 181
          CRITIC_WARM_UP_ITERATIONS = 1000


     # kill script
     TERMINATE: bool = False
     # always use GPU
     DEVICE: str = "cuda"

     #===============================================
     #             META CONFIGURATION
     #     (affects script behaviour as a whole)
     #===============================================
     AGENT: Agent = None
     PREDICT_PEDESTRIAN_PATH: bool = None
     RISK_AWARE_PATH: bool = None
     DISPLAY = None
     REWARD_FUNCTION: str = None
     SEED: int = None
     CALIBRATE_CONFIDENCE: bool = None

     #===============================================
     #             LOGGING CONFIGURATION
     #===============================================
     VERBOSE: bool = None
     # logging output directories
     OUTPUT_DIR: str = None
     METRICS_DIR: str = None
     MODEL_DIR: str = None
     DEBUG_DIR: str = None
     DATA_DIR: str = None
     CAR_INTENTION_DIR: str = None
     ERROR_DISTRIBUTION_DIR: str = None
     UNCALIBRATED_ECDF_DIR: str = None
     CALIBRATED_ECDF_DIR: str = None

     MODEL_SAVE_FREQUENCY_EPISODES: int = 250
     MODEL_CHECKPOINT: str = None
     RECORD_PEDESTRIAN_DATA: bool = None
     RECORD_CAR_INTENTION_IMAGES: bool = None
     GLOBAL_START_TIME: str = datetime.now().strftime('%d.%m.%Y_%H.%M.%S')

     #===============================================
     #   SCENARIOS & EPISODES & STEPS CONFIGURATION
     #===============================================
     MODE: Mode = None
     RESUME: bool = None
     INITIAL_EPISODE: int = None
     EPOCHS: int = None
     MAX_EPISODE_STEPS: int = None

     # bnechmark specifications
     TRAIN_SCENARIOS = None
     TRAIN_PED_SPEED_RANGE = [0.6, 2.0] # m/s
     TRAIN_PED_DIST_RANGE = [0, 40] # in ms

     VAL_SCENARIOS = None
     VAL_PED_SPEED_RANGE = [[0.2, 0.5], [2.1, 2.8]] # m/s
     VAL_PED_DIST_RANGE = [4.25, 49.25] # in ms

     TEST_SCENARIOS = None
     TEST_PED_SPEED_RANGE = [0.25, 2.85] # m/s
     TEST_PED_DIST_RANGE = [4.75, 49.75] # in ms
     TEST_CAR_SPEED_RANGE = [2.77, 5.55] # im m/s


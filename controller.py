import os
from argparse import ArgumentParser, SUPPRESS, RawTextHelpFormatter
import textwrap
import time
import json
from collections import Counter

import numpy as np
import psutil

from benchmark.environment.environment import CarlaCTS02
from agents.planner.isdespotp import ISDespotP
from agents.learner.a2c import A2C
from agents.learner.ppo import PPO
from agents.hybrid.leader import LEADER
from agents.hybrid.hyleap import HyLEAP
from agents.hybrid.hyplan import HyPLAN
from agents.hybrid.hylear_refactored import HyLEAR_NavA2C
from utils.config import Config, Address, Agent, DespotVariant, Mode, HYBRID_AGENTS, PLANNING_AGENTS, LEARNING_AGENTS
from utils.utils import start_carla_server, distance_locations, has_agent_stopped, find_free_port, kill_all_processes, velocity_kmh
from utils.logger import initialize_logging, log_debug, log_info, log_performance_metrics, log_data


if __name__ == '__main__':

    arg_parser = ArgumentParser(
        description='CARLA CTS02 Benchmark Script',
        argument_default=SUPPRESS,
        prog=__file__, 
        usage='%(prog)s [options]',
        allow_abbrev=False,
        add_help=True,
        formatter_class=RawTextHelpFormatter
    )   

    meta_level_group = arg_parser.add_argument_group(
        title="META arguments", description="arguments that affect script execution as a whole"
    )
    meta_level_group.add_argument(
        '--agent',
        dest="agent",
        metavar=(", ".join([str(agent) for agent in Agent])),
        action="store",
        type=str, 
        choices=[str(agent) for agent in Agent],
        nargs="?", 
        default="is_despot",
        help=textwrap.dedent(
            ''' 
            The agent to be evaluated over the Carla-CTS02 benchmark (default: %(default)s).
            
            ''')
    )
    meta_level_group.add_argument(
        '--mode',
        dest="mode",
        metavar="train, validate, test",
        action="store",
        type=str, 
        choices=["train", "validate", "test"],
        nargs="?", 
        default="train",
        help=textwrap.dedent(
            '''
            Whether to train, validate or test the specified agent (default: %(default)s).

            ''')
    )   
    meta_level_group.add_argument(
        '--scenario',
        dest="scenario",
        metavar="[0, 12]",
        action="store",
        type=str, 
        choices=["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"],
        nargs="+", 
        default=["00"],
        help=textwrap.dedent(
            '''
            Scenario(s) to evaluate the agent on (default: %(default)s = all). 
            Use like --scenario 01 02 03 04 to select the first four scenarios.
            Only mode --test can be run in conjunction with scenario 11 & 12.

            ''')
    ) 
    meta_level_group.add_argument(
        '--resume_from_episode',
        dest="resume_from_episode",
        metavar="1, 12626",
        action="store",
        type=int, 
        nargs="?", 
        default=1,
        help=textwrap.dedent(
            '''
            Episode number to resume from with the given episode included (default: %(default)s).

            ''')
    ) 
    meta_level_group.add_argument(
        '--remote',
        dest="remote",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Required if the script is run on SLURM cluster (default: %(default)s).
            
            ''')
    ) 
    meta_level_group.add_argument(
        '--predict_pedestrian_path',
        dest="predict_pedestrian_path",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Enables pedestrian path prediction using M2P3 (default: %(default)s).
            This requires the existence of a trained M2P3 model in the 'input' directory.
            
            ''')
    )
    meta_level_group.add_argument(
        '--plan_path_with_risk',
        dest="plan_path_with_risk",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Enables risk aware path planning (default: %(default)s).
            WARNING: This will influence execution time and performance.

            ''')
    ) 
    meta_level_group.add_argument(
        '--load_checkpoint',
        dest="load_checkpoint",
        metavar="DIRECTORY NAME",
        action="store",
        type=str, 
        nargs="?", 
        default="latest",
        help=textwrap.dedent(
            '''
            Specify model checkpoint directory to load for testing (default: %(default)s).
            
            ''')
    )
    meta_level_group.add_argument(
        '--output_directory',
        dest="output_directory",
        metavar="DIRECTORY NAME",
        action="store",
        type=str, 
        nargs="?", 
        default=None,
        help=textwrap.dedent(
            '''
            Directory under which all script outputs will be centralized (default: %(default)s).
            If none is specified, it will be constructed based on the arguments provided. 
            
            ''')
    )
    meta_level_group.add_argument(
        '--epochs',
        dest="epochs",
        metavar="1, 2, 3, 4",
        action="store",
        type=int,
        choices=[1, 2, 3, 4],
        nargs="?", 
        default=1,
        help=textwrap.dedent(
            '''
            Specifies the number of iterations over the entire training set (default: %(default)s).
            
            ''')
    )
    meta_level_group.add_argument(
        '--max_episode_steps',
        dest="max_episode_steps",
        metavar="1, 4000",
        action="store",
        type=int, 
        choices=range(1, 4000+1, 1),
        nargs="?", 
        default=1000,
        help=textwrap.dedent(
            '''
            Enforces a maximum number of steps for each simulated episode,
            after which any episode will be forcefully terminated even when 
            no conclusive result (collision or goal) has been reached yet (default: %(default)s).
            Prematurely terminated episodes will not be used for training.
            
            ''')
    )
    meta_level_group.add_argument(
        '--seed',
        dest="seed",
        metavar="0, 1024",
        action="store",
        type=int, 
        choices=range(0, 4096, 1),
        nargs="?", 
        default=42,
        help=textwrap.dedent(
            '''
            Random number seed (default: %(default)s).
            
            ''')
    ) 
    meta_level_group.add_argument(
        '--reward_function',
        dest="reward_function",
        metavar="akash, nils, despot",
        action="store",
        type=str, 
        choices=["akash", "nils", "despot"],
        nargs="?", 
        default="akash",
        help=textwrap.dedent(
            '''
            The reward function to use for both Python and C++ (default: %(default)s).

            ''')
    )
    meta_level_group.add_argument(
        '--calibrate_confidence',
        dest="calibrate_confidence",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Calculates and saves the empirical error distribution of the validation set
            or calibrate confidence estimates on the test set (default: %(default)s).
            WARNING: This will influence execution time and performance.

            ''')
    )     

    is_despot_group = arg_parser.add_argument_group("IS-DESPOT arguments")
    is_despot_group.add_argument(
        '--favor_accelerate',
        dest="favor_accelerate",
        action="store_true",
        default=True,
        help=textwrap.dedent(
            '''
            Favors accelerate over maintain/decelerate for belief node expansion duriong planning simulation of IS-DESPOT (default: %(default)s).
            This is used to reproduce legacy results.

            ''')
    )        
    is_despot_group.add_argument(
        '--correct_velocity',
        dest="correct_velocity",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Uses optimized ac- and deceleration velocity steps of the ego-vehicle in CARLA for planning (default: %(default)s).
            This option is there for the purpose of reproducing legacy results.

            ''')
    )    
    is_despot_group.add_argument(
        '--correct_timing',
        dest="correct_timing",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Uses actually elapsed time of scene simulation steps for planning (default: %(default)s).
            This option is there for the purpose of reproducing legacy results.

            ''')
    )    
    is_despot_group.add_argument(
        '--improved_heuristic',
        dest="improved_heuristic",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Uses an improved heuristic function inside the default policy and lower bound of IS-DESPOT 
            (default: %(default)s).

            ''')
    )    
    is_despot_group.add_argument(
        '--aggressive_belief_updates',
        dest="aggressive_belief_updates",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Belief is only influenced by current step (default: %(default)s).

            ''')
    )
    is_despot_group.add_argument(
        '--minimal_noise',
        dest="minimal_noise",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Removes as much (artificially injected) noise as possible during planning simulation (default: %(default)s).

            ''')
    )
    is_despot_group.add_argument(
        '--no_importance_sampling',
        dest="no_importance_sampling",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Do not use importance sampling (default: %(default)s).

            ''')
    )
    is_despot_group.add_argument(
        '--no_normalization',
        dest="no_normalization",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Disable normalization for importance distribution (default: %(default)s).

            ''')
    )
    is_despot_group.add_argument(
        '--noise',
        dest="noise",
        metavar="0, 1",
        action="store",
        type=float, 
        choices=np.arange(0, 1+0.01, 0.01).round(2),
        nargs="?", 
        default=0.05,
        help=textwrap.dedent(
            '''
            Noise level for transitions in belief update (default: %(default)s).

            ''')
    )  
    is_despot_group.add_argument(
        '--timeout',
        dest="timeout",
        metavar="0.01, 1",
        action="store",
        type=float, 
        choices=np.arange(0.01, 1.0, 0.01).round(2),
        nargs="?", 
        default=0.25,
        help=textwrap.dedent(
            '''
            Belief tree construction time per scene simulation step in seconds (default: %(default)s).

            ''')
    ) 
    is_despot_group.add_argument(
        '--time_per_planning_step',
        dest="time_per_planning_step",
        metavar="0.01, 1",
        action="store",
        type=float, 
        choices=np.arange(0.01, 1.0, 0.01).round(2),
        nargs="?", 
        default=0.25,
        help=textwrap.dedent(
            '''
            Time between planning simulation steps during belief tree construction in seconds (default: %(default)s).

            ''')
    ) 
    is_despot_group.add_argument(
        '--max_search_depth',
        dest="max_search_depth",
        metavar="1, 1000",
        action="store",
        type=int, 
        choices=range(1, 1000+1, 1),
        nargs="?", 
        default=20,
        help=textwrap.dedent(
            '''
            Maximum search depth during belief tree construction (default: %(default)s).

            ''')
    )  
    is_despot_group.add_argument(
        '--discount_factor',
        dest="discount_factor",
        metavar="0, 0.99",
        action="store",
        type=float, 
        choices=np.arange(0.0, 1.0, 0.01).round(2),
        nargs="?", 
        default=0.99,
        help=textwrap.dedent(
            '''
            Factor to discount future rewards (default: %(default)s).

            ''')
    )  
    is_despot_group.add_argument(
        '--particle_number',
        dest="particle_number",
        metavar="1, 10000",
        action="store",
        type=int, 
        choices=range(1, 10000+1, 1),
        nargs="?", 
        default=500,
        help=textwrap.dedent(
            '''
            Number of particles used to approximate belief nodes (default: %(default)s).

            ''')
    ) 
    is_despot_group.add_argument(
        '--gap_reduction_rate',
        dest="gap_reduction_rate",
        metavar="0, 1",
        action="store",
        type=float, 
        choices=np.arange(0.0, 1.0+0.01, 0.01).round(2),
        nargs="?", 
        default=0.95,
        help=textwrap.dedent(
            '''
            Required gap reduction rate of each trial (default: %(default)s).

            ''')
    ) 
    is_despot_group.add_argument(
        '--max_policy_simulation_length',
        dest="max_policy_simulation_length",
        metavar="0, 1000",
        action="store",
        type=int, 
        choices=range(0, 1000, 1),
        nargs="?", 
        default=60,
        help=textwrap.dedent(
            '''
            Number of steps to simulate the reactive controller at leaf nodes (default: %(default)s).

            ''')
    ) 
    is_despot_group.add_argument(
    '--pruning_constant',
    dest="pruning_constant",
    metavar="0.01, 1",
    action="store",
    type=float, 
    choices=np.arange(0.0, 1.0, 0.0001),
    nargs="?", 
    default=0,
    help=textwrap.dedent(
        '''
        Pruning constant for regularization (default: %(default)s).

        ''')
    ) 

    hyp_despot_group = arg_parser.add_argument_group("HYP-DESPOT arguments")
    hyp_despot_group.add_argument(
        '--GPU',
        dest="GPU",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Enable GPU parallelization (default: %(default)s).

            ''')
    )
    hyp_despot_group.add_argument(
        '--GPU_id',
        dest="GPU_id",
        metavar="0, 100",
        action="store",
        type=int, 
        choices=range(0, 100, 1),
        nargs="?", 
        default=0,
        help=textwrap.dedent(
            '''
            GPU used for parallelization (default: %(default)s).

            ''')
    )
    hyp_despot_group.add_argument(
        '--CPU',
        dest="CPU",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Enable CPU multithreading (default: %(default)s).

            ''')
    )
    hyp_despot_group.add_argument(
        '--num_threads',
        dest="num_threads",
        metavar="2, 100",
        action="store",
        type=int, 
        choices=range(2, 100, 1),
        nargs="?", 
        default=1,
        help=textwrap.dedent(
            '''
            Number of parallel CPU threads (default: %(default)s).

            ''')
    )
    hyp_despot_group.add_argument(
        '--exploration_scheme',
        dest="exploration_scheme",
        metavar="UCT | Vloss",
        action="store",
        type=str, 
        choices=["UCT", "Vloss"],
        nargs="?", 
        default="Vloss",
        help=textwrap.dedent(
            '''
            Scheme for guiding parallel simulation trajectory exploration (default: %(default)s).

            ''')
    )
    hyp_despot_group.add_argument(
        '--action_exploration_constant',
        dest="action_exploration_constant",
        metavar="0, 1",
        action="store",
        type=float, 
        choices=np.arange(0, 1+0.01, 0.01),
        nargs="?", 
        default=0.95,
        help=textwrap.dedent(
            '''
            Exploration constant for action branches (default: %(default)s).

            ''')
    )
    hyp_despot_group.add_argument(
        '--observation_exploration_const',
        dest="observation_exploration_const",
        metavar="0, 1",
        action="store",
        type=float, 
        choices=np.arange(0, 1+0.01, 0.01).round(2),
        nargs="?", 
        default=0.05,
        help=textwrap.dedent(
            '''
            Exploration constant for observation branches (default: %(default)s).

            ''')
    )

    a2c_group = arg_parser.add_argument_group("A2C arguments")
    a2c_group.add_argument(
        '--model_architecture',
        dest="model_architecture",
        action="store",
        type=str, 
        choices=["akash", "florian", "atari", "atari-lstm", "NavA2C"],
        nargs="?", 
        default=None,
        help=textwrap.dedent(
            '''
            Specifies the A2C model architecture to use (default: %(default)s).

            ''')
    ) 
    a2c_group.add_argument(
        '--ppo',
        dest="ppo",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Use Proximal Policy Optimization (default: %(default)s).

            ''')
    )  
    a2c_group.add_argument(
        '--hidden_layer_size',
        dest="hidden_layer_size",
        metavar="128, 256",
        action="store",
        type=int, 
        choices=[128, 256, 512],
        nargs="?", 
        default=None,
        help=textwrap.dedent(
            '''
            Specifies the size of the hidden layer of the neural network (128: A2C, 256: NavA2C) (default: %(default)s).

            ''')
    )
    a2c_group.add_argument(
        '--use_dropout',
        dest="use_dropout",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Trains A2C with a dropout layer in the critic's NN architecture (default: %(default)s).
            This is used for fair comparability between HyLEAP and HyPLAN as both NN architectures msut be identical.

            ''')
    )
    a2c_group.add_argument(
        '--standardize_return',
        dest="standardize_return",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Standardizes return across steps of a given episode (default: %(default)s).

            ''')
    )
    a2c_group.add_argument(
        '--standardize_advantage',
        dest="standardize_advantage",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Standardize advantage values (default: %(default)s).

            ''')
    )
    a2c_group.add_argument(
        '--gae_lambda',
        dest="gae_lambda",
        metavar="0, 1",
        action="store",
        type=float, 
        choices=np.arange(0, 1+0.01, 0.01).round(2),
        nargs="?", 
        default=None, # 0.9
        help=textwrap.dedent(
            '''
            Calculate episodic returns using generalized advantage estimation (default: %(default)s).
            See: https://arxiv.org/abs/1506.02438

            ''')
    )
    a2c_group.add_argument(
        '--clip_gradient',
        dest="clip_gradient",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Clip gradient values above 0.5 (default: %(default)s).

            ''')
    )
    a2c_group.add_argument(
        '--loss_clipping_coefficient',
        dest="loss_clipping_coefficient",
        metavar="0, 1",
        action="store",
        type=float, 
        choices=np.arange(0, 1+0.01, 0.01).round(2),
        nargs="?", 
        default=None, # 0.2
        help=textwrap.dedent(
            '''
            PPO's surrogate clipping coefficient for actor/policy loss calculation (default: %(default)s).

            ''')
    )
    a2c_group.add_argument(
        '--clip_critic_loss',
        dest="clip_critic_loss",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Use PPO's clipping coefficient also for critic's loss calculation (default: %(default)s).

            ''')
    )
    a2c_group.add_argument(
        '--critic_loss_coefficient',
        dest="critic_loss_coefficient",
        metavar="0, 1",
        action="store",
        type=float, 
        choices=np.arange(0, 1+0.01, 0.01).round(2),
        nargs="?", 
        default=None, # 0.5
        help=textwrap.dedent(
            '''
            Factor determining the contribution of the critic's loss in the overall loss of PPO (default: %(default)s).

            ''')
    )

    hyleap_group = arg_parser.add_argument_group("HyLEAP arguments")
    hyleap_group.add_argument(
        '--hacky_hyleap',
        dest="hacky_hyleap",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Enables Florian's hacky HyLEAP conde snippets (default: %(default)s).

            ''')
    )
    hyleap_group.add_argument(
        '--decouple_hyleap',
        dest="decouple_hyleap",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Trains HyLEAP network without interferring in belief tree construction (default: %(default)s).

            ''')
    )
        

    hyplan_group = arg_parser.add_argument_group("HyPLAN arguments")
    hyplan_group.add_argument(
        '--hyplan_num_forward_passes',
        dest="hyplan_num_forward_passes",
        metavar="10, 100",
        action="store",
        type=int, 
        choices=[10, 100],
        nargs="?", 
        default=None,
        help=textwrap.dedent(
            '''
            Specifies the number of forward passes used for uncertainty calculation (default: %(default)s).

            ''')
    )
    hyplan_group.add_argument(
        '--no_vertical_pruning',
        dest="no_vertical_pruning",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Disables vertical pruning for HyPlan (default: %(default)s).

            ''')
    )

    leader_group = arg_parser.add_argument_group("LEADER arguments")
    leader_group.add_argument(
        '--attention_sampling',
        dest="attention_sampling",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Samples pedestrian goal directions from attention distribution generated by LEADER (default: %(default)s).
            For more details view: https://arxiv.org/abs/2209.11422 or https://github.com/modanesh/LEADER

            ''')
    )  

    debug_group = arg_parser.add_argument_group("DEBUG arguments")
    debug_group.add_argument(
        '--carla_port', # optional command-line argument
        dest="carla_port", # name of the stored variable in the argument parser
        metavar="1024-65535", # name of the value-placeholder (displayed when --help is provided)
        action="store", # simply store provided value 
        type=int, # type conversion
        choices=range(1024, 65535+1, 1), # allowed non-system ports
        nargs="?", # extract as single item
        default=None, # default value if command-line option is not provided
        help=textwrap.dedent(
            '''
            TCP port to start the CARLA server with (default: None).
            Used exclusively when running the script locally.

            ''')
    )
    debug_group.add_argument(
        '--despot_port',
        dest="despot_port",
        metavar="1024, 65535",
        action="store",
        type=int, 
        choices=range(1024, 65535+1, 1),
        nargs="?", 
        default=None,
        help=textwrap.dedent(
            '''
            TCP port to start the C++ process running is-despot with (default: None).
            Used exclusively when running the script locally.
            
            ''')
    )
    debug_group.add_argument(
        '--track_planning_effort',
        dest="track_planning_effort",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Enables internal performance measure tracking in IS-DESPOT (default: %(default)s).
            
            ''')
    )
    debug_group.add_argument(
        '--record_pedestrian_path',
        dest="record_pedestrian_path",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Records the paths of all pedestrians of each simulated scene (default: %(default)s).
            
            ''')
    )
    debug_group.add_argument(
        '--record_car_intention_images',
        dest="record_car_intention_images",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Records the birdeye-view car intention image of each simulated scene step (default: %(default)s).
            
            ''')
    )
    debug_group.add_argument(
        '--verbose',
        dest="verbose",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Print debug output (default: %(default)s).
            WARNING: This will clutter the console massively!

            ''')
    )
    debug_group.add_argument(
        '--display',
        dest="display",
        action="store_true",
        default=False,
        help=textwrap.dedent(
            '''
            Renders the execution of each scene in the CARLA simulator (default: %(default)s).
            WARNING: This will influence execution time detrimentally and is only available locally!

            ''')
    )
    # process inputs
    args = vars(arg_parser.parse_args())
    
    # ========================================
    #           META LEVEL ARGUMENTS
    # *affecting script execution as a whole*
    # ========================================  
    Config.AGENT = Agent(args["agent"])
    Config.MODE = Mode(args["mode"])
    Config.Carla.NUM_PEDESTRIANS = 1
    Config.PREDICT_PEDESTRIAN_PATH = args["predict_pedestrian_path"]
    Config.RISK_AWARE_PATH = args["plan_path_with_risk"]
    Config.MAX_EPISODE_STEPS = args["max_episode_steps"]
    Config.SEED = args["seed"]
    Config.REWARD_FUNCTION = args["reward_function"]
    Config.CALIBRATE_CONFIDENCE = args["calibrate_confidence"]
    
    if Config.RISK_AWARE_PATH and Config.AGENT is not Agent.HyLEAR:
        raise ValueError("Considering risk has only been designed for HyLEAR.")
    
    if Config.REWARD_FUNCTION != "despot" and Config.AGENT in HYBRID_AGENTS + PLANNING_AGENTS:
        raise NotImplementedError("Missing C++ code for reward calculation.")

    if Config.MODE is Mode.TRAIN and Config.CALIBRATE_CONFIDENCE:
        raise ValueError("Calibrating confidence is only available during validating and testing.")
    
    # determine scenarios constituting the benchmark
    if args["scenario"] == ["00"]:
        if Config.MODE is Mode.TRAIN:
            Config.TRAIN_SCENARIOS = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
        elif Config.MODE is Mode.VAL:
            Config.VAL_SCENARIOS = ['01', '02', '03', '04', '05', '06', '07', '08', '09']
        elif Config.MODE is Mode.TEST:
            Config.TEST_SCENARIOS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', "11", "12"]
    else:
        if Config.MODE is Mode.TRAIN:
            Config.TRAIN_SCENARIOS = args["scenario"]
        elif Config.MODE is Mode.VAL:
            Config.VAL_SCENARIOS = args["scenario"]
        elif Config.MODE is Mode.TEST:
            Config.TEST_SCENARIOS = args["scenario"]


    # ========================================
    #    ARGUMENTS CONFLICTING WITH SLURM
    # ========================================  
    Config.Carla.REMOTE = args["remote"]
    Config.Carla.PORT = args["carla_port"]
    Config.Despot.PORT = args["despot_port"]
    Config.DISPLAY = args["display"]

    if Config.Carla.REMOTE and (Config.Carla.PORT is not None or Config.Despot.PORT is not None):
        raise ValueError("Ports for CARLA server and IS-DESPOT sub-processes are determined automatically on SLURM.")

    if Config.Carla.REMOTE and Config.DISPLAY: 
        raise ValueError("Can't render CARLA simulation on SLURM cluster.")  
    
    if not Config.Carla.REMOTE and Config.Carla.PORT is None:
        raise ValueError("Local script execution requires specifying a TCP port for the CARLA simulation server.")


    # ========================================
    #         ARGUMENTS FOR RESUMING 
    # ========================================    
    Config.EPOCHS = args["epochs"]
    Config.INITIAL_EPISODE = args["resume_from_episode"]
    Config.RESUME = True if Config.INITIAL_EPISODE != 1 else False 

    if not Config.MODE is Mode.TRAIN and Config.EPOCHS != 1:
        raise ValueError("Iterating multiple times over the test set doesn't make sense.")


    # ========================================
    #         ARGUMENTS FOR LOGGING 
    # ========================================    
    Config.VERBOSE = args["verbose"]
    Config.OUTPUT_DIR = args["output_directory"]
    Config.MODEL_CHECKPOINT = args["load_checkpoint"]
    Config.RECORD_PEDESTRIAN_DATA = args["record_pedestrian_path"]
    Config.RECORD_CAR_INTENTION_IMAGES = args["record_car_intention_images"]
    Config.Despot.TRACKING = args["track_planning_effort"]

    if Config.RECORD_PEDESTRIAN_DATA and Config.PREDICT_PEDESTRIAN_PATH:
        raise ValueError("Can't record pedestrian path data while also predicting them.")
    
    if Config.RECORD_PEDESTRIAN_DATA and Config.MODE is not Mode.TRAIN:
        raise ValueError("Can't record pedestrian path data of test or validation set as this leaks information.")
    
    if Config.RECORD_CAR_INTENTION_IMAGES and Config.AGENT is not Agent.IS_DESPOT:
        raise ValueError("Recording car intention images is only available for IS-DESPOT.")
    
    if Config.AGENT in [Agent.IS_DESPOT, Agent.HYP_DESPOT] and Config.MODE is Mode.TRAIN and not Config.RECORD_PEDESTRIAN_DATA:
        raise ValueError("POMDP planning algorithms can not be trained (this only makes sense if you want to record pedestrian path data).")


    #====================================================================
    #                 ARGUMENTS MODIFYING AGENT BEHAVIOUR
    #====================================================================
    # IS-DESPOT
    if Config.Despot.PORT is None: Config.Despot.PORT = find_free_port()
    Config.Despot.FAVOR_ACCELERATE = args["favor_accelerate"]
    Config.Despot.CORRECT_VELOCITY = args["correct_velocity"]
    Config.Despot.CORRECT_TIMING = args["correct_timing"]
    Config.Despot.IMPROVED_HEURISTIC = args["improved_heuristic"]
    Config.Despot.AGGRESSIVE_BELIEF_UPDATES = args["aggressive_belief_updates"]
    Config.Despot.MINIMAL_NOISE = args["minimal_noise"]
    Config.Despot.NO_IMPORTANCE_SAMPLING = args["no_importance_sampling"]
    Config.Despot.NO_NORMALIZATION = args["no_normalization"]
    Config.Despot.TIMEOUT = args["timeout"]
    Config.Despot.TIME_PER_PLANNING_STEP = args["time_per_planning_step"]
    Config.Despot.NOISE = args["noise"]
    Config.Despot.MAX_SEARCH_DEPTH = args["max_search_depth"]
    Config.Despot.DISCOUNT = args["discount_factor"]
    Config.Despot.PARTICLE_NUMBER = args["particle_number"]
    Config.Despot.GAP = args["gap_reduction_rate"]
    Config.Despot.MAX_POLICY_SIM_LEN = args["max_policy_simulation_length"]
    Config.Despot.PRUNING_CONSTANT = args["pruning_constant"]
    Config.Despot.PARTICLE_NUMBER = args["particle_number"]

    if Config.AGENT is Agent.HYP_DESPOT: 
        Config.Despot.VARIANT = DespotVariant.HYP_DESPOT
        Config.Despot.CPU_MULTITHREADING = args["CPU"]
        Config.Despot.NUM_THREADS = args["num_threads"]
    elif Config.AGENT is Agent.IS_DESPOT or Config.AGENT in HYBRID_AGENTS: 
        Config.Despot.VARIANT = DespotVariant.IS_DESPOT
    
    # A2C
    Config.A2C.MODEL_ARCHITECTURE = args["model_architecture"]
    Config.A2C.PPO = args["ppo"]
    Config.A2C.HIDDEN_LAYER_SIZE = args["hidden_layer_size"]
    Config.A2C.DROPOUT = args["use_dropout"]
    Config.A2C.STANDARDIZE_RETURN = args["standardize_return"]
    Config.A2C.STANDARDIZE_ADVANTAGE = args["standardize_advantage"]
    Config.A2C.GAE_LAMBDA = args["gae_lambda"]
    Config.A2C.CLIP_GRADIENT = args["clip_gradient"]
    Config.A2C.LOSS_CLIPPING_COEFFICIENT = args["loss_clipping_coefficient"]
    Config.A2C.CLIP_CRITIC_LOSS = args["clip_critic_loss"]
    Config.A2C.CRITIC_LOSS_COEFFICIENT = args["critic_loss_coefficient"]

    if Config.A2C.MODEL_ARCHITECTURE is None and Config.AGENT in [Agent.A2C, Agent.HyLEAP, Agent.HyPLAN]:
        raise ValueError("A2C's model architecture must be specified when using NavA2C, HyLEAP or HyPLAN.")
    if Config.A2C.MODEL_ARCHITECTURE is not None and Config.AGENT not in [Agent.A2C, Agent.HyLEAP, Agent.HyPLAN]:
        raise ValueError("Specifying A2C's model architecture makes sense only when using A2C, HylEAP or HyPLAN.")    
    if Config.A2C.MODEL_ARCHITECTURE != "florian" and Config.AGENT is not Agent.HyLEAP:
        raise ValueError("HyLEAP must use the NN architecture specified by Florian.")
    
    if Config.A2C.PPO and Config.AGENT not in [Agent.A2C, Agent.HyPLAN]:
        raise ValueError("PPO option only available for actor-critic architecture.")
    
    if Config.A2C.HIDDEN_LAYER_SIZE is not None and Config.AGENT not in [Agent.A2C, Agent.HyLEAP, Agent.HyPLAN]:
        raise ValueError("Specifying a hidden layer size only makes sense for agents that use a neural network.")    
    if Config.A2C.HIDDEN_LAYER_SIZE is None and Config.AGENT in [Agent.A2C, Agent.HyLEAP, Agent.HyPLAN]:
        # these use fixed hidden layer dimensions that are true to the original publications
        if Config.A2C.MODEL_ARCHITECTURE not in ["atari", "atari-lstm", "NavA2C"]:
            raise ValueError("The hidden layer size must be explicitly specified for A2C and its variants.")
        
    if Config.A2C.DROPOUT and Config.AGENT not in [Agent.A2C, Agent.HyLEAP, Agent.HyPLAN]:
        raise ValueError("Using a dropout layer in the critic's NN architecture is only available for agents that use A2C.")
    if Config.CALIBRATE_CONFIDENCE and not Config.A2C.DROPOUT:
        raise ValueError(
            "Calibrating confidence requires obtaining uncertainty estimates using MC-Dropout (enabled via --use_dropout)."
        )

    if Config.A2C.STANDARDIZE_RETURN and Config.AGENT not in [Agent.A2C, Agent.HyLEAP, Agent.HyPLAN]:
        raise ValueError("Standardizing rewards is only available for agents that learn.")

    if Config.A2C.STANDARDIZE_ADVANTAGE and Config.AGENT not in [Agent.A2C, Agent.HyLEAP, Agent.HyPLAN]:
        raise ValueError("Standardizing advantage values is only available for agents derived from A2C.")
    if Config.A2C.STANDARDIZE_ADVANTAGE and Config.MODE is not Mode.TRAIN:
        raise ValueError("Standardizing advantage values is only available during training.")    

    if Config.A2C.CLIP_GRADIENT and Config.AGENT not in [Agent.A2C, Agent.HyLEAP, Agent.HyPLAN]:
        raise ValueError("Clipping gradient values is only available for agents derived from A2C.")
    if Config.A2C.CLIP_GRADIENT and Config.MODE is not Mode.TRAIN:
        raise ValueError("Clipping gradient values is only available during training.")  

    if Config.A2C.GAE_LAMBDA is not None and not Config.A2C.PPO:
        raise ValueError("GAE is only available for PPO and derived agents.")
    if Config.A2C.GAE_LAMBDA is None and Config.A2C.PPO: 
        raise ValueError("PPO requires explicit specification of GAE's lambda value.")

    if Config.A2C.LOSS_CLIPPING_COEFFICIENT is not None and not Config.A2C.PPO:
        raise ValueError("Clipping objective function only available for PPO and derived.")
    if Config.A2C.LOSS_CLIPPING_COEFFICIENT is None and Config.A2C.PPO and Config.MODE is Mode.TRAIN:
        raise ValueError("PPO requires explicit specification of a clipping coefficient.")

    if Config.A2C.CLIP_CRITIC_LOSS and not Config.A2C.PPO:
        raise ValueError("Clipping critic loss is only available for PPO and derived agents.")
    if Config.A2C.CLIP_CRITIC_LOSS and Config.MODE is not Mode.TRAIN:
        raise ValueError("Clipping critic loss is only available during training.")
    
    if Config.A2C.CRITIC_LOSS_COEFFICIENT is not None and not Config.A2C.PPO:
        raise ValueError("Specifying the critic's loss coefficient is only available for PPO.")
    if Config.A2C.CRITIC_LOSS_COEFFICIENT is None and Config.A2C.PPO and Config.MODE is Mode.TRAIN:
        raise ValueError("PPO requires explicit specification of the critic's loss coefficient.")
    
    # HyLEAP
    Config.HyLEAP.HACKY = args["hacky_hyleap"]
    Config.HyLEAP.DECOUPLE = args["decouple_hyleap"]

    if Config.HyLEAP.HACKY and Config.AGENT not in [Agent.HyLEAP, Agent.HyPLAN]:
        raise ValueError("Florian's 'hacky' HyLEAP variant is only available for HyLEAP and HyPLAN.")

    if Config.HyLEAP.DECOUPLE and not Config.AGENT in [Agent.HyLEAP, Agent.HyPLAN]:
        raise ValueError("Decoupling HyLEAP during training is only available for HyLEAP and HyPLAN.")

    if Config.HyLEAP.DECOUPLE and not Config.MODE is Mode.TRAIN: 
        raise ValueError("Can't decouple HyLEAP architecture during testing.")
    
    if Config.HyLEAP.DECOUPLE and Config.HyLEAP.HACKY:
        raise ValueError("Florian's 'hacky' HyLEAP variant is already a partial decoupling.")
    
    # HyPLAN
    Config.HyPLAN.NUM_FORWARD_PASSES = args["hyplan_num_forward_passes"]
    Config.HyPLAN.NO_VERTICAL_PRUNING = args["no_vertical_pruning"]

    if Config.AGENT is Agent.HyPLAN and not Config.A2C.DROPOUT:
        raise ValueError("HyPLAN must use a dropout layer in its value head to obtain uncertainty estiamtes.")
    if Config.AGENT is Agent.HyPLAN and not Config.CALIBRATE_CONFIDENCE and Config.MODE is not Mode.TRAIN:
        raise ValueError("HyPLAN must use confidence calibration to obtain reliable uncertainty estimates.")

    if Config.HyPLAN.NUM_FORWARD_PASSES is not None and Config.AGENT is not Agent.HyPLAN:
        raise ValueError("Altering the number of forward passes is only available for HyPLAN.")
    if Config.HyPLAN.NUM_FORWARD_PASSES is None and Config.AGENT is Agent.HyPLAN:
        raise ValueError("The number of forward passes has to be specified for HyPLAN.")
    
    # LEADER
    Config.Leader.MAX_PEDESTRIANS = 1
    Config.Leader.ATTENTION_SAMPLING = args["attention_sampling"]

    if Config.AGENT is not Agent.LEADER and Config.Leader.ATTENTION_SAMPLING:
        raise ValueError("Attention based sampling is only available for LEADER.")
    
    '''
    # default values for agents
    if Config.AGENT is Agent.A2C or Config.AGENT is Agent.PPO:
        Config.REWARD_FUNCTION = "akash"

    if Config.AGENT is Agent.HyLEAP:
        Config.Despot.NOISE = 0.05
        Config.HyLEAP.HACKY = True
        Config.REWARD_FUNCTION = "despot"
        Config.Despot.CORRECT_TIMING = True
        Config.Despot.CORRECT_VELOCITY = True
        Config.A2C.MODEL_ARCHITECTURE = "florian"
        Config.A2C.HIDDEN_LAYER_SIZE = 256

    if Config.AGENT is Agent.LEADER:
        Config.Despot.NOISE = 0.05
        Config.REWARD_FUNCTION = "despot"
        Config.Despot.CORRECT_TIMING = True
        Config.Despot.CORRECT_VELOCITY = True

    if Config.AGENT is Agent.HyPLAN:
        Config.Despot.NOISE = 0.05
        Config.HyLEAP.HACKY = True
        Config.REWARD_FUNCTION = "despot"
        Config.A2C.MODEL_ARCHITECTURE = "florian"
        Config.A2C.HIDDEN_LAYER_SIZE = 256
        Config.Despot.CORRECT_VELOCITY = True
        Config.Despot.CORRECT_TIMING = True
        Config.Despot.IMPROVED_HEURISTIC = True
        Config.Despot.AGGRESSIVE_BELIEF_UPDATES = True
        Config.Despot.MINIMAL_NOISE = True
        Config.Despot.NO_IMPORTANCE_SAMPLING = True
        Config.Despot.NO_NORMALIZATION = True
    '''


    # =============
    # setup logging
    # =============  
    initialize_logging()
    log_debug(f"program executed with following arguments: {args}")

    # =================================================
    # START CARLA SERVER OR CONNECT TO RUNNING INSTANCE
    # =================================================
    # only start carla server when on slurm cluster
    if Config.Carla.REMOTE:
        log_info("running CARLA server on SLURM cluster")
        start_carla_server()
    else:
        Config.Carla.HOST = Address.LOCAL
        log_info("executing script on local machine: connecting to existing CARLA server instance")
    

    # ==============================================================
    #                   PYTHON CARLA INTERFACE
    # ==============================================================
    benchmark = CarlaCTS02()
    Config.Carla.CAR_INTENTION_IMAGE_WIDHT = '84' if Config.A2C.MODEL_ARCHITECTURE not in ["florian", "akash"] else '400'
    Config.Carla.CAR_INTENTION_IMAGE_HEIGHT = Config.Carla.CAR_INTENTION_IMAGE_WIDHT

    # ==============================================================
    #                       CREATE AGENT
    # ==============================================================
    if Config.AGENT is Agent.IS_DESPOT:
        agent = ISDespotP(benchmark.client, benchmark.client_world)
    elif Config.AGENT is Agent.HYP_DESPOT:
        agent = ISDespotP(benchmark.client, benchmark.client_world)
    elif Config.AGENT is Agent.A2C:
        if Config.A2C.PPO: agent = PPO(benchmark.client, benchmark.client_world)
        else: agent = A2C(benchmark.client, benchmark.client_world)
    elif Config.AGENT is Agent.HyLEAP:
        agent = HyLEAP(benchmark.client, benchmark.client_world)
    elif Config.AGENT is Agent.HyLEAR:
        raise NotImplementedError()
        #agent = HyLEAR_NavA2C(benchmark.client, benchmark.client_world)
    elif Config.AGENT is Agent.HyPLAN:
        agent = HyPLAN(benchmark.client, benchmark.client_world)
    elif Config.AGENT is Agent.LEADER:
        agent = LEADER(benchmark.client, benchmark.client_world)

    benchmark.set_agent(agent)

    # ===========================================================
    #                       PREPARE BENCHMARK
    # ===========================================================
    if Config.MODE is Mode.TRAIN:
        benchmark.prepare_train_episodes()
    if Config.MODE is Mode.VAL:
        benchmark.prepare_validation_episodes()
    elif Config.MODE is Mode.TEST:
        benchmark.prepare_test_episodes()
    if Config.A2C.PPO: agent.number_of_episodes = benchmark.number_of_episode

    # =======================
    # EXECUTE SIMULATION LOOP
    # =======================
    # determine from which episode to start (this happens when a job was interrupted due to reaching its time limit)
    episode_counter = Config.INITIAL_EPISODE
    log_info(f"{Config.MODE} on a total number of {benchmark.number_of_episode} episodes")
    # metrics that span all episodes
    goal_episodes = []
    ttg_episodes = []
    nearmiss_episodes = []
    crash_episodes = []
    rewards_episode = []
    execution_times_episodes = []
    steps_episodes = []
    skipped_steps_episodes = 0
    non_conclusive_episodes = 0
    # in GB
    initial_physical_memory = round(psutil.Process(os.getpid()).memory_info().rss / pow(1024, 3), 2)
    initial_virtual_memory = round(psutil.Process(os.getpid()).memory_info().vms / pow(1024, 3), 2)

    while episode_counter <= benchmark.number_of_episode:
        # metrics that are episode specific
        performance_log = {}
        episode_rewards = []
        episode_actions = []
        episode_execution_times = []
        episode_car_speeds = []
        episode_risks = [] 
        episode_skipped_steps = 0
        nearmiss = False

        # get the scenario id, parameters and instantiate the world
        benchmark.initialize_episode(episode_counter)
        # initial car position
        ego_vehicle_start_position = benchmark.client_world.ego_vehicle.get_location()
        pedestrian_start_position = benchmark.client_world.pedestrian.get_location()

        # episode simulation loop (step-wise incrementation)
        for step_counter in range(1, Config.MAX_EPISODE_STEPS + 1, 1):
            # check whether episode has termianted without reaching goal/causing collision
            non_conclusive = has_agent_stopped(episode_car_speeds, past_steps=Config.MAX_EPISODE_STEPS)
            # remember number of non-conclusive episodes
            non_conclusive_episodes += (non_conclusive == True)
            # non_conclusive flag is used in finalize_episode() for proper irregular clean-up
            if non_conclusive: break

            # render scene simulation in CARLA
            if Config.DISPLAY: benchmark.render()

            # start of episode 
            start_time = time.perf_counter()
            # ACTUALLY PERFORM THE SIMULATION STEP
            step_summary = benchmark.step(episode_counter, step_counter-episode_skipped_steps)
            time_taken = (time.perf_counter() - start_time)

            # did we skip this step because of buggy path planning? (ugly & hacky, but not my fault...)
            episode_skipped_steps += (step_summary["skipped_step"] == True)
            # do not remember step if so
            if not step_summary["skipped_step"]: 
                # track step execution time
                episode_execution_times.append(round(time_taken*1000, 4))
                # track step reward
                episode_rewards.append(step_summary["reward"])
                # track speed
                episode_car_speeds.append(round(velocity_kmh(benchmark.client_world.ego_vehicle), 4))
                # track agent actions
                episode_actions.append(step_summary["action"].value)
                ''' 
                log_info(
                    f"\nSTEP {step_counter}: execution time {round(time_taken*1000, 4)}, "
                    f"car velocity {velocity_kmh(benchmark.client_world.ego_vehicle):.4f}km/h, " 
                    f"action taken {'ACC' if int(step_summary['action'].value) == 2 else 'MAIN' if int(step_summary['action'].value) == 1 else 'DEC'}, "
                    f"reward. {step_summary['reward']:.6f}, "
                    f"pedestrian distance travelled: {distance_locations(pedestrian_start_position, benchmark.client_world.pedestrian.get_location()):.4f}, "
                    f"terminal {step_summary['terminal']}"
                )
                #'''

            # at most one nearmiss per episode
            # preserve nearmiss until end of episode
            nearmiss = nearmiss or (step_summary['nearmiss'] and velocity_kmh(benchmark.client_world.ego_vehicle) > 0.0)
            # break if goal reached unless recording pedestrian data because we need equally long episodes
            if step_summary["goal"] and not Config.RECORD_PEDESTRIAN_DATA: break
            # always terminate episode if collision occurred; discared during training of pedestrian path predictor
            if step_summary["collision"]: break
            # @@@ end step-loop @@@ #

        # check for non-conclusive episodes: maximum steps exhausted and non-terminal last state
        if step_counter == Config.MAX_EPISODE_STEPS and not step_summary["terminal"]:
            non_conclusive = True
            non_conclusive_episodes += 1

        ego_vehicle_travelled_distance = round(
            distance_locations(ego_vehicle_start_position, benchmark.client_world.ego_vehicle.get_location()), 2
        )
        # print debug information that might be useful in telling why a given episode hasn't terminated regularly
        if non_conclusive:
            # calculate averages for this particular episode
            avg_car_speed = round(np.mean(episode_car_speeds), 2)
            # calculate action distribution across steps
            episode_action_distribution = {}
            for key, val in Counter(episode_actions).items():
                if key == 0: verbose_key = "decelerate"
                elif key == 1: verbose_key = "maintain"
                else: verbose_key = "accelerate"
                percentage = round((val/len(episode_actions))*100, 2)
                episode_action_distribution.update({verbose_key:percentage})

            # additional debug information for episodes that have not been conclusive
            console_log = {
                "episode": episode_counter,
                "non_conclusive_episodes": non_conclusive_episodes,
                "episode_skipped_steps": episode_skipped_steps,
                "episode_travelled_distance": ego_vehicle_travelled_distance,
                "episode_car_speed_avg": avg_car_speed,
                "episode_action_distribution": episode_action_distribution
            }
            log_info(json.dumps(console_log, indent=2))

        # only calculate and show running statistics of episodes that have terminated in a crash/goal
        # as these are also not considered during the final performance evaluation  
        if not non_conclusive: 
            # update running averages across episodes
            steps_episodes.append(step_counter)
            goal_episodes.append(1 if step_summary['goal'] else 0)
            nearmiss_episodes.append(1 if nearmiss else 0)
            crash_episodes.append(1 if step_summary['collision'] else 0)
            rewards_episode.append(sum(episode_rewards))
            execution_times_episodes.append(round(np.mean(episode_execution_times), 4))
            skipped_steps_episodes += episode_skipped_steps
            if step_summary['goal']: ttg_episodes.append(step_counter * Config.Carla.SIMULATION_STEP)

            # show running averages during execution on console
            console_log = {
                "episode": episode_counter,
                "non_conclusive_episodes": non_conclusive_episodes,
                "skipped_steps": skipped_steps_episodes,
                "running_steps_avg": round(np.mean(steps_episodes)),
                "running_crash_avg": round(np.mean(crash_episodes), 4),
                "runing_nearmiss_avg": round(np.mean(nearmiss_episodes), 4),
                "running_goal_avg": round(np.mean(goal_episodes), 4),
                "running_ttg_avg": round(np.mean(ttg_episodes), 4),
                "running_reward_avg": round(np.mean(rewards_episode), 4),
                "running_execution_time_avg": round(np.mean(execution_times_episodes), 4)
            }
            log_info(json.dumps(console_log, indent=2))


        current_physical_memory = round(psutil.Process(os.getpid()).memory_info().rss / pow(1024, 3), 2)
        current_virtual_memory = round(psutil.Process(os.getpid()).memory_info().vms / pow(1024, 3), 2)
        # debug information
        log_debug(json.dumps({
            "initial_physical_memory (GB)": initial_physical_memory,
            "current_physical_memory (GB)": current_physical_memory,
            "physical_memory_increase (%)": round((current_physical_memory/initial_physical_memory)*100.0 - 100.0, 2),
            "initial_virtual_memory (GB)": initial_virtual_memory,
            "current_virtual_memory (GB)": current_virtual_memory,
            "virtual_memory_increase (%)": round((current_virtual_memory/initial_virtual_memory)*100.0 - 100.0, 2)
        }, indent=2))

        # evaluate episode statistics (crash rate, nearmiss rate, time to goal, smoothness, execution time, violations)
        performance_log = {
            "scenario": step_summary['scenario'],
            "episode": episode_counter,
            "ped_distance": step_summary['ped_distance'],
            "ped_speed": step_summary['ped_speed'],
            "collision": 1 if step_summary['collision'] else 0,
            "nearmiss": 1 if nearmiss else 0,
            "goal": 1 if step_summary['goal'] else 0,
            "ttg": round(step_counter * Config.Carla.SIMULATION_STEP, 4) if step_summary['goal'] else "nan",
            "total_episode_reward": sum(episode_rewards),
            "travelled_distance": ego_vehicle_travelled_distance,
            "execution_times": episode_execution_times,
            # allows us to analyze why some episodes are terminated early
            "non_conclusive": non_conclusive,
            "skipped_steps": episode_skipped_steps,
        }
        # log classical performance metrics
        log_performance_metrics(performance_log)

        if Config.RECORD_PEDESTRIAN_DATA and not step_summary['collision']:
            # log ego-vehicle and pedestrian trajectory
            log_data({
                "scenario": step_summary['scenario'],
                "episode": episode_counter,
                "pedestrian_trajectory": agent.episode_pedestrian_past_trajectory,
                "ego_vehicle_trajectory": agent.episode_ego_vehicle_past_trajectory
            })

        benchmark.finalize_episode(episode_counter, non_conclusive, step_counter-episode_skipped_steps)
        episode_counter += 1
        # @@@ end episode-loop @@@ #

    Config.TERMINATE = True
    kill_all_processes(kill_parent=False)



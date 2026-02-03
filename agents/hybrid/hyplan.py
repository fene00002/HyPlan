import os
import sys
import csv
import time
import json
import glob
from contextlib import nullcontext
from typing import List
from threading import Thread
import multiprocessing as mp

import cv2
import torch
from torch.distributions import Categorical
import numpy as np
from scipy.stats import norm

from agents.planner.isdespotp import ISDespotP
from agents.learner.models import A2C
from agents.hybrid.hyleap import run_car_intention_generation_process
from utils.config import Config, Mode
from utils.connector import DespotBridge, Connection
from utils.utils import load_model, save_model
from utils.logger import log_debug, log_info, log_exception, log_learning_metrics


class HyPLAN(ISDespotP):

    def __init__(self, client, client_world):
        log_info("initializing HyPLAN agent...")    
        super(HyPLAN, self).__init__(client, client_world)

        #====================================================================#
        #                            TRAINING                                #
        #====================================================================#       
        # episode buffers of neural network's predictions for training
        self.episode_belief_state_value_estimates: List[torch.Tensor] = []
        self.episode_predicted_policies: List[torch.Tensor] = []
        # used for tracking learning progress
        self.episode_action_distributions: List[torch.Tensor] = []
        # define losses with reduction="none" so we can check each loss individually for validity, i.e. > 0
        self.actor_loss_function = torch.nn.CrossEntropyLoss(reduction="none")
        self.critic_loss_function = torch.nn.MSELoss(reduction="none")
        # min/max variance (uncertainty estimates) for scaling during testing
        self.min_variance = sys.float_info.max
        self.max_variance = sys.float_info.min

        #====================================================================#
        #                            VALIDATION                              #
        #====================================================================# 
        # used for calibrating model on validation data set
        self.episode_predictive_mean: List[torch.Tensor] = []
        self.episode_predictive_variance: List[torch.Tensor] = []

        #====================================================================#
        #                              TESTING                               #
        #====================================================================# 
        # used for calculating calibration error on test set
        self.confidence_levels = np.arange(0.01, 1, 0.01).round(2)
        self.episode_uncalibrated_percentiles: List[torch.Tensor] = []
        self.episode_calibrated_percentiles: List[torch.Tensor] = []
        # used to track the development of uncertainty for an episode
        self.episode_uncertainty_estimates: List[float] = []

        #====================================================================#
        #                             SETUP/MISC                             #
        #====================================================================# 
        # must use cuda
        if not torch.cuda.is_available(): raise ValueError("No CUDA capable GPU found.")
        # model
        self.model: A2C = None
        # define optimizer
        self.optimizer = None
        # used for parallelizing car intention image generation as a single image can take up to 10ms to create
        self.car_intention_generation_workers: List[mp.Process] = []
        self.message_queues: List[mp.Queue] = []
        # car intention of the root node of each DESPOT associated with the current simulation step (used for debugging)
        self.root_node_car_intention: np.ndarray = None
        # current planning depth of IS-DESPOT in this step 
        self.step_despot_depth: int = 0
        # because the functions of hyleap & hyplan are not directly called by the controller
        self.step_counter: int = 0
        # init connections here so that controller.py waits for connections to be established
        self.establish_evaluation_connection()
        # prepare worker processes
        self.initialize_car_intention_generation_workers()
        # do not run as process to avoid having to communicate data/model params
        # speed up would only be marginal
        Thread(target=self.run_hyplan_server).start()


    # ================================================================================================= #
    #               METHODS FOR CONTROLLING IS-DESPOT C++ PROCESS INTERACTION                           #
    # ================================================================================================= #
    def establish_evaluation_connection(self):
        if self.despot_connection is None:
            raise TypeError("IS-DESPOT C++ process interface has not been initialized.")
        if not isinstance(self.despot_connection, DespotBridge):
            raise TypeError(
                f"Invalid IS-DESPOT C++ prcess interface: Expected '{DespotBridge}', got '{type(self.despot_connection)}'."
            )
        
        self.despot_connection.establish_connection(Connection.HyPLAN_EVALUATION)


    # ================================================================================================= #
    #                                       HyPLAN SETUP METHODS                                        #
    # ================================================================================================= #
    def initialize_model(self):
        if self.model is not None:
            raise TypeError("Exisitng HyPLAN model cannot be overwritten.")
        log_info(f"initializing A2C model...")

        self.model = A2C(hidden_dim=Config.A2C.HIDDEN_LAYER_SIZE, use_dropout=True).double().cuda()
        # if we are not training, we are automatically calibrating confidence
        if Config.MODE is not Mode.TRAIN: 
            self.model.eval()
            self.model.keep_dropout_active()
        log_info(self.model)
                

    def initialize_optimizer(self):
        if self.optimizer is not None:
            raise TypeError("Exisitng optimizer cannot be overwritten.")
        if not isinstance(self.model, A2C):
            raise ValueError(f"Invalid HyPLAN model type: Expected '{A2C}', got '{type(self.model)}'.")
        log_info("initializing optimizer...")

        # initialize optimizers with same paramaters as HyLEAP
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(), 
            lr=Config.HyLEAP.LEARNING_RATE, 
            alpha=Config.HyLEAP.DECAY,
            momentum=Config.HyLEAP.MOMENTUM, 
            eps=Config.HyLEAP.EPSILON, 
            weight_decay=Config.HyLEAP.L2_DECAY
        )


    def initialize_car_intention_generation_workers(self):
        if len(self.car_intention_generation_workers) != 0:
            raise ValueError("Car intention generation processes already initialized.")
        if os.cpu_count() < 12:
            raise NotImplementedError("Invalid number of CPU cores. Fixed implementation requires at least 12.")
        log_info("initializing car intention generation workers...")

        # create as many processes as possible - (this script process + IS-DESPOT process + CARLA simulation server process)
        number_of_workers = os.cpu_count()-3
        if number_of_workers > 12: number_of_workers = 12
        # local script execution is just for debugging
        if not Config.Carla.REMOTE: number_of_workers = 1
        # start workers
        for cpu_core_id in range(number_of_workers):
            # fork instead of spawning because latter does some funky stuff when threads are created (c.f. run_hyleap_server())
            master_to_slave = mp.get_context('spawn').Queue()
            slave_to_master = mp.get_context('spawn').Queue()
            process = mp.get_context('spawn').Process(
                target=run_car_intention_generation_process, 
                args=(
                    self.birdview_car_intention_producer, 
                    master_to_slave,
                    slave_to_master,
                    Config.DISPLAY,
                )
            )
            process.start()
            log_info(f"started worker process with ID: {cpu_core_id}")
            self.message_queues.append((master_to_slave, slave_to_master))
            self.car_intention_generation_workers.append(process) 

            
    # ================================================================================================= #
    #                                        HyPLAN CORE METHODS                                        #
    # ================================================================================================= #
    def initialize_episode(self, episode_counter: int, scenario: tuple):
        self.step_counter = 0
        # clear training buffers
        self.episode_belief_state_value_estimates.clear()
        self.episode_predicted_policies.clear()
        self.episode_action_distributions.clear()
        # clear confidence calibration buffers for validating and testing
        self.episode_predictive_mean.clear()
        self.episode_predictive_variance.clear()
        self.episode_uncalibrated_percentiles.clear()
        self.episode_calibrated_percentiles.clear()
        self.episode_uncertainty_estimates.clear()
        return super().initialize_episode(episode_counter, scenario)
    

    def finalize_episode(self, episode_counter: int, non_conclusive: bool, step_counter: int):
        # save undsicounted episodic rewards as debug statistic
        discounted_episode_rewards = self.episode_rewards.copy()
        # delete first dummy reward entry (defaults to 0.0)
        del discounted_episode_rewards[0]
        # rewards used for training: must contain the final reward received for the terminal state
        N = len(discounted_episode_rewards) - 1
        # propagate rewards backward in time
        for i in range(1, N + 1):
            # the current reward is equal to itself + the discounted reward of the next steps
            discounted_episode_rewards[N - i] += Config.A2C.DISCOUNT * discounted_episode_rewards[N - i + 1]
        # shape: [steps, 1]
        returns = torch.tensor(discounted_episode_rewards, dtype=torch.float64, device=Config.DEVICE)
        # rewards vector is normalised to have 0 mean and 1 std
        if Config.A2C.STANDARDIZE_RETURN:
            returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float64).eps.item())  

        # regular epsiode clean-up
        if Config.MODE is Mode.TRAIN:
            # sanity check for consistent episode memory
            if returns.shape[0] != step_counter:
                raise ValueError(
                    f"Inconsistent episode memory: Number of steps {step_counter} "
                    f"vs. returns {returns.shape[0]}."
                )
            if len(self.episode_belief_state_value_estimates) != step_counter:
                raise ValueError(
                    f"Inconsistent episode memory: Number of steps {step_counter}" 
                    f" vs. belief state value estimates {len(self.episode_belief_state_value_estimates)}."
                )
            if len(self.episode_despot_policies) != step_counter:
                raise ValueError(
                    f"Inconsistent episode memory: Number of steps {step_counter}" 
                    f" vs. IS-DESPOT policies {len(self.episode_predicted_policies)}."
                )
            if len(self.episode_predicted_policies) != step_counter:
                raise ValueError(
                    f"Inconsistent episode memory: Number of steps {step_counter}" 
                    f" vs. NN's predicted policies {len(self.episode_predicted_policies)}."
                )
            if len(self.episode_action_distributions) != step_counter:
                raise ValueError(
                    f"Inconsistent episode memory: Number of steps {step_counter} " 
                    f"vs. action distributions {len(self.episode_action_distributions)}."
                )                
            # train on this episode
            self.training_iteration(episode_counter, step_counter, returns)
            # model checkpointing
            if episode_counter % Config.MODEL_SAVE_FREQUENCY_EPISODES == 0:
                save_model(
                    model_state_dict=self.model.state_dict(),
                    model_dir=Config.MODEL_DIR,
                    checkpoint=str(episode_counter),
                    name="hyplan"
                )
            # always save most recent model
            save_model(
                model_state_dict=self.model.state_dict(),
                model_dir=Config.MODEL_DIR,
                checkpoint="latest",
                name="hyplan"
            )
            # save min/max variance in case of training gets interrupted/resumed
            with open(os.path.join(Config.DATA_DIR, Config.HyPLAN.MIN_MAX_VARIANCE_FILENAME), "w") as file:
                file.write(json.dumps({"min_variance": self.min_variance, "max_variance": self.max_variance}))

        elif Config.CALIBRATE_CONFIDENCE:
            # calculate empirical error distribution
            if Config.MODE is Mode.VAL:
                # sanity check for consistent episode memory
                if returns.shape[0] != step_counter:
                    raise ValueError(
                        f"Inconsistent episode memory: Number of steps {step_counter} " 
                        f"vs. returns {returns.shape[0]}."
                    )
                if len(self.episode_predictive_mean) != step_counter:
                    raise ValueError(
                        f"Inconsistent episode memory: Number of steps {step_counter} " 
                        f"vs. predicted means {len(self.episode_predictive_mean)}."
                    )
                if len(self.episode_predictive_variance) != step_counter:
                    raise ValueError(
                        f"Inconsistent episode memory: Number of steps {step_counter} " 
                        f"vs. predicted standard deviations {len(self.episode_predictive_variance)}."
                    )
                log_info(f"saving Z-scores for episode {self.episode_counter} in '{Config.ERROR_DISTRIBUTION_DIR}'.")
                predictive_mean = torch.vstack(self.episode_predictive_mean).squeeze()
                predictive_stddev = torch.vstack(self.episode_predictive_variance).squeeze()
                Z = (returns - predictive_mean) / predictive_stddev
                torch.save( 
                    Z, os.path.join(
                        Config.ERROR_DISTRIBUTION_DIR, 
                        f"Z-scores_S-{self.scenario_id}_episode_{self.episode_counter}.pt"
                    )
                )

            # calculate empirical frequency of percentiles
            if Config.MODE is Mode.TEST:
                if returns.shape[0] != step_counter:
                    raise ValueError(
                        f"Inconsistent episode memory: Number of steps {step_counter} " 
                        f"vs. returns {returns.shape[0]}."
                    )
                if len(self.episode_belief_state_value_estimates) != step_counter:
                    raise ValueError(
                        f"Inconsistent episode memory: Number of steps {step_counter}" 
                        f" vs. belief state value estimates {len(self.episode_belief_state_value_estimates)}."
                    )
                if len(self.episode_uncertainty_estimates) != step_counter:
                    raise ValueError(
                        f"Inconsistent episode memory: Number of steps {step_counter}" 
                        f" vs. uncertainty estimates {len(self.episode_uncertainty_estimates)}."
                    )
                if len(self.episode_uncalibrated_percentiles) != step_counter:
                    raise ValueError(
                        f"Inconsistent episode memory: Number of steps {step_counter} " 
                        f"vs. uncalibrated percentiles {len(self.episode_uncalibrated_percentiles)}."
                    )
                if len(self.episode_calibrated_percentiles) != step_counter:
                    raise ValueError(
                        f"Inconsistent episode memory: Number of steps {step_counter} " 
                        f"vs. calibrated percentiles {len(self.episode_calibrated_percentiles)}."
                    )
                if len(self.episode_ego_vehicle_speeds) != step_counter:
                    raise ValueError(
                        f"Inconsistent episode memory: Number of steps {step_counter} " 
                        f"vs. ego-vehicle speeds {len(self.episode_ego_vehicle_speeds)}."
                    )
                if len(self.episode_ego_vehicle_pedestrian_distance) != step_counter:
                    raise ValueError(
                        f"Inconsistent episode memory: Number of steps {step_counter} " 
                        f"vs. ego-vehicle/pedestrian distances {len(self.episode_ego_vehicle_pedestrian_distance)}."
                    )
                if len(self.episode_pedestrian_visibility) != step_counter:
                    raise ValueError(
                        f"Inconsistent episode memory: Number of steps {step_counter} " 
                        f"vs. pedestrian visibility {len(self.episode_pedestrian_visibility)}."
                    )
                log_info(
                    f"saving (un-)calibrated frequencies for scenario: "
                    f"{self.scenario_id} and episode: {self.episode_counter}"
                )
                #====================================================================#
                #                  UNCERTAINTY VS. EXPLAINED VARIANCE                #
                #====================================================================#
                episode_mean_uncertainty = np.mean(self.episode_uncertainty_estimates)
                episode_explained_variance = self.calculate_explained_variance(
                    torch.vstack(self.episode_belief_state_value_estimates).squeeze().detach().cpu().numpy(), 
                    returns.detach().cpu().numpy()
                )
                episode_mse = torch.nn.functional.mse_loss(
                    torch.vstack(self.episode_belief_state_value_estimates).squeeze().detach(), returns.detach()
                )
                with open(
                    os.path.join(
                        Config.DATA_DIR, 
                        Config.HyPLAN.UNCERTAINTY_ANALYSIS_DIR, 
                        f"S-{self.scenario_id}_uncertainty_vs_explained_variance.csv"
                    ), 
                    mode='a', newline='', encoding='utf-8'
                ) as file:
                    csv.writer(file, delimiter=",").writerow(
                        [self.scenario_id, episode_mean_uncertainty, episode_explained_variance, episode_mse.cpu().item()]
                    )
                #====================================================================#
                #  UNCERTAINTY VS. SCENES (PEDESTRIAN VELOCITY + CROSSING DISTANCE)  #      
                #====================================================================#       
                with open(
                    os.path.join(
                        Config.DATA_DIR, 
                        Config.HyPLAN.UNCERTAINTY_ANALYSIS_DIR, 
                        f"S-{self.scenario_id}_uncertainty_per_scene.csv"
                    ), mode='a', newline='', encoding='utf-8'
                ) as file:
                    csv.writer(file, delimiter=",").writerow(
                        [self.scenario_id, self.pedestrian_velocity, self.pedestrian_crossing_distance, episode_mean_uncertainty]
                    )                
                #====================================================================#
                #                        UNCERTAINTY CALIBRATION                     #
                #====================================================================# 
                # expand shape from [#steps, ] to [#steps, 99]
                returns = returns.unsqueeze(1).expand(-1, 99)
                # shape: [#steps, 99]
                episode_uncalibrated_percentiles = torch.vstack(self.episode_uncalibrated_percentiles)
                episode_calibrated_percentiles = torch.vstack(self.episode_calibrated_percentiles)
                # check whether the discounted reward is less than each (un-)calibrated percentile
                empirical_frequency_uncalibrated = (returns <= episode_uncalibrated_percentiles).int()
                empirical_frequency_calibrated = (returns <= episode_calibrated_percentiles).int()
                # save for plotting calibration plots and calcluating calibration error off-line
                torch.save(empirical_frequency_uncalibrated, os.path.join(
                    Config.UNCALIBRATED_ECDF_DIR,
                    f"empirical_frequency_S-{self.scenario_id}_episode_{self.episode_counter}.pt")
                )
                torch.save(
                    empirical_frequency_calibrated, os.path.join(
                        Config.CALIBRATED_ECDF_DIR,
                        f"empirical_frequency_S-{self.scenario_id}_episode_{self.episode_counter}.pt"
                    )
                )
                #====================================================================#
                #                    UNCERTAINTY VS CRITICALITY                      #
                #====================================================================#
                uncertaint_criticality_episode_log = {
                    "scenario": self.scenario_id,
                    "episode_uncertainty": self.episode_uncertainty_estimates,
                    "episode_total_velocities": (self.episode_ego_vehicle_speeds + self.pedestrian_velocity).tolist(),
                    "episode_ego_vehicle_pedestrian_distance": self.episode_ego_vehicle_pedestrian_distance,
                    "episode_pedestrian_visibility": self.episode_pedestrian_visibility
                }
                with open(
                    os.path.join(
                        Config.DATA_DIR, 
                        Config.HyPLAN.UNCERTAINTY_ANALYSIS_DIR, 
                        f"S-{self.scenario_id}_uncertainty_vs_criticality.json"
                    ), "a"
                ) as file:
                    file.write(json.dumps(uncertaint_criticality_episode_log))
                    file.write("\n")                
                #====================================================================#
                #                        UNCERTAINTY VS STEPS                        #
                #====================================================================# 
                # make lists of uniform length, pad using nans
                self.episode_uncertainty_estimates += [float('nan')] * (Config.MAX_EPISODE_STEPS - len(self.episode_uncertainty_estimates))
                # first and last occurrence calculation only applicable for one pedestrian
                if Config.Carla.NUM_PEDESTRIANS != 1:
                    raise ValueError(f"Invalid number of pedestrians: Expected 1, got {Config.Carla.NUM_PEDESTRIANS}.")
                # first scene simulation step where pedestrian was visible
                if True in self.episode_pedestrian_visibility:
                    self.episode_uncertainty_estimates.append(
                        self.episode_pedestrian_visibility.index(True)
                    )
                    # last scene simulation step where pedestrian was visible
                    # only applicable if agent reached goal (for collisions, pedestrian is always visible for last step)
                    self.episode_uncertainty_estimates.append(
                        len(self.episode_pedestrian_visibility) - 1 - self.episode_pedestrian_visibility[::-1].index(True)
                    )
                    # add whether episode ended in collision or not
                    self.episode_uncertainty_estimates.append(
                        -1.0 if self.is_ego_vehicle_in_collision else 1.0
                    )         
                    # save uncertainty estimates (+ auxiliary information) for episode                
                    with open(
                        os.path.join(
                            Config.DATA_DIR, 
                            Config.HyPLAN.UNCERTAINTY_ANALYSIS_DIR, 
                            f"S-{self.scenario_id}_uncertainty_per_step.csv"
                        ), mode='a', newline='', encoding='utf-8'
                    ) as file:
                        csv.writer(file, delimiter=",").writerow(self.episode_uncertainty_estimates)
                    
        # always called
        return super().finalize_episode(episode_counter, non_conclusive, step_counter)
    

    # recycles predictions made during inference and essentially only calculates the loss
    def training_iteration(self, episode_counter: int, step_counter: int, returns: torch.Tensor):
        if not isinstance(returns, torch.Tensor):
            raise ValueError(f"Invalid returns type: Expected '{torch.Tensor}', got {type(returns)}.")
        
        # required for getting a proper time-reading of cuda related tasks
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        # shape: [steps, ]
        belief_state_value_estimates = torch.vstack(self.episode_belief_state_value_estimates).squeeze()
        # shape: [steps, 3]
        predicted_policies = torch.vstack(self.episode_predicted_policies)
        # shape: [steps, 3]
        despot_policies = torch.tensor(self.episode_despot_policies, dtype=torch.float64, device=Config.DEVICE)

        # calculate actor loss
        actor_loss_episode = self.actor_loss_function(predicted_policies, despot_policies)
        for loss in actor_loss_episode.tolist():
            if loss < 0: raise ValueError(
                f"Invalid actor loss: By definition '{self.actor_loss_function}' can not be negative, but returned {loss:.4f}."
            )
        actor_loss = actor_loss_episode.mean()
        # calculate critic loss
        critic_loss_episode = self.critic_loss_function(belief_state_value_estimates, returns)
        for loss in critic_loss_episode.tolist():
            if loss < 0: raise ValueError(
                f"Invalid critic loss: By definition '{self.critic_loss_function}' can not be negative, but returned {loss:.4f}"
            )
        critic_loss = critic_loss_episode.mean()

        # get total loss
        total_loss = actor_loss + critic_loss
        # debug information
        torch.cuda.synchronize()
        log_info(
            f"total loss for episode {episode_counter}: {total_loss.item():.4f} "
            f"calculated in {(time.perf_counter() - start_time)*1000:.4f}ms"
        )

        # reset gradients
        self.model.zero_grad()
        # time backward pass
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        # propagate gradient backwards
        total_loss.backward()
        # calculate l2-norm of gradients after backward pass
        grads = [param.grad.detach().flatten() for param in self.model.parameters() if param.grad is not None]
        grad_norm = torch.cat(grads).norm().cpu()
        # gradient clipping
        if Config.A2C.CLIP_GRADIENT: torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        # update weights
        self.optimizer.step()
        # debug information
        torch.cuda.synchronize()
        log_info(
            f"performed backward pass for episode {episode_counter} in "\
            f"{((time.perf_counter() - start_time)*1000):.4f}ms"
        )

        # reconstruct log probabilities using learner's action distributions and planner's actions
        episode_log_probabilities = []
        # skip dummy action (defaults to MAINTAIN)
        for learner_action_distribution, planner_action in zip(self.episode_action_distributions, self.episode_actions[1:]):
            episode_log_probabilities.append(
                learner_action_distribution.log_prob(torch.tensor(planner_action.value, dtype=torch.int64, device=Config.DEVICE)) 
            )
        episode_log_probabilities = torch.vstack(episode_log_probabilities).squeeze()

        # reconstruct mean entropy of learner's action distributions
        episode_entropies = torch.vstack([
            learner_action_distribution.entropy() for learner_action_distribution in self.episode_action_distributions
        ]).squeeze().mean()

        learning_log = {
            "iteration": episode_counter,
            "actor_loss": actor_loss.detach().cpu().squeeze().item(),
            "critic_loss": critic_loss.detach().cpu().squeeze().item(),
            # sum of undiscounted rewards
            "reward": sum(self.episode_rewards),
            "entropy": episode_entropies.detach().cpu().item(),
            "explained_variance": self.calculate_explained_variance(
                belief_state_value_estimates.detach().cpu().numpy(), returns.detach().cpu().numpy()
            ),
            # approximate the kullback-leibler divergence between the previous and current policies
            "kld": self.approximate_kullback_leibler_divergence(episode_log_probabilities, step_counter),
            "grad_norm": grad_norm.squeeze().item(),
            "variance_avg": np.mean(self.episode_predictive_variance)
        }
        # print learning log
        log_info(json.dumps(learning_log, indent=2))
        # track learning progress
        log_learning_metrics(learning_log)


    def calculate_explained_variance(self, value_estimates: np.ndarray, value_true: np.ndarray) -> float:
        variance_value_true = np.var(value_true)
        if variance_value_true == 0: return np.nan
        else: return 1 - np.var(value_true - value_estimates) / variance_value_true


    # calculate the relative difference between policies after a training iteration 
    # (i.e. how quickly does our policy change?)
    # used by PPO
    def approximate_kullback_leibler_divergence(self, episode_log_probabilities: torch.Tensor, step_counter: int) -> float:
        # terminal reward is not used
        del self.episode_rewards[-1]
        if len(self.episode_rewards) != step_counter:
            raise ValueError(
                f"Inconsistent episode memory: Expected {step_counter} rewards, got {len(self.episode_rewards)}."
            )
        # dummy and terminal actions are used
        if len(self.episode_actions) != step_counter + 1:
            raise ValueError(
                f"Inconsistent episode memory: Expected {step_counter} actions, got {len(self.episode_actions)}."
            )
        # because step_counter starts at 1
        if len(self.episode_ego_vehicle_speeds) != step_counter:
            raise ValueError(
                f"Inconsistent episode memory: Expected {step_counter} ego vehicle speeds, "
                f"got {len(self.episode_ego_vehicle_speeds)}."
            )
        if len(self.episode_birdview_car_intentions) != step_counter:
            raise ValueError(
                f"Inconsistent episode memory: Expected {step_counter} car intention images, "
                f"got {len(self.episode_birdview_car_intentions)}."
            )       

        # calculate KLD between previous and current policy
        with torch.no_grad():
            # log probabilities of the new policy after training on the current episode
            new_policy_log_probabilities = []
            # DUMMY initial inputs for first LSTM cell
            previous_lstm_hidden_state_first = torch.zeros((1, 64), dtype=torch.float64, device=Config.DEVICE)
            previous_lstm_cell_state_first = torch.zeros((1, 64), dtype=torch.float64, device=Config.DEVICE)
            # all observation at any given depth have the same previous lstm state, because they all share the same parent
            previous_lstm_hidden_state_second = torch.zeros(
                (1, Config.HyLEAP.LSTM_STATE_SIZE//2), 
                dtype=torch.float64, 
                device=Config.DEVICE
            )
            previous_lstm_cell_state_second = torch.zeros(
                (1, Config.HyLEAP.LSTM_STATE_SIZE//2), 
                dtype=torch.float64, 
                device=Config.DEVICE
            )
            # we must make sequential calls to the model in order for the lstm state to be correctly updated
            # a batched forward pass doesn't treat the individual samples as pre-/successors
            for step in range(step_counter):
                # observation encountered during policy rollout
                car_intention = torch.tensor(
                    self.episode_birdview_car_intentions[step], dtype=torch.float64, device=Config.DEVICE
                ).transpose(-1, 0).unsqueeze(0)
                # previous reward
                auxiliary_input_first = torch.tensor(
                    (self.episode_rewards[step]), dtype=torch.float64, device=Config.DEVICE
                ).expand(1, 1)
                # current speed and previous action
                auxiliary_input_second = torch.tensor(
                   (self.episode_ego_vehicle_speeds[step], self.episode_actions[step].value),
                    dtype=torch.float64,
                    device=Config.DEVICE
                ).unsqueeze(0)
                # all-in-one forward pass without multiple stochastic forward passes
                action_logits, value, \
                (previous_lstm_hidden_state_first, previous_lstm_cell_state_first), \
                (previous_lstm_hidden_state_second, previous_lstm_cell_state_second) = self.model(
                    car_intention, 
                    previous_lstm_hidden_state_first, previous_lstm_cell_state_first,  
                    previous_lstm_hidden_state_second, previous_lstm_cell_state_second,
                    auxiliary_input_first, auxiliary_input_second
                )
                # re-calculate log probabilities BUT based on the same action executed in the initial policy rollout
                action_distribution = Categorical(logits=action_logits)
                new_policy_log_probabilities.append(
                    action_distribution.log_prob(
                        # current action as selected by the policy during rollout (i.e. skips the dummy action)
                        # i.e. in the first iteration this is equal to the first action that was actually executed
                        # by the agent during its initial policy rollout
                        torch.tensor(self.episode_actions[step + 1].value, dtype=torch.int64, device=Config.DEVICE)
                    )
                )

            # log probs for this epoch as well as ratio to log probs during initial policy rollout
            new_policy_log_probabilities = torch.vstack(new_policy_log_probabilities).squeeze()
            log_probabilities_ratio = new_policy_log_probabilities - episode_log_probabilities
            log_probabilities_ratio_exp = log_probabilities_ratio.exp()

            # calculate approximate kullback leibler divergence http://joschu.net/blog/kl-approx.html
            approximate_kld = ((log_probabilities_ratio_exp - 1) - log_probabilities_ratio).mean()
            return approximate_kld.detach().cpu().squeeze().item()


    def inference_iteration(self):
        # receive newly expanded nodes from IS-DESPOT
        previous_lstm_hidden_state, \
        previous_lstm_cell_state, \
        nodes,\
        agent_vehicle_past_simulated_trajectories, \
        is_root_node = self.despot_connection.receive_expanded_nodes(Connection.HyPLAN_EVALUATION)

        # for debugging
        if is_root_node:
            self.step_counter += 1
            self.despot_search_depth = 0
        else:
            self.despot_search_depth += 1

        number_of_samples = len(nodes)
        # obtain current simulation state of ego vehicle and pedestrian in CARLA
        agent_vehicle_transform = self.client_world.ego_vehicle.get_transform() 
        pedestrian_transform = self.client_world.pedestrian.get_transform()

        # list of car intentions
        observations = []
        # modify CARLA world state according to IS-DESPOT's simulation for each received observation
        # this general case also correctly captures the special case when a node is a root node
        car_intention_start = time.perf_counter()
        for index in range(number_of_samples):
            # obtain current node
            node = nodes[index]                
            # add waypoints to past trajectory that have already been visited in IS-DESPOT's planning simulation
            modified_ego_vehicle_past_trajectory = self.episode_ego_vehicle_past_trajectory.copy() + \
                                                   agent_vehicle_past_simulated_trajectories[index]
            # prepare message for worker process
            observation = {
                "scenario_id": self.scenario_id, # required for loading pre-drawn parked car canvas
                "agent_vehicle_x": node[0],
                "agent_vehicle_y": node[1],
                "agent_vehicle_z": agent_vehicle_transform.location.z,
                "agent_vehicle_pitch": agent_vehicle_transform.rotation.pitch,
                "agent_vehicle_yaw": node[2],
                "agent_vehicle_roll": agent_vehicle_transform.rotation.roll,
                "pedestrian_x": node[3],
                "pedestrian_y": node[4],
                "pedestrian_z": pedestrian_transform.location.z,
                "pedestrian_pitch": pedestrian_transform.rotation.pitch,
                "pedestrian_yaw": pedestrian_transform.rotation.yaw,
                "pedestrian_roll": pedestrian_transform.rotation.roll,
                "agent_vehicle_past_trajectory": modified_ego_vehicle_past_trajectory,
                "agent_vehicle_future_trajectory": self.step_ego_vehicle_future_trajectory,
                "pedestrian_future_trajectory": self.step_pedestrian_future_trajectory,
                # debug information when displaying
                "initial_position": f"{self.episode_ego_vehicle_start_position}",
                "goal_position": f"{self.episode_ego_vehicle_goal_position}",
                "step_counter": self.step_counter,
                "search_depth": self.despot_search_depth,
                "agent_vehicle_velocity": node[5] * 3.6, # for km/h
                "previous_reward": node[5],
                "previous_action": node[6]
            }
            # cycle through workers
            index = index % len(self.car_intention_generation_workers)
            # send observation in master_to_slave queue
            self.message_queues[index][0].put(observation)             

        # collect car intention images
        for index in range(number_of_samples):
            # cycle through workers
            index = index % len(self.car_intention_generation_workers)
            # get generated car intention from slave_to_master queue
            car_intention = self.message_queues[index][1].get(True)
            observations.append(car_intention)
            # update root node car intention (for debugging)
            if is_root_node: 
                self.root_node_car_intention = car_intention
                # save root node car intention images for calcualting episodic kbd
                self.episode_birdview_car_intentions.append(self.root_node_car_intention)

        log_debug(
            f"constructed car-intention images for {len(nodes)} nodes in "
            f"{(time.perf_counter()-car_intention_start)*1000:.4f}ms"
        )

        with torch.no_grad() if Config.MODE in [Mode.VAL, Mode.TEST] else nullcontext():
            # construct tensor of all car intention images generated at the current planning depth of IS-DESPOT
            observations = torch.tensor(
                np.array(observations), dtype=torch.float64, device=Config.DEVICE
            ).transpose(1, -1)
            # DUMMY initial inputs for first LSTM cell
            previous_lstm_hidden_state_first = torch.zeros((1, 64), dtype=torch.float64, device=Config.DEVICE)
            previous_lstm_cell_state_first = torch.zeros((1, 64), dtype=torch.float64, device=Config.DEVICE)
            # all observation at any given depth have the same previous lstm state, 
            # because they all share the same parent
            previous_lstm_hidden_state_second = torch.tensor(
                previous_lstm_hidden_state, 
                dtype=torch.float64, 
                device=Config.DEVICE
            )
            previous_lstm_cell_state_second = torch.tensor(
                previous_lstm_cell_state, 
                dtype=torch.float64, 
                device=Config.DEVICE
            )
            # construct feature tensor, which is fed to the LSTM layer in addition to the convolved observation
            # for each step t it consists of the car's current speed v_t, the last executed action a_t-1 
            # (as determined by IS-DESPOT) and the reward received for the previous simulation step r_t-1.
            auxiliary_input_first = torch.tensor(
                nodes[:, 7].reshape(number_of_samples, 1), dtype=torch.float64, device=Config.DEVICE
            )
            auxiliary_input_second = torch.tensor(
                np.concatenate((nodes[:, 5].reshape(number_of_samples, 1),
                                nodes[:, 6].reshape(number_of_samples, 1)), axis=1),
                dtype=torch.float64,
                device=Config.DEVICE
            )

            # forward pass without torch.no_grad() because predictions will be recycled for training later
            # split forward pass into three seperate ones for better computational efficiency during uncertainty estimation
            (current_lstm_hidden_state_first, current_lstm_cell_state_first), \
            (current_lstm_hidden_state_second, current_lstm_cell_state_second)  = \
            self.model.forward_feature_extractor(
                observations, 
                previous_lstm_hidden_state_first, previous_lstm_cell_state_first,  
                previous_lstm_hidden_state_second, previous_lstm_cell_state_second,
                auxiliary_input_first, auxiliary_input_second
            )

            # get actions with a single forward pass
            action_logits = self.model.forward_actor(current_lstm_hidden_state_second)

            # inflate the current lstm hidden state in order to perform multiple forward passes simultaneously
            inflated_current_lstm_hidden_state_second = current_lstm_hidden_state_second\
                                                        .unsqueeze(1)\
                                                        .expand(-1, Config.HyPLAN.NUM_FORWARD_PASSES, -1)\
                                                        .reshape(number_of_samples*Config.HyPLAN.NUM_FORWARD_PASSES, -1)
            # batched critic forward pass
            values = self.model.forward_critic(inflated_current_lstm_hidden_state_second)

            # shape: [#nodes, #passes, 1]
            values_exploded = values.reshape(number_of_samples, Config.HyPLAN.NUM_FORWARD_PASSES, 1)
            # obtain aggregated statistics, shape: [#nodes , 1]
            predictive_mean = torch.mean(values_exploded, dim=1)
            # variance (or analogously standard deviation) represents epistemic uncertainty in MC-Dropout
            predictive_variance = torch.var(values_exploded, dim=1)

            # the network is only trained on the root nodes of the episode (which were not simulated by IS-DESPOT)
            if Config.MODE is Mode.TRAIN and is_root_node:
                # used for debug statistics
                self.episode_action_distributions.append(Categorical(logits=action_logits))
                # remember neural network's prediction for training
                self.episode_belief_state_value_estimates.append(predictive_mean)
                self.episode_predicted_policies.append(action_logits)
                # update min/max uncertainty esitmates
                predictive_variance = predictive_variance.squeeze(1).detach().cpu().item()
                if predictive_variance > self.max_variance: self.max_variance = predictive_variance
                if predictive_variance < self.min_variance: self.min_variance = predictive_variance

            # only for root nodes that correspond to actual scene simulation steps
            if Config.MODE in [Mode.TRAIN, Mode.VAL] and is_root_node:
                # save for calcluation of empirical error distribution
                self.episode_predictive_mean.append(predictive_mean)
                self.episode_predictive_variance.append(predictive_variance)

            # calibrate confidence during testing for all nodes
            if Config.MODE is Mode.TEST and Config.CALIBRATE_CONFIDENCE:
                # uncalibrated percentiles based on raw predictive mean and standard deviation
                # shape [#nodes, 99]
                uncalibrated_percentiles = torch.tensor(
                    norm.ppf(
                        self.confidence_levels, 
                        loc=predictive_mean.cpu().numpy(), 
                        scale=torch.sqrt(predictive_variance).cpu().numpy()
                    ),
                    dtype=torch.float64,
                    device=Config.DEVICE
                )
                # expand mean and stddev for vector-based multiplication with z-scores to obtain calibrated percentiles
                # shape: [#nodes, 99, 1]
                calibrated_percentiles = predictive_mean.unsqueeze(1).expand(-1, 99, -1) + \
                                         torch.sqrt(predictive_variance).unsqueeze(1).expand(-1, 99, -1) * \
                                         self.z_scores.unsqueeze(0).expand(number_of_samples, -1, -1)
                
                # calculate mean and standard deviation of calibrated CDF defined by calibrated percentiles
                # shape: [#nodes, 99, 1]
                probabilities = torch.tensor(
                    self.confidence_levels, 
                    dtype=torch.float64, 
                    device=Config.DEVICE   
                ).reshape(1, -1, 1)
                # calculate summary statistics of calibrated CDF
                delta_probabilities = probabilities[:, 1:, :] - probabilities[:, :-1, :]
                calibrated_predictive_mean = torch.sum( 
                    (calibrated_percentiles[:, 1:, :] + calibrated_percentiles[:, :-1, :]) * delta_probabilities, dim=1
                ) / 2
                calibrated_predictive_variance = torch.sum(
                    (calibrated_percentiles[:, 1:, :]**2 + calibrated_percentiles[:, :-1, :]**2) * delta_probabilities, dim=1
                ) / 2 - calibrated_predictive_mean**2
                # min-max scaling and clipping
                uncertainty_estimate = torch.clip(
                    (predictive_variance - self.min_variance) / (self.max_variance - self.min_variance), 0, 1
                ).squeeze(1).detach().cpu().numpy()

                if is_root_node:
                    # used to calculate empirical percentile frequency later
                    self.episode_uncalibrated_percentiles.append(uncalibrated_percentiles.squeeze())
                    self.episode_calibrated_percentiles.append(calibrated_percentiles.squeeze())
                    # track uncertainty development over time
                    self.episode_uncertainty_estimates.append(uncertainty_estimate[0])
                    # required for calculating explained variance
                    self.episode_belief_state_value_estimates.append(predictive_mean)

                    # save car intention images of root nodes
                    if Config.RECORD_CAR_INTENTION_IMAGES:
                        car_intention_save_dir = os.path.join(Config.CAR_INTENTION_DIR, f"episode_{self.episode_counter}")
                        os.makedirs(car_intention_save_dir, exist_ok=True)
                        car_intention_save_path = os.path.join(
                            car_intention_save_dir, f"car_intention_image_at_step_{self.step_counter}.jpg"
                        )
                        cv2.imwrite(car_intention_save_path, self.root_node_car_intention)

        # shape: [#nodes, 1]
        actions = torch.argmax(action_logits, dim=1).detach().cpu().numpy()
        # shape: [#nodes, 1]
        values = predictive_mean.squeeze(1).detach().cpu().numpy()
        # shape: [#nodes, 128]
        current_lstm_hidden_state_second = current_lstm_hidden_state_second.detach().cpu().numpy()
        # shape: [#nodes, 128]
        current_lstm_cell_state_second = current_lstm_cell_state_second.detach().cpu().numpy()

        message = []
        # len(nodes) = batch_size
        for node in range(len(nodes)):
            # append node's lstm state
            [message.append(element) for element in np.concatenate(
                (current_lstm_hidden_state_second[node], current_lstm_cell_state_second[node]), axis=0
            )]
            message.append(actions[node])
            message.append(values[node])
            # send dummy uncertainty values if HyPLAN is training, i.e disable vertical pruning
            if Config.MODE in [Mode.TRAIN, Mode.VAL]: message.append(1.0)
            # only use calibrated uncertainty which is available during testing 
            elif Config.MODE is Mode.TEST: message.append(uncertainty_estimate[node])

        self.despot_connection.send_drla_prediction(message, Connection.HyPLAN_EVALUATION)


    def run_hyplan_server(self):
        try:
            self.initialize_model()
            if Config.RESUME or Config.MODE is not Mode.TRAIN: 
                self.model = load_model(
                    model=self.model, model_dir=Config.MODEL_DIR, checkpoint=Config.MODEL_CHECKPOINT, key="hyplan"
                )

            if Config.MODE is Mode.TRAIN: self.initialize_optimizer()
            if Config.MODE is Mode.TRAIN and self.optimizer is None:
                raise ValueError(f"Invalid HyPLAN optimizer: Not initialized.")     
            if Config.MODE is Mode.TRAIN and not isinstance(self.optimizer, torch.optim.RMSprop):
                raise ValueError(
                    f"Invalid HyPLAN optimizer: Expected '{torch.optim.RMSprop}', got '{type(self.optimizer)}'."
                )
            
            if self.despot_connection is None:
                raise TypeError("IS-DESPOT C++ process interface has not been initialized.")
            if not isinstance(self.despot_connection, DespotBridge):
                raise TypeError(
                    f"Invalid IS-DESPOT C++ prcess interface: "
                    f"Expected '{DespotBridge}', got {type(self.despot_connection)}."
                )
            
            # load min/max variance observed during training
            is_file = os.path.isfile(os.path.join(Config.DATA_DIR, Config.HyPLAN.MIN_MAX_VARIANCE_FILENAME))
            if not is_file and (Config.MODE is not Mode.TRAIN or (Config.MODE is Mode.TRAIN and Config.RESUME)):
                raise ValueError("No min/max variance file found - is required for HyPlan to function properly.")
            if is_file:
                with open(os.path.join(Config.DATA_DIR, Config.HyPLAN.MIN_MAX_VARIANCE_FILENAME)) as file:
                    min_max_variance_dict = json.load(file)
                    self.min_variance = min_max_variance_dict["min_variance"]
                    self.max_variance = min_max_variance_dict["max_variance"]
                    log_info(f"loaded min ({self.min_variance:.4f}) and max: ({self.max_variance:.4f}) variance observed during training")

            # load empirical error distribution 
            if Config.CALIBRATE_CONFIDENCE and Config.MODE is Mode.TEST:
                log_info("loading empirical error distribution...")
                # do not maintain computational graph for speed-up
                with torch.no_grad():
                    # store all tensors
                    z_score_tensors = []
                    # load all tensors in respective directory
                    z_score_tensor_filenames = glob.glob(f"{Config.ERROR_DISTRIBUTION_DIR}/" + "*.pt")
                    for z_score_tensor_filename in z_score_tensor_filenames:
                        if "Z-scores" not in z_score_tensor_filename: continue
                        z_score_tensors.append(torch.load(z_score_tensor_filename, map_location="cuda:0"))
                    if not len(z_score_tensors):
                        raise ValueError("Invalid empirical error distribution: No Z-scores found.")
                    # concatenate as one tensor and sort in ascending order
                    Z = torch.sort(torch.cat(z_score_tensors).squeeze(), dim=0, descending=False).values
                    # get indeces corresponding to percentiles of error distribution
                    percentiles = torch.tensor(
                        # percentiles = relative to sorted length of error distribution 
                        np.rint(self.confidence_levels * Z.shape).astype(int), 
                        dtype=torch.int32, 
                        device=Config.DEVICE
                    )
                    # index error distribution to obtain actual percentiles
                    self.z_scores = torch.index_select(Z, dim=0, index=percentiles).unsqueeze(1)            
                
                os.makedirs(os.path.join(Config.DATA_DIR, Config.HyPLAN.UNCERTAINTY_ANALYSIS_DIR), exist_ok=True)
                # delete previously saved uncertainty estimates (if not resumed)
                filename = os.path.join(
                    Config.DATA_DIR,
                    Config.HyPLAN.UNCERTAINTY_ANALYSIS_DIR, 
                    f"S-{self.scenario_id}_uncertainty_per_step.csv"
                )
                if os.path.isfile(filename) and not Config.RESUME: os.remove(filename)

                filename = os.path.join(
                    Config.DATA_DIR,
                    Config.HyPLAN.UNCERTAINTY_ANALYSIS_DIR, 
                    f"S-{self.scenario_id}_uncertainty_per_scene.csv"
                )
                if os.path.isfile(filename) and not Config.RESUME: os.remove(filename)

                filename = os.path.join(
                    Config.DATA_DIR,
                    Config.HyPLAN.UNCERTAINTY_ANALYSIS_DIR, 
                    f"S-{self.scenario_id}_uncertainty_vs_criticality.json"
                )
                if os.path.isfile(filename) and not Config.RESUME: os.remove(filename)

                filename = os.path.join(
                    Config.DATA_DIR,
                    Config.HyPLAN.UNCERTAINTY_ANALYSIS_DIR, 
                    f"S-{self.scenario_id}_uncertainty_vs_explained_variance.csv"
                )
                if os.path.isfile(filename) and not Config.RESUME: os.remove(filename)

            while not Config.TERMINATE: self.inference_iteration()

        except: log_exception(*sys.exc_info())
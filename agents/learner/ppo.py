import os
import time
import json
from typing import List
import glob

import cv2
import torch
from torch.distributions import Categorical
import numpy as np
import carla
from scipy.stats import norm

from agents.learner.rlagent import RLAgent
from agents.learner.models import A2C as A2CModel
from utils.config import Config, Mode, Action
from utils.utils import load_model, save_model
from utils.logger import log_debug, log_info, log_learning_metrics


class PPO(RLAgent):
    def __init__(self, client, client_world):
        log_info("initializing PPO agent...")    

        # init RLAgent
        super(PPO, self).__init__(client, client_world)

        # must use cuda
        if not torch.cuda.is_available(): raise ValueError("No CUDA capable GPU found.")

        # declare model
        self.model: A2CModel = None
        # define model
        self.initialize_model()

        # declare optimizer
        self.optimizer = None
        # define optimizer
        if Config.MODE is Mode.TRAIN: self.initialize_optimizer()

        # memory
        self.previous_lstm_hidden_state_first: torch.Tensor = None
        self.previous_lstm_cell_state_first: torch.Tensor = None
        self.previous_lstm_hidden_state_second: torch.Tensor = None
        self.previous_lstm_cell_state_second: torch.Tensor = None

        # episode buffers of neural network's predictions for training
        self.episode_value_estimates: List[torch.Tensor] = []
        self.episode_log_probabilities: List[torch.Tensor] = []
        self.episode_entropies: List[torch.Tensor] = []
        # used for training
        self.critic_loss = torch.nn.MSELoss(reduction="mean")
        # used for learning rate annealing
        self.number_of_episodes = None

        # used for calibrating model on validation data set
        self.episode_predictive_mean: List[torch.Tensor] = []
        self.episode_predictive_stddev: List[torch.Tensor] = []
        
        # used for calculating calibration error on test set
        self.confidence_levels = np.arange(0.01, 1, 0.01).round(2)
        self.episode_uncalibrated_percentiles: List[torch.Tensor] = []
        self.episode_calibrated_percentiles: List[torch.Tensor] = []

        # load model if necessary
        if Config.RESUME or Config.MODE is not Mode.TRAIN: 
            self.model = load_model(
                model=self.model, model_dir=Config.MODEL_DIR, checkpoint=Config.MODEL_CHECKPOINT, # defaults to "latest"
                key=f"ppo-{Config.A2C.MODEL_ARCHITECTURE}" 
            )

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

            
    # ================================================================================================= #
    #                                         PPO SETUP METHODS                                         #
    # ================================================================================================= #
    def initialize_model(self):
        if self.model is not None:
            raise TypeError("Exisitng A2C model cannot be overwritten.")
        log_info("initializing A2C model...")

        self.model = A2CModel(hidden_dim=Config.A2C.HIDDEN_LAYER_SIZE, use_dropout=Config.A2C.DROPOUT).double().cuda()
        if Config.MODE is not Mode.TRAIN: self.model.eval()
        if Config.CALIBRATE_CONFIDENCE: self.model.keep_dropout_active()
        log_info(self.model)
          

    def initialize_optimizer(self):
        if self.optimizer is not None:
            raise TypeError("Exisitng optimizer cannot be overwritten.")
        if not isinstance(self.model, A2CModel):
            raise ValueError(f"Invalid A2C model type: Expected '{A2CModel}', got '{type(self.model)}'.")
        log_info("initializing optimizer...")

        # initialize optimizers
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=Config.A2C.LEARNING_RATE,
            betas=(0.9, 0.999), # momentun and velocity
            eps=1e-5
        )

        if self.optimizer is None:
            raise ValueError(f"Invalid PPO optimizer: Not initialized.")     
        if not isinstance(self.optimizer, torch.optim.Adam):
            raise ValueError(f"Invalid PPO optimizer: Expected '{torch.optim.Adam}', got '{type(self.optimizer)}'.")


    def initialize_memory(self):
        log_info("initializing memory...")
        # setup initial inputs for first LSTM cell
        self.previous_lstm_hidden_state_first = torch.zeros((1, 64), dtype=torch.float64, device=Config.DEVICE)
        self.previous_lstm_cell_state_first = torch.zeros((1, 64), dtype=torch.float64, device=Config.DEVICE)
        # setup initial inputs for first LSTM cell
        self.previous_lstm_hidden_state_second = torch.zeros(
            (1, 128 if Config.A2C.MODEL_ARCHITECTURE == "atari-lstm" else 256), 
            dtype=torch.float64, device=Config.DEVICE
        )
        self.previous_lstm_cell_state_second = torch.zeros(
            (1, 128 if Config.A2C.MODEL_ARCHITECTURE == "atari-lstm" else 256), 
            dtype=torch.float64, device=Config.DEVICE
        )


    # ================================================================================================= #
    #                                        NavA2C CORE METHODS                                        #
    # ================================================================================================= #
    def initialize_episode(self, episode_counter: int, scenario: tuple):
        # clear training buffers
        self.episode_value_estimates.clear()
        self.episode_log_probabilities.clear()
        self.episode_entropies.clear()
        # clear confidence calibration buffers for validating and testing
        self.episode_predictive_mean.clear()
        self.episode_predictive_stddev.clear()
        self.episode_uncalibrated_percentiles.clear()
        self.episode_calibrated_percentiles.clear()
        # clear lstm states
        self.initialize_memory()
        # this will clear all relevant buffers
        return super().initialize_episode(episode_counter, scenario)


    def finalize_episode(self, episode_counter: int, non_conclusive: bool, step_counter: int):
        # delete first dummy reward entry (defaults to 0.0)
        del self.episode_rewards[0]
        with torch.no_grad():
            # calculate advantages using generalized advantage estimation
            advantages = []
            # advantage of terminal step
            previous_gae_lambda = self.episode_rewards[-1] - self.episode_value_estimates[-1]
            advantages.append(previous_gae_lambda)
            # for all time steps except the last
            for t in reversed(range(len(self.episode_rewards)-1)):
                # current reward + discounted next reward - estimated reward
                delta = self.episode_rewards[t] + \
                        Config.A2C.DISCOUNT * self.episode_rewards[t+1] - \
                        self.episode_value_estimates[t]
                previous_gae_lambda = delta + Config.A2C.DISCOUNT * Config.A2C.GAE_LAMBDA * previous_gae_lambda
                advantages.append(previous_gae_lambda)

            # calculate retunrs
            advantages.reverse()
            advantages = torch.tensor(advantages, dtype=torch.float64, device=Config.DEVICE)
            episode_value_estimates = torch.vstack(self.episode_value_estimates).squeeze()
            returns = advantages + episode_value_estimates

        # advantage vector is normalised to have 0 mean and 1 std
        if Config.A2C.STANDARDIZE_ADVANTAGE:
            advantages = (advantages - advantages.mean()) / (advantages.std() + np.finfo(np.float64).eps.item())

        # returns vector is normalised to have 0 mean and 1 std
        if Config.A2C.STANDARDIZE_RETURN:
            returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float64).eps.item())  

        # we also train on episode that did not reach a conclusive end (e.g. goal or collision)
        # because otherwise purely learning based agents will never be trained, as their initially 
        # random policy decelerates too often, preventing them from coming to a conclusive end
        if Config.MODE is Mode.TRAIN:
            # sanity check for consistent episode memory
            if returns.shape[0] != step_counter:
                raise ValueError(
                    f"Inconsistent episode memory: Number of steps {step_counter} "
                    f"vs. returns {returns.shape[0]}."
                )
            if episode_value_estimates.shape[0] != step_counter:
                raise ValueError(
                    f"Inconsistent episode memory: Number of steps {step_counter} " 
                    f"vs. value estimates {episode_value_estimates.shape[0]}."
                )
            if len(self.episode_log_probabilities) != step_counter:
                raise ValueError(
                    f"Inconsistent episode memory: Number of steps {step_counter} " 
                    f"vs. log probabilities {len(self.episode_log_probabilities)}."
                )                
            if len(self.episode_entropies) != step_counter:
                raise ValueError(
                    f"Inconsistent episode memory: Number of steps {step_counter} " 
                    f"vs. entropies {len(self.episode_entropies)}."
                )   
            # train on this episode
            self.training_iteration(episode_counter, step_counter, episode_value_estimates, advantages, returns)
            # model checkpointing
            if episode_counter != 0 and episode_counter % Config.MODEL_SAVE_FREQUENCY_EPISODES == 0:
                save_model(
                    model_state_dict=self.model.state_dict(), 
                    model_dir=Config.MODEL_DIR, 
                    checkpoint=str(episode_counter), 
                    name=f"ppo-{Config.A2C.MODEL_ARCHITECTURE}" 
                )
            # always save most recent model
            save_model(
                model_state_dict=self.model.state_dict(),
                model_dir=Config.MODEL_DIR,
                checkpoint="latest",
                name=f"ppo-{Config.A2C.MODEL_ARCHITECTURE}" 
            )

        # can't compute discounted return for non-terminal episodes
        elif Config.CALIBRATE_CONFIDENCE and not non_conclusive:
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
                if len(self.episode_predictive_stddev) != step_counter:
                    raise ValueError(
                        f"Inconsistent episode memory: Number of steps {step_counter} " 
                        f"vs. predicted standard deviations {len(self.episode_predictive_stddev)}."
                    )
                log_info(f"saving Z-scores for episode {self.episode_counter} in '{Config.ERROR_DISTRIBUTION_DIR}'.")
                predictive_mean = torch.vstack(self.episode_predictive_mean).squeeze()
                predictive_stddev = torch.vstack(self.episode_predictive_stddev).squeeze()
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
                if len(self.episode_uncalibrated_percentiles) != step_counter:
                    raise ValueError(
                        f"Inconsistent episode memory: Number of steps {step_counter} " 
                        f"vs. uncalibrated percentiles {len(self.episode_uncalibrated_percentiles)}."
                    )
                if len(self.episode_calibrated_percentiles) != step_counter:
                    raise ValueError(
                        f"Inconsistent episode memory: Number of rewards {step_counter} " 
                        f"vs. calibrated percentiles {len(self.episode_calibrated_percentiles)}."
                    )
                log_info(
                    f"saving (un-)calibrated frequencies for scenario: "
                    f"{self.scenario_id} and episode: {self.episode_counter}"
                )
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

        return super().finalize_episode(episode_counter, non_conclusive, step_counter)


    def run_step(self, step_counter: int):
        self.vehicle = self.client_world.ego_vehicle
        transform = self.vehicle.get_transform()
        start = (self.vehicle.get_location().x, self.vehicle.get_location().y, transform.rotation.yaw)
        end = self.episode_ego_vehicle_goal_position

        (self.step_ego_vehicle_future_trajectory, risk) = super(PPO, self).get_path_simple(start, end)

        agent_vehicle_control = carla.VehicleControl(
            throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False, manual_gear_shift=False, gear=0
        )

        # buggy path planner returning empty path means we skip this step and just maintain current velocity
        if not len(self.step_ego_vehicle_future_trajectory):
            # obtain partial step summary without remembering observation corresponding to this step
            step_summary = super(PPO, self).get_current_observation(step_counter, skipped_step=True)
            step_summary["skipped_step"] = True
            step_summary["control"] = agent_vehicle_control
            return step_summary
        
        # if planned future agent vehicle trajectory is non-trivial
        agent_vehicle_control.steer = \
            (self.step_ego_vehicle_future_trajectory[2][2] - start[2]) / Config.Carla.MAX_STEERING_ANGLE

        # perceive current observation
        super(PPO, self).get_current_observation(step_counter)

        birdview_car_intention = super(PPO, self).get_birdview_car_intention(
            self.vehicle.get_transform(),
            self.client_world.pedestrian.get_transform(),
            self.episode_ego_vehicle_past_trajectory,
            self.step_ego_vehicle_future_trajectory,
            step_counter
        )

        # save car intention images
        if Config.RECORD_CAR_INTENTION_IMAGES:
            car_intention_save_dir = os.path.join(Config.DATA_DIR, f"episode_{self.episode_counter}")
            os.makedirs(car_intention_save_dir, exist_ok=True)
            car_intention_save_path = os.path.join(
                car_intention_save_dir, f"car_intention_image_at_step_{step_counter}.jpg"
            )
            cv2.imwrite(car_intention_save_path, birdview_car_intention)
            
        # get A2C's prediction
        learner_action = self.inference_iteration(birdview_car_intention, step_counter)

        # translate action into CARLA vehicle control commands
        if learner_action is Action.DECELERATE:
            agent_vehicle_control.brake = 0.6
        elif learner_action is Action.ACCELERATE:
            agent_vehicle_control.throttle = 0.6

        # remember action
        self.episode_controls.append(agent_vehicle_control)
        self.episode_actions.append(learner_action)

        # we need the current action before we can calculate the reward
        if Config.REWARD_FUNCTION == "akash":
            step_summary = super(PPO, self).get_reward_akash(step_counter)
        elif Config.REWARD_FUNCTION == "nils":
            step_summary = super(PPO, self).get_reward_nils(step_counter)
        elif Config.REWARD_FUNCTION == "despot":
            step_summary = super(PPO, self).get_reward_despot(step_counter)

        step_summary["skipped_step"] = False
        step_summary["car_intention"] = birdview_car_intention
        step_summary["control"] = agent_vehicle_control
        step_summary["action"] = learner_action

        return step_summary


    def anneal_learning_rate(self, episode_counter: int):
        if self.number_of_episodes is None:
            raise ValueError("Number of episodes has not been set.")
        
        factor = 1.0 - (episode_counter - 1.0) / self.number_of_episodes
        annealed_learning_rate = Config.A2C.LEARNING_RATE * factor
        log_info(f"learning rate for episode {episode_counter}: {annealed_learning_rate}")
        self.optimizer.param_groups[0]["lr"] = annealed_learning_rate


    # trains model multiple times on the same trajectory of states obtained during policy rtollout
    def training_iteration(
            self, 
            episode_counter: int, 
            step_counter: int, 
            episode_value_estimates: torch.Tensor,
            advantages: torch.Tensor,
            returns: torch.Tensor):
        # dummy reward needed for kld calculation
        self.episode_rewards.insert(0, 0.0)
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
        if not isinstance(episode_value_estimates, torch.Tensor):
            raise ValueError(f"Invalid value estimates type: Expected '{torch.Tensor}', got {type(episode_value_estimates)}.")        
        if not isinstance(advantages, torch.Tensor):
            raise ValueError(f"Invalid advantages type: Expected '{torch.Tensor}', got {type(advantages)}.")          
        if not isinstance(returns, torch.Tensor):
            raise ValueError(f"Invalid returns type: Expected '{torch.Tensor}', got {type(returns)}.")

        # update learning rate
        self.anneal_learning_rate(episode_counter)
        
        # required for getting a proper time-reading of cuda related tasks
        torch.cuda.synchronize()
        start_time = time.perf_counter()   

        episode_log_probabilities = torch.vstack(self.episode_log_probabilities).squeeze()
        # losses over all epochs (for debugging)
        total_actor_loss = total_critic_loss = 0
        mean_total_entropy = []
        # number of times we clipped the actor loss
        clipfracs = []
        # sum of gradients across all epochs
        grad_norm = 0
        # learn from trajectory multiple times
        for epoch in range(4):
            # epoch buffers for calculating loss
            epoch_entropies = []
            epoch_log_probabilities = []
            epoch_value_estimates = []
            # reset lstm hidden and cell states
            self.initialize_memory()
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
                # current speed and previous action as one-hot encoded vector
                actions_one_hot_encoded = [0, 0, 0]
                actions_one_hot_encoded[self.episode_actions[step].value] = 1
                auxiliary_input_second = torch.tensor(
                   (self.episode_ego_vehicle_speeds[step], 
                    actions_one_hot_encoded[0], actions_one_hot_encoded[1], actions_one_hot_encoded[2]),
                    dtype=torch.float64,
                    device=Config.DEVICE
                ).unsqueeze(0)
                # forward pass without torch.no_grad() because predictions will be recycled for training later
                action_logits, value, \
                (self.previous_lstm_hidden_state_first, self.previous_lstm_cell_state_first), \
                (self.previous_lstm_hidden_state_second, self.previous_lstm_cell_state_second) = self.model(
                    car_intention, 
                    self.previous_lstm_hidden_state_first, self.previous_lstm_cell_state_first,  
                    self.previous_lstm_hidden_state_second, self.previous_lstm_cell_state_second,
                    auxiliary_input_first, auxiliary_input_second
                )
                # re-calculate entropy using new action logits
                action_distribution = Categorical(logits=action_logits)
                epoch_entropies.append(action_distribution.entropy())
                # re-calculate log probabilities BUT based on the same action executed in the initial policy rollout
                epoch_log_probabilities.append(
                    action_distribution.log_prob(
                        # current action as selected by the policy during rollout (i.e. skips the dummy action)
                        # i.e. in the first iteration this is equal to the first action that was actually executed
                        # by the agent during its initial policy rollout
                        torch.tensor(self.episode_actions[step + 1].value, dtype=torch.int64, device=Config.DEVICE)
                    )
                )
                # save value
                epoch_value_estimates.append(value)

            # log probs for this epoch as well as ratio to log probs during initial policy rollout
            epoch_log_probabilities = torch.vstack(epoch_log_probabilities).squeeze()
            epoch_log_probabilities_ratio = epoch_log_probabilities - episode_log_probabilities
            epoch_log_probabilities_ratio_exp = epoch_log_probabilities_ratio.exp()

            # calculate loss for this epoch
            with torch.no_grad():
                # calculate approximate kullback leibler divergence http://joschu.net/blog/kl-approx.html
                approximate_kld = ((epoch_log_probabilities_ratio_exp - 1) - epoch_log_probabilities_ratio).mean()
                clipfracs += [
                    ((epoch_log_probabilities_ratio_exp - 1.0).abs() > Config.A2C.LOSS_CLIPPING_COEFFICIENT).float().mean().item()
                ]

            # actor loss
            actor_loss_unclipped = -advantages * epoch_log_probabilities_ratio_exp
            actor_loss_clipped = -advantages * torch.clamp(
                epoch_log_probabilities_ratio_exp, 
                1 - Config.A2C.LOSS_CLIPPING_COEFFICIENT, 
                1 + Config.A2C.LOSS_CLIPPING_COEFFICIENT
            )
            actor_loss = torch.max(actor_loss_unclipped, actor_loss_clipped).mean()

            # critic loss
            epoch_value_estimates = torch.vstack(epoch_value_estimates).squeeze()
            if Config.A2C.CLIP_CRITIC_LOSS:
                critic_loss_unclipped = (epoch_value_estimates - returns) ** 2
                clipped_values = episode_value_estimates + torch.clamp(
                    epoch_value_estimates - episode_value_estimates,
                    -Config.A2C.LOSS_CLIPPING_COEFFICIENT,
                    Config.A2C.LOSS_CLIPPING_COEFFICIENT
                )
                critic_loss_clipped = (clipped_values - returns) ** 2
                critic_loss = 0.5 * torch.max(critic_loss_unclipped, critic_loss_clipped).mean()
            else:
                critic_loss = 0.5 * ((epoch_value_estimates - episode_value_estimates) ** 2).mean()

            epoch_entropies = torch.vstack(epoch_entropies).squeeze()
            entropy_loss = epoch_entropies.mean()
            total_loss = actor_loss - \
                         Config.A2C.ENTROPY_COEFFICIENT * entropy_loss + \
                         critic_loss * Config.A2C.CRITIC_LOSS_COEFFICIENT

            # debug information
            torch.cuda.synchronize()
            log_info(
                f"total loss for epoch {epoch + 1} of episode {episode_counter}: {total_loss.item():.4f} "
                f"calculated in {(time.perf_counter() - start_time)*1000:.4f}ms"
            )

            # model.zero_grad() and optimizer.zero_grad() are the same IF all model parameters are in that optimizer
            # it is safer to call model.zero_grad() to make sure all grads are zero, 
            # e.g. if we have two or more optimizers for one model
            # reset gradients
            self.model.zero_grad()
            # time backward pass
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            # propagate gradient backwards
            total_loss.backward()
            # calculate l2-norm of gradients after backward pass
            grads = [param.grad.detach().flatten() for param in self.model.parameters() if param.grad is not None]
            # sum of grad norms across as epochs
            grad_norm += torch.cat(grads).norm().cpu().squeeze().item()
            # gradient clipping
            if Config.A2C.CLIP_GRADIENT: torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)                 
            # update weights
            self.optimizer.step()
            # debug information
            torch.cuda.synchronize()
            log_info(
                f"performed backward pass for epoch {epoch + 1} of episode {episode_counter} in "\
                f"{((time.perf_counter() - start_time)*1000):.4f}ms"
            )
            # debug stats over all epochs
            total_actor_loss += actor_loss.detach().cpu().squeeze().item()
            total_critic_loss += critic_loss.detach().cpu().squeeze().item()
            mean_total_entropy.append(entropy_loss.detach())

        learning_log = {
            "iteration": episode_counter,
            "actor_loss": total_actor_loss,
            "critic_loss": total_critic_loss,
            # sum of undiscounted rewards
            "reward": sum(self.episode_rewards),
            "entropy": torch.mean(torch.vstack(mean_total_entropy), dim=0).cpu().squeeze().item(),
            # explained variance of value estimates during policy rollout
            "explained_variance": self.calculate_explained_variance(
                episode_value_estimates.detach().cpu().numpy(), returns.detach().cpu().numpy()
            ),
            # approximate kullback-leibler divergence between rollout policy and final epoch's policy
            "kld": approximate_kld.detach().cpu().squeeze().item(),
            "grad_norm": grad_norm,
            # over all epochs
            "policy_clip_fraction": np.mean(clipfracs)
        }
        # print learning log
        log_info(json.dumps(learning_log, indent=2))
        # track learning progress
        log_learning_metrics(learning_log)


    def calculate_explained_variance(self, value_estimates: np.ndarray, value_true: np.ndarray) -> float:
        variance_value_true = np.var(value_true)
        if variance_value_true == 0: return np.nan
        else: return 1 - np.var(value_true - value_estimates) / variance_value_true


    def inference_iteration(self, car_intention: np.ndarray, step_counter: int) -> Action:
     # we must have perceived the current velocity that is associated with the provided car intention image
        if len(self.episode_ego_vehicle_speeds) != step_counter:
            raise ValueError(
                f"Invalid numer of episode agent vehicle speeds: "
                f"Expected {step_counter}, got {len(self.episode_ego_vehicle_speeds)}."
            )   
        # because of the dummy reward (0.0) for the initial step (= 1)
        if len(self.episode_actions) != step_counter:
            raise ValueError(
                f"Invalid number of episode actions: Expected {step_counter}, got {len(self.episode_actions)}."
            )
        # at this point we do not yet have calculated the reward associated with the provided (current) car intention image
        if len(self.episode_rewards) != step_counter:
            raise ValueError(
                f"Invalid number of episode rewards: Expected {step_counter}, got {len(self.episode_rewards)}."
            )
        if len(self.episode_birdview_car_intentions) != step_counter:
            raise ValueError(
                f"Invalid number of episode car intention images: Expected {step_counter}, "
                f"got {len(self.episode_birdview_car_intentions)}."
            )
        
        # construct tensor car intention image
        observation = torch.tensor(
            car_intention, 
            dtype=torch.float64, 
            device=Config.DEVICE
        ).transpose(-1, 0).unsqueeze(0)

        # construct feature tensors, which are fed to the LSTM layers in addition to the convolved car intention image
        auxiliary_input_first = torch.tensor(
            (self.episode_rewards[-1]), # reward received for the previous simulation step r_t-1
            dtype=torch.float64, 
            device=Config.DEVICE
        ).expand(1, 1)

        actions_one_hot_encoded = [0, 0, 0]
        actions_one_hot_encoded[self.episode_actions[-1].value] = 1 # last executed action a_t-1
        auxiliary_input_second = torch.tensor(
           (self.episode_ego_vehicle_speeds[-1], # ego vehicle's current speed v_t
            actions_one_hot_encoded[0], actions_one_hot_encoded[1], actions_one_hot_encoded[2]),
            dtype=torch.float64,
            device=Config.DEVICE
        ).unsqueeze(0)
        
        # always with no grad graph because we train for multiple epochs
        with torch.no_grad():
            # split forward pass into three seperate ones for better computational efficiency during uncertainty estimation
            (self.previous_lstm_hidden_state_first, self.previous_lstm_cell_state_first), \
            (self.previous_lstm_hidden_state_second, self.previous_lstm_cell_state_second) = \
            self.model.forward_feature_extractor(
                observation, 
                self.previous_lstm_hidden_state_first, self.previous_lstm_cell_state_first,  
                self.previous_lstm_hidden_state_second, self.previous_lstm_cell_state_second,
                auxiliary_input_first, auxiliary_input_second
            )
            # get actions with a single forward pass
            action_logits = self.model.forward_actor(self.previous_lstm_hidden_state_second)
            
            if Config.CALIBRATE_CONFIDENCE:
                # uncertainty estimation: MC-Dropout with 10 stochastic forward passes (tunable hypoerparameter)
                inflated_current_lstm_hidden_state = self.previous_lstm_hidden_state_second.expand(10, -1)
                # batched critic forward pass
                values = self.model.forward_critic(inflated_current_lstm_hidden_state)
                # obtain aggregated statistics
                predictive_mean = torch.mean(values, dim=0)
                predictive_stddev = torch.std(values, dim=0)
                # always required for calculating the return
                self.episode_value_estimates.append(predictive_mean)

                if Config.MODE is Mode.VAL:
                    # save for calcluation of empirical error distribution
                    self.episode_predictive_mean.append(predictive_mean)
                    self.episode_predictive_stddev.append(predictive_stddev)

                # calibrate confidence during testing
                if Config.MODE is Mode.TEST:
                    # uncalibrated percentiles based on raw predictive mean and standard deviation
                    uncalibrated_percentiles = torch.tensor(
                        norm.ppf(
                            self.confidence_levels, 
                            loc=predictive_mean.detach().cpu().numpy(), 
                            scale=predictive_stddev.detach().cpu().numpy()
                        ),
                        dtype=torch.float64,
                        device=Config.DEVICE
                    )
                    # expand mean and stddev for vector-based multiplication with z-scores to obtain calibrated percentiles
                    predictive_mean = predictive_mean.expand(99, -1)
                    predictive_stddev = predictive_stddev.expand(99, -1)
                    calibrated_percentiles = predictive_mean + predictive_stddev * self.z_scores
                    # used to calculate empirical percentile frequency later
                    self.episode_uncalibrated_percentiles.append(uncalibrated_percentiles)
                    self.episode_calibrated_percentiles.append(calibrated_percentiles.squeeze())
                    # calculate mean and standard deviation of calibrated CDF defined by calibrated percentiles
                    probabilities = torch.tensor(
                        self.confidence_levels, 
                        dtype=torch.float64, 
                        device=Config.DEVICE  
                    )
                    delta_probabilities = probabilities[1:] - probabilities[:-1]
                    cdf_mean = torch.sum((calibrated_percentiles[1:] + calibrated_percentiles[:-1]) * delta_probabilities) / 2
                    cdf_var = torch.sum(
                        (calibrated_percentiles[1:]**2 + calibrated_percentiles[:-1]**2) * delta_probabilities
                    ) / 2 - cdf_mean**2
                    cdf_stddev = torch.sqrt(cdf_var)
            # no uncertainty estimation/calibration
            else:
                value = self.model.forward_critic(self.previous_lstm_hidden_state_second)

        # sample action
        action_distribution = Categorical(logits=action_logits)
        learner_action = action_distribution.sample()

        # remember neural network's prediction for training
        if Config.MODE is Mode.TRAIN:
            # required for actor loss calculation
            self.episode_log_probabilities.append(action_distribution.log_prob(learner_action))
            self.episode_entropies.append(action_distribution.entropy())
            self.episode_value_estimates.append(value)

        return Action(int(learner_action.item()))
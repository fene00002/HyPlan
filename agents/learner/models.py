import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from utils.utils import Config, Agent
from utils.logger import log_info

'''
Formula for CONV-LAYER output dimension: [(W−K+2P)/S]+1.

    W is the input volume
    K is the Kernel size
    P is the padding
    S is the stride

'''

class A2C(nn.Module):
    def __init__(self, hidden_dim, use_dropout):
        super(A2C, self).__init__()

        # use Akash Sinha's NavA2C architecture published here: 
        # https://arxiv.org/abs/2311.12875 or https://github.com/roboak/Nav-Q
        if Config.A2C.MODEL_ARCHITECTURE == "akash":
            self.feature_extractor = nn.Sequential(
                self.init_layer(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(8, 8), stride=(4, 4))), 
                nn.ReLU(),
                self.init_layer(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2))), 
                nn.LayerNorm([64, 48, 48]),
                nn.ReLU(),
                self.init_layer(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1))), 
                nn.ReLU(),
                self.init_layer(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(9, 9), stride=(3, 3))), 
                nn.LayerNorm([128, 13, 13]),
                nn.ReLU(),
                self.init_layer(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(9, 9), stride=(1, 1))), 
                nn.ReLU(),
                self.init_layer(nn.Conv2d(in_channels=128, out_channels=hidden_dim, kernel_size=(5, 5), stride=(1, 1))), 
                nn.ReLU(),
                nn.Flatten(),
                self.init_layer(nn.Linear(in_features=hidden_dim, out_features=hidden_dim)),
                nn.LayerNorm([hidden_dim]),
                nn.ReLU()
            )
            # + 1 = previous reward, + 1 = current velocity, + 3 = previous action one-hot encoded
            self.memory = self.init_layer(nn.LSTMCell(input_size=hidden_dim + 1 + 1 + 3, hidden_size=hidden_dim), std=1.0)
            self.actor = nn.Sequential(
                self.init_layer(nn.Linear(in_features=hidden_dim, out_features=64)),
                nn.LayerNorm([64]),
                nn.ReLU(),
                self.init_layer(nn.Linear(in_features=64, out_features=Config.Carla.NUM_ACTIONS), std=0.01)             
            )
            self.critic = nn.Sequential(
                self.init_layer(nn.Linear(in_features=+hidden_dim, out_features=64)),
                nn.LayerNorm([64]),
                nn.ReLU(),
                nn.Dropout(p = 0.5 if use_dropout else 0.0, inplace = False),
                self.init_layer(nn.Linear(in_features=64, out_features=1), std=1.0)              
            )

        # use Florian Pusse's A2C architecture (originally proposed for HyLEAP)
        # published here: https://github.com/FlorianPusse/OpenDS-CTS
        elif Config.A2C.MODEL_ARCHITECTURE == "florian":
            self.feature_extractor = nn.Sequential(
                self.init_layer(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(8, 8), stride=(4, 4))),
                nn.LeakyReLU(),
                self.init_layer(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=(2, 2))),
                nn.LeakyReLU(),
                nn.Flatten(),
                self.init_layer(nn.Linear(in_features=73728, out_features=hidden_dim)),
                nn.LeakyReLU()
            )
            # + 1 = previous reward, + 1 = current velocity, + 3 = previous action one-hot encoded
            self.memory = self.init_layer(
                nn.LSTMCell(input_size=hidden_dim + 1 + 1 + 1, hidden_size=hidden_dim // 2), 
                std=1.0
            )
            self.actor = self.init_layer(
                nn.Linear(in_features=hidden_dim // 2, out_features=Config.Carla.NUM_ACTIONS),
                std=0.01
            )
            self.critic = nn.Sequential(
                nn.Dropout(p = 0.5 if use_dropout else 0.0, inplace=False),
                self.init_layer(nn.Linear(in_features=hidden_dim // 2, out_features=1), std=1.0)
            )

        # CNN used by PPO for playing image-based Atari games:
        # https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/#Cobbe2021
        elif Config.A2C.MODEL_ARCHITECTURE == "atari" or Config.A2C.MODEL_ARCHITECTURE == "atari-lstm":
            self.feature_extractor = nn.Sequential(
                self.init_layer(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(8, 8), stride=(4, 4))),
                nn.LayerNorm([32, 20, 20]),
                nn.ReLU(),
                self.init_layer(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=(2, 2))),
                nn.LayerNorm([64, 9, 9]),
                nn.ReLU(),
                self.init_layer(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1))),
                nn.LayerNorm([64, 7, 7]),
                nn.ReLU(),
                nn.Flatten(),
                self.init_layer(nn.Linear(in_features=64 * 7 * 7, out_features=512)),
                nn.ReLU()
            )
            if Config.A2C.MODEL_ARCHITECTURE == "atari-lstm":
                # we are not using additional auxiliary input here (speed, action, reward)
                self.memory = self.init_layer(nn.LSTMCell(input_size=512, hidden_size=128), std=1.0)
            # input size depends on whether we are using a lstm cell or not
            self.actor = self.init_layer(
                nn.Linear(
                    in_features=128 if Config.A2C.MODEL_ARCHITECTURE == "atari-lstm" else 512, 
                    out_features=Config.Carla.NUM_ACTIONS
                ),
                std=0.01 # small std
            ) 
            self.critic = nn.Sequential(
                nn.Dropout(p = 0.5 if Config.A2C.DROPOUT else 0.0, inplace=False), # needed for HyPLAN
                self.init_layer(
                    nn.Linear(
                        in_features=128 if Config.A2C.MODEL_ARCHITECTURE == "atari-lstm" else 512, 
                        out_features=1
                    ),
                    std=1.0 # large std
                ) 
            )

        # architecture as in https://www.cs.swarthmore.edu/~meeden/cs81/f17/papers/Navigate.pdf
        # withiout auxiliary prediction tasks (i.e. depth and loop closure) but with layer norm (instead of dividing by 255)
        elif Config.A2C.MODEL_ARCHITECTURE == "NavA2C":
            self.feature_extractor = nn.Sequential(
                nn.LayerNorm([int(Config.Carla.CAR_INTENTION_IMAGE_WIDHT), int(Config.Carla.CAR_INTENTION_IMAGE_HEIGHT)]),
                self.init_layer(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(8, 8), stride=(4, 4))),
                nn.LayerNorm([16, 20, 20]),
                nn.ReLU(),
                self.init_layer(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=(2, 2))),
                nn.LayerNorm([32, 9, 9]),
                nn.ReLU(),
                nn.Flatten(),
                self.init_layer(nn.Linear(in_features=32 * 9 * 9, out_features=256)),
                nn.LayerNorm([256]),
                nn.ReLU()
            )
            # + 1 = previous reward
            self.memory_reward = self.init_layer(nn.LSTMCell(input_size=256 + 1 , hidden_size=64), std=1.0)
            # + 128 = current hidden of self.memory_reward
            # + 1 = current velocity, + 3 = previous action one-hot encoded
            self.memory_velocity_action = self.init_layer(
                nn.LSTMCell(input_size=256 + 64 + 1 + 3, hidden_size=256),
                std=1.0
            )
            self.actor = self.init_layer(nn.Linear(in_features=256, out_features=Config.Carla.NUM_ACTIONS), std=0.01)
            self.critic = nn.Sequential(
                nn.Dropout(p = 0.5 if use_dropout else 0.0, inplace=False),
                self.init_layer(nn.Linear(in_features=256, out_features=1), std=1.0)
            )


    def init_layer(self, layer, std=np.sqrt(2), bias_const=0.0):
        # lstm cell special case
        if isinstance(layer, nn.LSTMCell):
            for name, param in layer.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, bias_const)
                elif "weight" in name:
                    nn.init.orthogonal_(param, std)
        # general case
        elif isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)
        else: raise ValueError(f"Invalid layer type {layer} for initialization.")
        return layer


    def _keep_dropout_active(self, layer):
        if isinstance(layer, nn.Dropout):
            layer.train()
    

    def keep_dropout_active(self):
        self.critic.apply(self._keep_dropout_active)


    def forward(
        self, 
        car_intention, 
        previous_lstm_hidden_state_first, previous_lstm_cell_state_first, 
        previous_lstm_hidden_state_second, previous_lstm_cell_state_second, 
        auxiliary_input_first, auxiliary_input_second
    ):
        extracted_features = self.feature_extractor(car_intention)
        # without LSTM and additional input
        if Config.A2C.MODEL_ARCHITECTURE in ["atari"]:
            # input to actor and critic are just the extracted visual features from the CNN encoder
            current_lstm_hidden_state_first = current_lstm_cell_state_first = extracted_features
            current_lstm_hidden_state_second = current_lstm_cell_state_second = extracted_features
        elif Config.A2C.MODEL_ARCHITECTURE in ["atari-lstm"]:
            # first memory not used
            current_lstm_hidden_state_first = previous_lstm_hidden_state_first
            current_lstm_cell_state_first = previous_lstm_cell_state_first
            # second memory pass without auxiliary input
            current_lstm_hidden_state_second, current_lstm_cell_state_second = self.memory(
                extracted_features, (previous_lstm_hidden_state_second, previous_lstm_cell_state_second)
            )
        if Config.A2C.MODEL_ARCHITECTURE in ["akash", "florian"]:
            # everything goes into the single LSTM cell
            input = torch.cat((extracted_features, auxiliary_input_second, auxiliary_input_first), dim=-1)
            # the first lstm hidden and cell states are not used
            current_lstm_hidden_state_second, current_lstm_cell_state_second = self.memory(
                input, (previous_lstm_hidden_state_second, previous_lstm_cell_state_second)
            )
            # first hidden and cell states not used; just return old ones
            current_lstm_hidden_state_first = previous_lstm_hidden_state_first
            current_lstm_cell_state_first = previous_lstm_cell_state_first
        if Config.A2C.MODEL_ARCHITECTURE in ["NavA2C"]:
            # concat reward
            input_first = torch.cat((extracted_features, auxiliary_input_first), dim=-1)
            # first LSMT
            current_lstm_hidden_state_first, current_lstm_cell_state_first = self.memory_reward(
                input_first, (previous_lstm_hidden_state_first, previous_lstm_cell_state_first)
            )
            input_second = torch.cat((extracted_features, current_lstm_hidden_state_first, auxiliary_input_second), dim=-1)
            # second LSTM
            current_lstm_hidden_state_second, current_lstm_cell_state_second = self.memory_velocity_action(
                input_second, (previous_lstm_hidden_state_second, previous_lstm_cell_state_second)
            )
        # same for every model architecture
        policy = self.actor(current_lstm_hidden_state_second)
        value = self.critic(current_lstm_cell_state_second)
        return policy, value, \
               (current_lstm_hidden_state_first, current_lstm_cell_state_first), \
               (current_lstm_hidden_state_second, current_lstm_cell_state_second)
        

    # split forward pass in order to minimize computational redundancy when obtaining uncertainty values
    def forward_feature_extractor(
        self, 
        car_intention, 
        previous_lstm_hidden_state_first, previous_lstm_cell_state_first, 
        previous_lstm_hidden_state_second, previous_lstm_cell_state_second, 
        auxiliary_input_first, auxiliary_input_second
    ):
        extracted_features = self.feature_extractor(car_intention)
        # without LSTM and additional input
        if Config.A2C.MODEL_ARCHITECTURE in ["atari"]:
            # input to actor and critic are just the extracted visual features from the CNN encoder
            current_lstm_hidden_state_first = current_lstm_cell_state_first = extracted_features
            current_lstm_hidden_state_second = current_lstm_cell_state_second = extracted_features
        elif Config.A2C.MODEL_ARCHITECTURE in ["atari-lstm"]:
            # first memory not used
            current_lstm_hidden_state_first = previous_lstm_hidden_state_first
            current_lstm_cell_state_first = previous_lstm_cell_state_first
            # second memory pass without auxiliary input
            current_lstm_hidden_state_second, current_lstm_cell_state_second = self.memory(
                extracted_features, (previous_lstm_hidden_state_second, previous_lstm_cell_state_second)
            )
        if Config.A2C.MODEL_ARCHITECTURE in ["akash", "florian"]:
            # everything goes into the single LSTM cell
            input = torch.cat((extracted_features, auxiliary_input_second, auxiliary_input_first), dim=-1)
            # the first lstm hidden and cell states are not used
            current_lstm_hidden_state_second, current_lstm_cell_state_second = self.memory(
                input, (previous_lstm_hidden_state_second, previous_lstm_cell_state_second)
            )
            # first hidden and cell states not used; just return old ones
            current_lstm_hidden_state_first = previous_lstm_hidden_state_first
            current_lstm_cell_state_first = previous_lstm_cell_state_first
        if Config.A2C.MODEL_ARCHITECTURE in ["NavA2C"]:
            # concat reward
            input_first = torch.cat((extracted_features, auxiliary_input_first), dim=-1)
            # first LSMT
            current_lstm_hidden_state_first, current_lstm_cell_state_first = self.memory_reward(
                input_first, (previous_lstm_hidden_state_first, previous_lstm_cell_state_first)
            )
            input_second = torch.cat((extracted_features, current_lstm_hidden_state_first, auxiliary_input_second), dim=-1)
            # second LSTM
            current_lstm_hidden_state_second, current_lstm_cell_state_second = self.memory_velocity_action(
                input_second, (previous_lstm_hidden_state_second, previous_lstm_cell_state_second)
            )
        return (current_lstm_hidden_state_first, current_lstm_cell_state_first), \
               (current_lstm_hidden_state_second, current_lstm_cell_state_second)
    

    def forward_actor(self, current_lstm_hidden_state):
        return self.actor(current_lstm_hidden_state)
    

    # only the forward pass of the critic has to be executed multiple times,
    # because it is the only sub-network that has a dropout layer
    def forward_critic(self, current_lstm_hidden_state) -> torch.Tensor:
        return self.critic(current_lstm_hidden_state)



class BaseNetwork(nn.Module):

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=True)


class DQNBase(BaseNetwork):

    def __init__(self, num_channels):
        super(DQNBase, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=(8, 8), stride=(4, 4), padding=0),
            nn.ReLU(),
            # nn.Conv2d(32, 32, kernel_size=(8, 8), stride=(4, 4), padding=0),
            # nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.Flatten(),
        ).apply(self.initialize_weights_he)

    def initialize_weights_he(m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
                
    def forward(self, states):
        states = states.permute(0, 3, 1, 2)
        return self.net(states)


class QNetwork(BaseNetwork):

    def __init__(self, num_channels, num_actions, shared=False,
                 dueling_net=False):
        super().__init__()

        if not shared:
            self.conv = DQNBase(num_channels)

        if not dueling_net:
            self.head = nn.Sequential(
                nn.Linear(46 * 46 * 64 + 6, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_actions))
        else:
            self.a_head = nn.Sequential(
                nn.Linear(46 * 46 * 64 + 6, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_actions))
            self.v_head = nn.Sequential(
                nn.Linear(46 * 46 * 64 + 6, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 1))

        self.shared = shared
        self.dueling_net = dueling_net

    def forward(self, states):
        if not self.shared:
            states = self.conv(states)

        if not self.dueling_net:
            return self.head(states)
        else:
            a = self.a_head(states)
            v = self.v_head(states)
            return v + a - a.mean(1, keepdim=True)


class TwinnedQNetwork(BaseNetwork):

    def __init__(self, num_channels, num_actions, shared=False,
                 dueling_net=False):
        super().__init__()
        self.Q1 = QNetwork(num_channels, num_actions, shared, dueling_net)
        self.Q2 = QNetwork(num_channels, num_actions, shared, dueling_net)

    def forward(self, states):
        q1 = self.Q1(states)
        q2 = self.Q2(states)
        return q1, q2


class CateoricalPolicy(BaseNetwork):

    def __init__(self, num_channels, num_actions, shared=False):
        super().__init__()
        if not shared:
            self.conv = DQNBase(num_channels)

        self.head = nn.Sequential(
            nn.Linear(46 * 46 * 64 + 6, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions))

        self.shared = shared

    def act(self, states):
        if not self.shared:
            states = self.conv(states)

        action_logits = self.head(states)
        greedy_actions = torch.argmax(
            action_logits, dim=1, keepdim=True)
        return greedy_actions

    def sample(self, states, probs=None, steps=None):
        if not self.shared:
            states = self.conv(states)

        action_probs = F.softmax(self.head(states), dim=1)
        if probs is not None and steps is not None:
            action_probs = probs
            # if steps < Config.pre_train_steps:
            #     action_probs = probs
        action_dist = Categorical(action_probs)
        actions = action_dist.sample().view(-1, 1)

        # Avoid numerical instability.
        z = (action_probs == 0.0).float() * 1e-8
        log_action_probs = torch.log(action_probs + z)

        return actions, action_probs, log_action_probs
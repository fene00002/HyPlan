# HyPlan
This repository provides the Python and C++ implementation for **HyPlan**. Additionally, the CARLA-CTS2 benchmark for training, validating and testing is provided.

## Repository structure
<pre>
HyPlan/
├── agents/            # All agent implementations
│   ├── hybrid/        # HyPlan python implementation
│   ├── planner/       # IS-DESPOT C++ implementation           
├── assests/           # CARLA map layout and costmap definitions
├── benchmark/         # CARLA-CTS benchmark definition
├── path_planner/      # Anytime Weighted Hybrid A* for ego-vehicle path planning
├── ped_path_predictor/# AutoBot models for trajectory prediction of exo-agents
├── process_logs/      # Python notebooks for performance evaluation
└── utils/             # Utility functions for e.g. car-intention image generation, inter-process communication (Python & C++), etc.
</pre>

## Requirements:
* CARLA: https://carla.readthedocs.io/en/latest/#getting-started (0.9.15)
* Python: https://www.python.org/downloads/ (3.10 or higher)
* A compiler that supports C++17 (e.g. GCC ≥ 7)
* Cmake: https://cmake.org/download/ (3.8 or higher)
* Miniconda: https://www.anaconda.com/docs/getting-started/miniconda/install (recommended)
* Install python packages specified in requirements.txt

## Building:
1. Navigate to /HyPlan/agents/planner/isdespot/
2. `rm -rf build; mkdir build; cd build; cmake ..; make`
3. `/HyPlan/agents/planner/isdespotO/build/carla_car/is_despotO_carla_car` is the executable binary file of IS-DESPOT's C++ implementation.

### META arguments:
The following provides explanations and arguments influencing the general behavior of the HyPlan algorithm.

* `--mode [train, validate, test]`

  Whether to train, validate or test the specified agent (default: train).

* `--scenario [[0, 09] ...]`

  Scenario(s) to evaluate the agent on (default: ['00'] = all). 
  Use `--scenario 01 02 03 04` to select the first four scenarios.

* `--resume_from_episode [1, 12626]`

  Episode number to resume from with the given episode included (default: 1).

* `--remote`

  Required if the script is run on SLURM cluster (default: False).

* `--predict_pedestrian_path`

  Enables pedestrian path prediction using AutoBots (default: False).

* `--plan_path_with_risk`

  Enables risk aware path planning (default: False).
  WARNING: This will influence execution time and performance.

* `--load_checkpoint [DIRECTORY NAME]`

  Specify model checkpoint directory to load for testing (default: latest).

* `--output_directory [DIRECTORY NAME]`

  Directory under which all script outputs will be centralized (default: None).
  If none is specified, it will be constructed based on the arguments provided. 

* `--epochs [1, 2, 3, 4]`

  Specifies the number of iterations over the entire training set (default: 1).

* `--max_episode_steps [1, 4000]`

  Enforces a maximum number of steps for each simulated episode,
  after which any episode will be forcefully terminated even when 
  no conclusive result (collision or goal) has been reached yet (default: 500).
  Prematurely terminated episodes will not be used for training.

* `--seed [0, 1024]`

  Random number seed (default: 42).

### IS-DESPOT arguments:
* `--no_importance_sampling`

  Do not use importance sampling (default: False).

* `--no_normalization`

  Disable normalization for importance distribution (default: False).

* `--noise [0, 1]`

  Noise level for transitions in belief update (default: 0.5).

* `--timeout [0.01, 1]`

  Belief tree construction time per scene simulation step in seconds (default: 0.25).

* `--time_per_planning_step [0.01, 1]`

  Time between planning simulation steps during belief tree construction in seconds (default: 0.25).

* `--max_search_depth [1, 1000]`

  Maximum search depth during belief tree construction (default: 20).

* `--discount_factor [0, 0.99]`

  Factor to discount future rewards (default: 0.99).

* `--particle_number [1, 10000]`

  Number of particles used to approximate belief nodes (default: 500).

* `--gap_reduction_rate [0, 1]`

  Required gap reduction rate of each trial (default: 0.95).

* `--max_policy_simulation_length [0, 1000]`

  Number of steps to simulate the reactive controller at leaf nodes (default: 90).

* `--pruning_constant [0.01, 1]`

  Pruning constant for regularization (default: 0).

### HyPLAN arguments:
* `--hyplan_num_forward_passes [10, 100]`

  Specifies the number of forward passes used for confidence calculation (default: 10).

* `--calibrate_confidence`

  Calculates and saves the empirical error distribution of the validation set or calibrate confidence estimates on the test set (default: False).
  WARNING: This will influence execution time and performance.

## Executing HyPlan:

Training:
```bash
python controller.py --agent HyPLAN --mode train --reward_function despot --hyplan_num_forward_passes 10 --hidden_layer_size 256 --model_architecture florian --use_dropout --output_directory hyplan --predict_pedestrian_path --gae_lambda 0.9 --clip_gradient --loss_clipping_coefficient 0.2 --clip_critic_loss --critic_loss_coefficient 0.5 --ppo
```

Validate:
```bash
python controller.py --agent HyPLAN --mode validate --reward_function despot --hyplan_num_forward_passes 10 --hidden_layer_size 256 --model_architecture florian --use_dropout --output_directory hyplan --predict_pedestrian_path --calibrate_confidence
```

Testing:
```bash
python controller.py --agent HyPLAN --mode test --reward_function despot --hyplan_num_forward_passes 10 --hidden_layer_size 256 --model_architecture florian --use_dropout --output_directory hyplan --predict_pedestrian_path --calibrate_confidence --track_planning_effort
```

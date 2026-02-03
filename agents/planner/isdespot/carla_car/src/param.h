#ifndef MODELPARAMS_H
#define MODELPARAMS_H

#include <string>
#include "sin_cos_lookup_table.cpp"
#include "map"

namespace ModelParams {
	/******************************************************************** 
	******************* REWARD (RELEVANT) DEFINITIONS *******************
	*********************************************************************/
	extern double COLLISION_PENALTY;
	extern double GOAL_REWARD;
	extern double REWARD_FACTOR_VEL;
	extern double REWARD_BASE_CRASH_VEL;

	/******************************************************************** 
	******************* NOISE (RELEVANT) DEFINITIONS ********************
	*********************************************************************/
    extern int BELIEF_ANGLE_DISCRETIZATION;
	extern double NOISE_GOAL_ANGLE;
	extern double BELIEF_SMOOTHING;
	extern double NOISE_ROBVEL;

	/******************************************************************** 
	**** COLLISION, NEARMISS & GOAL BOUNDARY (RELEVANT) DEFINITIONS *****
	*********************************************************************/
    extern double CAR_WIDTH;
    extern double CAR_LENGTH;
	extern double COLLISION_DISTANCE;
	extern double COLLISION_SIDE_DISTANCE;
	extern double front_nearmiss_margin;
	extern double side_nearmiss_margin;
	extern double back_nearmiss_margin;
	extern double IN_FRONT_ANGLE_DEG;
    extern double LASER_RANGE;
	extern double GOAL_TOLERANCE;

	/******************************************************************** 
	*********** OBSERVATION GENERATION (RELEVANT) DEFINITIONS ***********
	*********************************************************************/
	extern double pos_rln; // position resolution
	extern double vel_rln; // velocity resolution

	/******************************************************************** 
	****************** VELOCITY (RELEVANT) DEFINITIONS ******************
	*********************************************************************/
	extern double VELOCITY_STEP;
	extern double VEL_MAX;
	extern double PED_SPEED;
	extern double PATH_STEP;

	/******************************************************************** 
	******************************* MISC *******************************
	*********************************************************************/
	extern double control_freq;
	const int N_PED_WORLD = 1;
	const int N_PED_IN = 1;
	const int NUM_PEDESTRIANS = 1;
	// HyLEAP's LSTM state is composed of hidden and cell state
	const int LSTM_STATE_SIZE = 2 * 128;
	const int OBSERVATION_SIZE = 4 + 2 * NUM_PEDESTRIANS;
    extern SinCosLookupTable* lookupTable;
	extern std::map<int, std::string> action_idx_to_string;
	enum {
		ACT_DEC = 0,
		ACT_MAIN = 1,
		ACT_ACC = 2
	};
};

#endif


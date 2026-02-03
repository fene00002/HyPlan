#include "param.h"

namespace ModelParams 
{
	/******************************************************************** 
	******************* REWARD (RELEVANT) DEFINITIONS *******************
	*********************************************************************/
	double GOAL_REWARD = 1.0;
    double COLLISION_PENALTY = -1.0;
	double REWARD_FACTOR_VEL = 0.5;
    double REWARD_BASE_CRASH_VEL = 0.5;

	/******************************************************************** 
	******************* NOISE (RELEVANT) DEFINITIONS ********************
	*********************************************************************/
    int BELIEF_ANGLE_DISCRETIZATION = 2;
	// noisy determination of pedestrian goal angle
	double NOISE_GOAL_ANGLE = M_PI * 0.05;
	double BELIEF_SMOOTHING = 0.05;
	// noisy application of agent vehicle velocity update
    double NOISE_ROBVEL = 0.05;

	/******************************************************************** 
	**** COLLISION, NEARMISS & GOAL BOUNDARY (RELEVANT) DEFINITIONS *****
	*********************************************************************/
    double CAR_WIDTH = 1.994;
    double CAR_LENGTH = 4.182;
    double COLLISION_DISTANCE = 5.0;
    double COLLISION_SIDE_DISTANCE = 1.8;
	double front_nearmiss_margin = ModelParams::COLLISION_DISTANCE;
	double side_nearmiss_margin = ModelParams::CAR_WIDTH / 2.0 + ModelParams::COLLISION_SIDE_DISTANCE;
	double back_nearmiss_margin = ModelParams::CAR_LENGTH + 0.1;
    double IN_FRONT_ANGLE_DEG = 180;
	double LASER_RANGE = 50.0;
	double GOAL_TOLERANCE = 3;

	/******************************************************************** 
	*********** OBSERVATION GENERATION (RELEVANT) DEFINITIONS ***********
	*********************************************************************/
	// discretization of agent vehicle and pedestrian positions
	// any pair of particles that differ less than pos_rln in their respective agent vehicle or pedestrian
	// positions are grouped together in the same observation
	double pos_rln = 1.0; //0.25;
	// discretization of agent vehicle velocity
	double vel_rln = 5 * 0.2778; // velocity resolution m/s

	/******************************************************************** 
	****************** VELOCITY (RELEVANT) DEFINITIONS ******************
	*********************************************************************/
	double VELOCITY_STEP = 1.3888; // 5 km/h 
	double VEL_MAX = 50 * 0.27778; // in m/s
	double PATH_STEP = 0.2;
	double PED_SPEED = 1.2;

	/******************************************************************** 
	******************************* MISC *******************************
	*********************************************************************/
	double control_freq = 4;
	SinCosLookupTable* lookupTable = new SinCosLookupTable();
	std::map<int, std::string> action_idx_to_string = {{0, "DEC"}, {1, "MAIN"}, {2, "ACC"}};
}


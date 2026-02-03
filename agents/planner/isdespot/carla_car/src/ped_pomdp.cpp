#include "despot/core/builtin_policy.h"
#include "despot/core/globals.h"
#include "state.h"
#include "ped_pomdp.h"
#include <algorithm>
#include <limits>

using namespace std;

PedPomdp::PedPomdp(WorldModel& _world_model) : world(_world_model), random_(despot::Random(despot::Seeds::Next())) { ; }


class PedPomdpParticleLowerBound : public despot::ParticleLowerBound 
{
	private:
		const PedPomdp* ped_pomdp_;

	public:
		PedPomdpParticleLowerBound(const despot::DSPOMDP* model) : 
		despot::ParticleLowerBound(model), ped_pomdp_(static_cast<const PedPomdp*>(model)) { ; }

		virtual despot::ValuedAction Value(const vector<despot::State*>& particles) const {
			if (despot::Globals::config.IMPROVED_HEURISTIC) { return ImprovedValue(particles); }
			else { return OldValue(particles);}
		}

		despot::ValuedAction ImprovedValue(const vector<despot::State*>& particles) const 
		{
			const PedPomdp*	ped_pomdp_ = static_cast<const PedPomdp*>(model_);
			// calculate random sample weight threshold
			vector<double> importance_weight = ped_pomdp_->ImportanceWeight(particles);
        	double total_weight = std::accumulate(importance_weight.begin(), importance_weight.end(), double(0.0));
			double probability_mass = despot::Random::RANDOM.NextDouble(0, total_weight);
			// sample particle based on importance
			double current_weight = importance_weight[0];
			int particle_position = 0;
			while (probability_mass > current_weight) {
				particle_position++;
				// emulate modulo operation
				if (particle_position == particles.size()) { particle_position = 0; }
				current_weight += importance_weight[particle_position];
			}
			PomdpState* sampled_state = static_cast<PomdpState*>(particles[particle_position]);

			int min_step = numeric_limits<int>::max();
			double min_dist = 100000;
			auto& carpos = ped_pomdp_->world.path[sampled_state->car.pos];
			// Find mininum num of steps for car-pedestrian collision
			for (int i = 0; i < sampled_state->num; i++) {
				auto& p = sampled_state->peds[i];
				// TODO: this should take the ego vehicle's angle and dimensions into accoeunt...
				// 3.25 is maximum distance to collision boundary from front laser (see collsion.cpp)
				double agent_vehicle_pedestrian_distance = COORD::EuclideanDistance(carpos, p.pos) - 3.25;

				// is pedestrian NOT to the left, right or in front of the car?
				if (!ped_pomdp_->world.is_in_front(p.pos, sampled_state->car.pos) ||
					// is agent vehicle moving away from pedestrian?
					ped_pomdp_->world.is_moving_away(*sampled_state, i) ||
					// did the pedestrian stop moving (only happens when on sidewalk)
					ped_pomdp_->world.goals[sampled_state->peds[i].goal] == -1 ||
					// is püedestrian within 50m distance to agent vehicle?
					agent_vehicle_pedestrian_distance > 50.0 ||
					// both ego vehicle and pedestrian are not moving
					(p.vel == 0.0 && sampled_state->car.vel == 0.0)) { 
					// if not do not consider pedestrian
					break;
				}
			
				double dist = max(agent_vehicle_pedestrian_distance, 0.0);
				min_dist = min(dist, min_dist);
				int step = int(ceil(ModelParams::control_freq * dist / ((p.vel + sampled_state->car.vel))));
				min_step = min(step, min_step);
			}

			// the faster we go get better, i.e. higher the lower bound
			double move_penalty = ped_pomdp_->MovementPenalty(*sampled_state);

			// Case 1, no pedestrian: Constant car speed
			double value = move_penalty / (1 - despot::Globals::Discount());

			// Case 2, with pedestrians: Constant car speed, head-on collision with nearest neighbor
			if (min_step != numeric_limits<int>::max()) {
				double crash_penalty = ped_pomdp_->CrashPenalty(*sampled_state);
				value = (move_penalty) * (1 - despot::Globals::Discount(min_step)) / (1 - despot::Globals::Discount())
					+ crash_penalty * despot::Globals::Discount(min_step);
			}
			return despot::ValuedAction(ModelParams::ACT_MAIN, despot::State::Weight(particles) * value);
		}

		// old lower bound as in: https://github.com/dikshant2210/Carla-CTS02/blob/master/ISDESPOT/isdespot-ped-pred/is-despot/problems/isdespotp_car/src/ped_pomdp.cpp
		despot::ValuedAction OldValue(const vector<despot::State*>& particles) const {
			PomdpState* state = static_cast<PomdpState*>(particles[0]);

			int min_step = numeric_limits<int>::max();
			auto& carpos = ped_pomdp_->world.path[state->car.pos];
			double carvel = state->car.vel;

			double min_dist = 100000;

			// Find mininum num of steps for car-pedestrian collision
			for (int i=0; i<state->num; i++) {
				auto& p = state->peds[i];

				// 3.25 is maximum distance to collision boundary from front laser (see collsion.cpp)
				double dist = max(COORD::EuclideanDistance(carpos, p.pos) - 3.25, 0.0);
				min_dist = min(dist, min_dist);
				int step = int(ceil(ModelParams::control_freq * dist / ((p.vel + carvel))));
				min_step = min(step, min_step);
			}

			double move_penalty = ped_pomdp_->MovementPenalty(*state);

			// Case 1, no pedestrian: Constant car speed
			double value = move_penalty / (1 - despot::Globals::Discount());

			// Case 2, with pedestrians: Constant car speed, head-on collision with nearest neighbor
			if (min_step != numeric_limits<int>::max()) {
				double crash_penalty = ped_pomdp_->CrashPenalty(*state);
				value = (move_penalty) * (1 - despot::Globals::Discount(min_step)) / (1 - despot::Globals::Discount())
					+ crash_penalty * despot::Globals::Discount(min_step);
			}
			return despot::ValuedAction(ModelParams::ACT_MAIN, despot::State::Weight(particles) * value);
		}
};


class PedPomdpScenarioLowerBound : public despot::DefaultPolicy
{
	protected:
		const PedPomdp* ped_pomdp_;

	public:
		PedPomdpScenarioLowerBound(const despot::DSPOMDP* model, despot::ParticleLowerBound* bound) :
			despot::DefaultPolicy(model, bound), ped_pomdp_(static_cast<const PedPomdp*>(model)) { ; }

		// reactive controller
		// if breaking distance is less than distance to closest pedestrian with increased velocity: accelerate
		// if breaking distance is greater than distance to closest pedestrian with current velocity: decellerate
		// else: maintain
		int Action(
			const std::vector<despot::State*>& particles, 
			despot::RandomStreams& streams, 
			despot::History& history
		) const 
		{
			if (despot::Globals::config.IMPROVED_HEURISTIC) { return ImprovedAction(particles, streams, history); }
			else { return OldAction(particles, streams, history); }
		}

		int ImprovedAction(
			const std::vector<despot::State*>& particles, 
			despot::RandomStreams& streams, 
			despot::History& history
		) const 
		{
			const PedPomdp* ped_pomdp_ = static_cast<const PedPomdp*>(model_);
			// calculate random sample weight threshold
			vector<double> importance_weight = ped_pomdp_->ImportanceWeight(particles);
			double total_weight = std::accumulate(importance_weight.begin(), importance_weight.end(), double(0.0));
			double probability_mass = despot::Random::RANDOM.NextDouble(0, total_weight);
			// sample particle based on importance
			double current_weight = importance_weight[0];
			int particle_position = 0;
			while (probability_mass > current_weight) {
				particle_position++;
				// emulate modulo operation
				if (particle_position == particles.size()) { particle_position = 0; }
				current_weight += importance_weight[particle_position];
			}
			PomdpState* sampled_state = static_cast<PomdpState*>(particles[particle_position]);

			// sanity check
			if (sampled_state->num != 0 && sampled_state->num != 1) {
				printf("IS-DESPOT::[%s] Invalid number of pedestrians in scene simulation step: "
						"Expected either 0 or 1, got %d.\n", __PRETTY_FUNCTION__, sampled_state->num); 
				exit(-1);				
			} 
			double acceleration_per_step = ModelParams::VELOCITY_STEP / ModelParams::control_freq;
			// trivial case (no pedestrian)
			// either accelerate or maintain velocity when no pedestrian is in scene simulation step
			if (sampled_state->num != 1) {
				// only accelerate if we will maintain a velocity below maximum allowed threshold
				if (sampled_state->car.vel + acceleration_per_step < ModelParams::VEL_MAX) {
					return ModelParams::ACT_ACC;
				} else {
					return ModelParams::ACT_MAIN;
				}
			}
			// at this point, there is a pedestrian in the scene simulation step
			const COORD& ego_vehicle_position = sampled_state->car.coordinates;
			const PedStruct& ped = sampled_state->peds[0];

			// TODO: this should take the ego vehicle's angle and dimensions into accoeunt...
			double agent_vehicle_pedestrian_distance = COORD::EuclideanDistance(ego_vehicle_position, ped.pos) - 3.25;
			// is pedestrian to the left, right or in front of the car?
			if (ped_pomdp_->world.is_in_front(ped.pos, sampled_state->car.pos) &&
				// is agent vehicle NOT moving away from pedestrian?
				!ped_pomdp_->world.is_moving_away(*sampled_state, 0) &&
				// is the pedestrian STILL moving? (stops when it has already crossed the street)
				ped_pomdp_->world.goals[ped.goal] != -1 &&
				// is püedestrian within 50m distance to agent vehicle?
				agent_vehicle_pedestrian_distance <= 50.0 &&
				// either agent vehicle or pedestrian is still moving
				(ped.vel != 0.0 || sampled_state->car.vel != 0.0)) { 

				// reference: https://korkortonline.se/en/theory/reaction-braking-stopping/
				// 250 = fixed figure which is always used * f coefficient of friction, ~0.8 on dry asphalt and 0.1 on ice
				double braking_distance = pow(sampled_state->car.vel*3.6, 2) / (250*0.8);
				// braking distance if we were to accelerate once
				double next_braking_distance = 
					pow((sampled_state->car.vel + acceleration_per_step)*3.6, 2) / (250*0.8);
				// we can still brake in time after accelerating
				if (next_braking_distance < agent_vehicle_pedestrian_distance && 
				// is accelerating a valid action?
					(sampled_state->car.vel + acceleration_per_step) < ModelParams::VEL_MAX) {
						return ModelParams::ACT_ACC;
				// current velocity can not guarantee braking in time
				} else if (braking_distance >= agent_vehicle_pedestrian_distance) {
					return ModelParams::ACT_DEC;
				// braking in time is possible with current velocity
				} else {
					return ModelParams::ACT_MAIN;
				}
			// trivial case (pedestrian can be disregarded)
			} else {
				// only accelerate if we will maintain a velocity below maximum allowed threshold
				if ((sampled_state->car.vel + acceleration_per_step) < ModelParams::VEL_MAX) {
					return ModelParams::ACT_ACC;
				} else {
					return ModelParams::ACT_MAIN;
				}
			}
		}

		// old default policy as in: https://github.com/dikshant2210/Carla-CTS02/blob/master/ISDESPOT/isdespot-ped-pred/is-despot/problems/isdespotp_car/src/WorldModel.cpp
		int OldAction(
			const std::vector<despot::State*>& particles, 
			despot::RandomStreams& streams, 
			despot::History& history
		) const 		
		{
			const PomdpState *state=static_cast<const PomdpState*>(particles[0]);

			double mindist = numeric_limits<double>::infinity();
			auto& carpos = ped_pomdp_->world.path[state->car.pos];
			double carvel = state->car.vel + 1.5;

			double mindist_all = numeric_limits<double>::infinity();

			// Closest pedestrian in front
			for (int i=0; i<state->num; i++) {
				auto& p = state->peds[i];

				double d = COORD::EuclideanDistance(carpos, p.pos);
				if (d >= 0 && d < mindist_all)
					mindist_all = d;

				if(!ped_pomdp_->world.inFront(p.pos, state->car.pos))
					continue;

				d = COORD::EuclideanDistance(carpos, p.pos);
				if (d >= 0 && d < mindist)
					mindist = d;
			}

			double brakingDistance = ((carvel*3.6)*(carvel*3.6)) / (250*0.8);

			if(state->car.vel > 0.1 && (brakingDistance > mindist || mindist_all < 2)){
				return ModelParams::ACT_DEC;
			}

			double nextBrakingDistance = (((carvel+ModelParams::VELOCITY_STEP)*3.6)*((carvel+ModelParams::VELOCITY_STEP)*3.6)) / (250*0.8);
			if((state->car.vel + 0.1 < ModelParams::VEL_MAX) && (nextBrakingDistance < mindist)){
				return ModelParams::ACT_ACC;
			}

			return ModelParams::ACT_MAIN;
		}
};


despot::ScenarioLowerBound* PedPomdp::CreateScenarioLowerBound(std::string particle_bound_name) const 
{
	if (particle_bound_name == "SMART") {
		return new PedPomdpScenarioLowerBound(this, new PedPomdpParticleLowerBound(this));
	} else {
		cerr << "Unsupported scenario lower bound: " << particle_bound_name << endl;
		exit(0);
	}
}


class PedPomdpParticleUpperBound : public despot::ParticleUpperBound {
	protected:
		const PedPomdp* ped_pomdp_;

	public:
		PedPomdpParticleUpperBound(const despot::DSPOMDP* model) : ped_pomdp_(static_cast<const PedPomdp*>(model)) {;}

		double Value(const despot::State& s) const 
		{
			const PomdpState& state = static_cast<const PomdpState&>(s);
			// unless we are in a collision state
			if (ped_pomdp_->world.inCollision(state)) { return ped_pomdp_->CrashPenalty(state); }
			// we return the goal reward diooscounted by the number of steps until the agent vehilce would reach it,
			// if it were to continue driving with constant velocity and other exo agents' positions kept fixed
			// in other words: there is no risk of collision
			return ModelParams::GOAL_REWARD * despot::Globals::Discount(ped_pomdp_->world.min_steps_to_goal(state)) + 0.2;
		}
};


despot::ScenarioUpperBound* PedPomdp::CreateScenarioUpperBound(std::string particle_bound_name) const
{
	if (particle_bound_name == "SMART") {
		return new PedPomdpParticleUpperBound(this);
	} else {
		cerr << "Unsupported scenario upper bound: " << particle_bound_name << endl;
		exit(0);
	}
}


uint64_t PedPomdp::Observe(const despot::State& state) const 
{
    const PomdpState &state_ = static_cast<const PomdpState&>(state);

    static vector<int> obs_vec;
    obs_vec.resize(state_.num * 2 + 2);

    obs_vec[0] = state_.car.pos;
    obs_vec[1] = int(state_.car.vel / ModelParams::vel_rln);

    int i = 2;
    for(int j = 0; j < state_.num; j ++) {
    	obs_vec[i++] = int(state_.peds[j].pos.x / ModelParams::pos_rln);
    	obs_vec[i++] = int(state_.peds[j].pos.y / ModelParams::pos_rln);
    }

	hash<vector<int>> myhash;
	return myhash(obs_vec);
}


double PedPomdp::GetMaxReward() const 
{
	cerr << __PRETTY_FUNCTION__ << " is called, but hasn't been implemented." << endl;
	exit(-1);
}


despot::ValuedAction PedPomdp::GetBestAction() const 
{
	cerr << __PRETTY_FUNCTION__ << " is called, but hasn't been implemented." << endl;
	exit(-1);
	return despot::ValuedAction(
		0, ModelParams::COLLISION_PENALTY * (ModelParams::VEL_MAX*ModelParams::VEL_MAX + ModelParams::REWARD_BASE_CRASH_VEL)
	);
}


int PedPomdp::NumActiveParticles() const 
{
	return memory_pool_.num_allocated();
}


int PedPomdp::NumObservations() const 
{
	cerr << __PRETTY_FUNCTION__ << " is called, but hasn't been implemented." << endl;
	exit(-1);
    return std::numeric_limits<int>::max();
}


int PedPomdp::ParallelismInStep() const 
{
    return ModelParams::N_PED_IN;
}


// Very high cost for collision
double PedPomdp::CrashPenalty(const PomdpState& state) const 
{ 
    return -0.2 + ModelParams::COLLISION_PENALTY * pow(0.5 + state.car.vel / ModelParams::VEL_MAX, 1.4);
}


// Very high cost for collision
double PedPomdp::CrashPenalty(const PomdpStateWorld& state) const 
{ 
    return -0.2 + ModelParams::COLLISION_PENALTY * pow(0.5 + state.car.vel / ModelParams::VEL_MAX, 1.4);
}


// Avoid frequent dec or acc
double PedPomdp::ActionPenalty(int action) const 
{
    if (action == ModelParams::ACT_DEC || ModelParams::ACT_ACC) { return -0.01; }
    else { return 0; }
}


// Less penalty for longer distance travelled
double PedPomdp::MovementPenalty(const PomdpState& state) const 
{
    return (ModelParams::REWARD_FACTOR_VEL * (state.car.vel - ModelParams::VEL_MAX) / ModelParams::VEL_MAX) / 100.0;
}


// Less penalty for longer distance travelled
double PedPomdp::MovementPenalty(const PomdpStateWorld& state) const 
{
    return (ModelParams::REWARD_FACTOR_VEL * (state.car.vel - ModelParams::VEL_MAX) / ModelParams::VEL_MAX) / 100.0;
}


bool PedPomdp::Step(despot::State& state_, double rNum, int action, double& reward, uint64_t& obs) const {
    PomdpState& state = static_cast<PomdpState&>(state_);

	// reset reward
    reward = 0.0;

    // goal reward
    if (world.is_goal(state.car) || state.car.dist_travelled >= 25) {
        reward = ModelParams::GOAL_REWARD;
        return true;
    }
    // collision penalty; applicable only when car is moving
    if(state.car.vel > 0.01 && world.inCollision(state) ) {
        reward = CrashPenalty(state); 
        return true;
    }
    // accelerating beyond allowed maximum velocity; acceleration happens less quickly than deceleration
	if (action == ModelParams::ACT_ACC && state.car.vel >= ModelParams::VEL_MAX) {
			reward = CrashPenalty(state);
			return true;
	}
    // braking when already standing still
    if (action == ModelParams::ACT_DEC && state.car.vel <= 0.01) {
        reward = CrashPenalty(state);
        return true;
    }
    // smoothness control
    reward += ActionPenalty(action);
    // penalize too little distance travelled
    reward += MovementPenalty(state);

	// calculate next pomdp state with given action
	despot::Random random(rNum);
	// push back old ego vehicle coordinates and angle before they are overwritten
	state.past_trajectory.push_back(state.car.coordinates);
	// agent vehicle position update
	world.RobStep(state.car, random);

	// ego-vehicle velocity update
	double acc;
	if (action == ModelParams::ACT_DEC) { acc = -ModelParams::VELOCITY_STEP; } 
	else if (action == ModelParams::ACT_ACC) { 
		if (despot::Globals::config.CORRECT_VELOCITY) { 
			acc = ModelParams::VELOCITY_STEP / ModelParams::control_freq; 
		} else { 
			acc = ModelParams::VELOCITY_STEP; 
		}
	} 
	else { acc = 0; }
	world.RobVelStep(state.car, acc, random);

	for (int i = 0; i < state.num; i++) {
		if (despot::Globals::config.MINIMAL_NOISE) { world.PedStepDeterministic(state.peds[i], 1); }
		else { world.PedStep(state.peds[i], random); }
	}

	// generate observation
	obs = Observe(state);
	return false;
}


bool PedPomdp::ImportanceSamplingStep(
	despot::State& state_, 
	double rNum, 
	despot::ACT_TYPE action, 
	double& reward, 
	despot::OBS_TYPE& obs
) const 
{
    PomdpState& state = static_cast<PomdpState&>(state_);

	// reset reward
    reward = 0.0;

    // goal reward
    if (world.is_goal(state.car) || state.car.dist_travelled >= 25) {
        reward = ModelParams::GOAL_REWARD;
        return true;
    }
    // collision penalty; applicable only when car is moving
    if(state.car.vel > 0.01 && world.inCollision(state) ) {
        reward = CrashPenalty(state); 
        return true;
    }
    // accelerating beyond allowed maximum velocity; acceleration happens less quickly than deceleration
	if (action == ModelParams::ACT_ACC && state.car.vel >= ModelParams::VEL_MAX) {
			reward = CrashPenalty(state);
			return true;
	}
    // braking when already standing still
    if (action == ModelParams::ACT_DEC && state.car.vel <= 0.01) {
        reward = CrashPenalty(state);
        return true;
    }
    // smoothness control
    reward += ActionPenalty(action);
    // penalize too little distance travelled
    reward += MovementPenalty(state);

	// calculate next pomdp state with given action
	despot::Random random(rNum);
	// push back old ego vehicle coordinates and angle before they are overwritten
	state.past_trajectory.push_back(state.car.coordinates);
	// agent vehicle position update
	world.RobStep(state.car, random);

	// ego vehicle velocity update
	double acc;
	if (action == ModelParams::ACT_DEC) { acc = -ModelParams::VELOCITY_STEP; } 
	else if (action == ModelParams::ACT_ACC) { 
		if (despot::Globals::config.CORRECT_VELOCITY) { 
			acc = ModelParams::VELOCITY_STEP / ModelParams::control_freq; 
		} else { 
			acc = ModelParams::VELOCITY_STEP; 
		}
	} 
	else { acc = 0; }
	// always apply correct speed action: why wouldn't we? wtf
	state.weight *= world.ISRobVelStep(state.car, acc, random);
	
	for(int i = 0; i < state.num; i++) {
		// this only changes the weight according to ped-car angle
		state.weight *= world.ISPedStep(state.car, state.peds[i], random, false);
	}

	// generate observation
	obs = Observe(state);
	
	return false;
}


bool PedPomdp::ImportanceSamplingStep(
	despot::State& state_, 
	double rNum, 
	despot::ACT_TYPE action, 
	double& reward, 
	despot::OBS_TYPE& obs, 
	double& x, 
	double& y
) const 
{
    PomdpState& state = static_cast<PomdpState&>(state_);

	// reset reward
    reward = 0.0;

    // goal reward
    if (world.is_goal(state.car) || state.car.dist_travelled >= 25) {
        reward = ModelParams::GOAL_REWARD;
        return true;
    }
    // collision penalty; applicable only when car is moving
    if(state.car.vel > 0.01 && world.inCollision(state) ) {
        reward = CrashPenalty(state); 
        return true;
    }
    // accelerating beyond allowed maximum velocity; acceleration happens less quickly than deceleration
	if (action == ModelParams::ACT_ACC && state.car.vel >= ModelParams::VEL_MAX) {
			reward = CrashPenalty(state);
			return true;
	}
    // braking when already standing still
    if (action == ModelParams::ACT_DEC && state.car.vel <= 0.01) {
        reward = CrashPenalty(state);
        return true;
    }
    // smoothness control
    reward += ActionPenalty(action);
    // penalize too little distance travelled
    reward += MovementPenalty(state);

	// calculate next pomdp state with given action
	despot::Random random(rNum);
	// push back old ego vehicle coordinates and angle before they are overwritten
	state.past_trajectory.push_back(state.car.coordinates);
	// agent vehicle position update
	world.RobStep(state.car, random);

	// ego vehicle velocity update
	double acc;
	if (action == ModelParams::ACT_DEC) { acc = -ModelParams::VELOCITY_STEP; } 
	else if (action == ModelParams::ACT_ACC) { 
		if (despot::Globals::config.CORRECT_VELOCITY) { 
			acc = ModelParams::VELOCITY_STEP / ModelParams::control_freq; 
		} else { 
			acc = ModelParams::VELOCITY_STEP; 
		}
	 } 
	else { acc = 0; }
    state.weight *= world.ISRobVelStep(state.car, acc, random);

	for(int i = 0; i < state.num; i++) {
		// this only changes the weight according to ped-car angle
		state.weight *= world.ISPedStep(state.car, state.peds[i], random, x, y);
	}

	// generate observation
    obs = Observe(state);
    return false;
}


double PedPomdp::ObsProb(uint64_t obs, const despot::State& s, int action) const 
{
	cerr << __PRETTY_FUNCTION__ << " is called, but hasn't been implemented." << endl;
	exit(-1);
	// return obs == Observe(s);
}


vector<vector<double>> PedPomdp::GetBeliefVector(const std::vector<despot::State*> particles) const 
{
	cerr << __PRETTY_FUNCTION__ << " is called, but hasn't been implemented." << endl;
	exit(-1);
}


despot::Belief* PedPomdp::InitialBelief(const despot::State* start, std::string type) const 
{
	cerr << __PRETTY_FUNCTION__ << " is called, but hasn't been implemented." << endl;
	exit(-1);
}


void PedPomdp::PrintState(const despot::State& s, std::ostream& out) const 
{
	cerr << __PRETTY_FUNCTION__ << " is called, but hasn't been implemented." << endl;
	exit(-1);
}


void PedPomdp::PrintObs(const despot::State&state, uint64_t obs, std::ostream& out) const 
{
	cerr << __PRETTY_FUNCTION__ << " is called, but hasn't been implemented." << endl;
	exit(-1);
}


void PedPomdp::PrintAction(int action, std::ostream& out) const 
{
	cerr << __PRETTY_FUNCTION__ << " is called, but hasn't been implemented." << endl;
	exit(-1);
}


// print belief node id, depth, bounds, importance distribution and observation statistics,
// i.e. average, min, max and std values for agent vehicle position, velocity, theta 
// as well as pedestrain position across all particles that approximate the belief
void PedPomdp::PrintBelief(int id, int depth, double lower_bound, double upper_bound, std::vector<despot::State*> particles) const 
{
	// printing debug information about belief states is disabled
	if (!despot::logging::get_scope(despot::logging::DESPOT)) { return; }

	logd << "\t\t- depth(b" << id << "): " << depth << endl
		 << "\t\t- l(b" << id << "): " << lower_bound 
		 << ", u(b" << id << "): " << upper_bound << endl;

	std::vector<double> agent_vehicle_x;
	std::vector<double> agent_vehicle_y;
	// agent vehilce velocity and angle shouldn't really differ between particles so this is more of a sanity check
	std::vector<double> agent_vehicle_velocity; 
	std::vector<double> agent_vehicle_theta;
	std::vector<double> pedestrian_x;
	std::vector<double> pedestrian_y;

	// additional information which might be of interest
	std::vector<double> agent_vehicle_pedestrian_distance;
	std::vector<double> agent_vehicle_distance_travelled;
	std::vector<int> is_in_front;
	std::vector<int> is_moving_away;

	for (const auto& particle: particles) {
		const PomdpState* pomdp_particle = static_cast<PomdpState*>(particle);
		agent_vehicle_x.push_back(pomdp_particle->car.coordinates.x);
		agent_vehicle_y.push_back(pomdp_particle->car.coordinates.y);
		agent_vehicle_velocity.push_back(pomdp_particle->car.vel);
		agent_vehicle_theta.push_back(pomdp_particle->car.coordinates.theta);
		agent_vehicle_distance_travelled.push_back(pomdp_particle->car.dist_travelled);

		if (pomdp_particle->num == 1) {
			pedestrian_x.push_back(pomdp_particle->peds[0].pos.x);
			pedestrian_y.push_back(pomdp_particle->peds[0].pos.y);

			agent_vehicle_pedestrian_distance.push_back(
				COORD::EuclideanDistance(pomdp_particle->car.coordinates, pomdp_particle->peds[0].pos)
			);

			is_in_front.push_back(world.is_in_front(pomdp_particle->peds[0].pos, pomdp_particle->car.pos));
			is_moving_away.push_back(world.is_moving_away(*pomdp_particle, 0));
		}
	}

	logd << "\t\t- b" << id << " statistics (across all particles): " << endl;
	std::pair<double, double> agent_vehicle_x_avg_and_std = MathUtils::get_average_and_stdev(agent_vehicle_x);
	std::pair<double, double> agent_vehicle_x_min_and_max = MathUtils::get_min_and_max(agent_vehicle_x);
	logd << "\t\t\t- agent vehicle x (avg: " <<  agent_vehicle_x_avg_and_std.first 
		 << ", std: " << agent_vehicle_x_avg_and_std.second
		 << ", min: " << agent_vehicle_x_min_and_max.first 
		 << ", max: " << agent_vehicle_x_min_and_max.second << ")" << endl;

	std::pair<double, double> agent_vehicle_y_avg_and_std = MathUtils::get_average_and_stdev(agent_vehicle_y);
	std::pair<double, double> agent_vehicle_y_min_and_max = MathUtils::get_min_and_max(agent_vehicle_y);
	logd << "\t\t\t- agent vehicle y (avg: " <<  agent_vehicle_y_avg_and_std.first 
		 << ", std: " << agent_vehicle_y_avg_and_std.second
		 << ", min: " << agent_vehicle_y_min_and_max.first 
		 << ", max: " << agent_vehicle_y_min_and_max.second << ")" << endl;

	std::pair<double, double> agent_vehicle_velocity_avg_and_std = MathUtils::get_average_and_stdev(agent_vehicle_velocity);
	std::pair<double, double> agent_vehicle_velocity_min_and_max = MathUtils::get_min_and_max(agent_vehicle_velocity);
	logd << "\t\t\t- agent vehicle velocity (avg: " <<  agent_vehicle_velocity_avg_and_std.first 
		 << ", std: " << agent_vehicle_velocity_avg_and_std.second
		 << ", min: " << agent_vehicle_velocity_min_and_max.first 
		 << ", max: " << agent_vehicle_velocity_min_and_max.second << ")" << endl;

	std::pair<double, double> agent_vehicle_theta_avg_and_std = MathUtils::get_average_and_stdev(agent_vehicle_theta);
	std::pair<double, double> agent_vehicle_theta_min_and_max = MathUtils::get_min_and_max(agent_vehicle_theta);
	logd << "\t\t\t- agent vehicle angle [DEG] (avg: " <<  agent_vehicle_theta_avg_and_std.first 
		 << ", std: " << agent_vehicle_theta_avg_and_std.second
		 << ", min: " << agent_vehicle_theta_min_and_max.first 
		 << ", max: " << agent_vehicle_theta_min_and_max.second << ")" << endl;

	std::pair<double, double> pedestrian_x_avg_and_std = MathUtils::get_average_and_stdev(pedestrian_x);
	std::pair<double, double> pedestrian_x_min_and_max = MathUtils::get_min_and_max(pedestrian_x);
	logd << "\t\t\t- pedestrian x (avg: " <<  pedestrian_x_avg_and_std.first 
		 << ", std: " << pedestrian_x_avg_and_std.second
		 << ", min: " << pedestrian_x_min_and_max.first 
		 << ", max: " << pedestrian_x_min_and_max.second << ")" << endl;

	std::pair<double, double> pedestrian_y_avg_and_std = MathUtils::get_average_and_stdev(pedestrian_y);
	std::pair<double, double> pedestrian_y_min_and_max = MathUtils::get_min_and_max(pedestrian_y);
	logd << "\t\t\t- pedestrian y (avg: " <<  pedestrian_y_avg_and_std.first 
		 << ", std: " << pedestrian_y_avg_and_std.second
		 << ", min: " << pedestrian_y_min_and_max.first 
		 << ", max: " << pedestrian_y_min_and_max.second << ")" << endl;

	std::pair<double, double> distance_between_avg_and_std = MathUtils::get_average_and_stdev(agent_vehicle_pedestrian_distance);
	std::pair<double, double> distance_between_min_and_max = MathUtils::get_min_and_max(agent_vehicle_pedestrian_distance);
	logd << "\t\t\t- agent vehicle-pedestrian distance (avg: " <<  distance_between_avg_and_std.first 
		 << ", std: " << distance_between_avg_and_std.second
		 << ", min: " << distance_between_min_and_max.first 
		 << ", max: " << distance_between_min_and_max.second << ")" << endl;

	std::pair<double, double> distance_travelled_avg_and_std = MathUtils::get_average_and_stdev(agent_vehicle_distance_travelled);
	std::pair<double, double> distance_travelled_min_and_max = MathUtils::get_min_and_max(agent_vehicle_distance_travelled);
	logd << "\t\t\t- agent vehicle distance travelled (avg: " <<  distance_travelled_avg_and_std.first 
		 << ", std: " << distance_travelled_avg_and_std.second
		 << ", min: " << distance_travelled_min_and_max.first 
		 << ", max: " << distance_travelled_min_and_max.second << ")" << endl;

	logd << "\t\t\t- is_in_front(): " << MathUtils::get_average_and_stdev(is_in_front).first*100 << "%" << endl
		 << "\t\t\t- is_moving_away(): " << MathUtils::get_average_and_stdev(is_moving_away).first*100 << "%" << endl;

	PrintParticles(particles);
}


// prints the importance distribution encoded in the provided particles vector 
// in ascending order of pedestrian goal direction weight
void PedPomdp::PrintParticles(const std::vector<despot::State*> particles) const 
{
	// printing debug information about importance distributions is disabled
	if (!despot::logging::get_scope(despot::logging::IMPORTANCE_SAMPLING)) { return; }

	logis << "\t\t- importance distribution [particles: " 
		  << particles.size() << ", weight: " << despot::State::Weight(particles) << "]:" << endl;
	// how are pedestrian goal angles distributed across particles?
	std::map<int, int> ped_goal_count;
	// how much weight does each pedestrian goal angle have associated with it?
	std::map<int, double> ped_goal_weight;

	for(const auto& particle: particles) {
		const PomdpState* pomdp_particle = static_cast<PomdpState*>(particle);
		if (pomdp_particle->num == 0) {
			logis << "\t\t\t- no pedestrian in scene simulation step" << endl;
			return;
		} else if (pomdp_particle->num > 1) {
			printf("IS-DESPOT::[%s] Invalid number of pedestrians in scene simulation step: Expected 1, got %d.\n",
				   __PRETTY_FUNCTION__, pomdp_particle->num);
			exit(-1);
		} else {
			// goal direction in degree
			ped_goal_count[pomdp_particle->peds[0].goal*2]++;
			ped_goal_weight[pomdp_particle->peds[0].goal*2] += pomdp_particle->weight;
		}
	}
	for (const auto& el: despot::SortByValue(ped_goal_weight)) {
		logis << "\t\t\t- pedestrian goal angle [DEG]: " << el.first 
			  << ", weight: " << el.second 
			  << ", #particles: " << ped_goal_count[el.first] << endl;
	}
}


despot::State* PedPomdp::Allocate(int state_id, double weight) const 
{
	PomdpState* particle = memory_pool_.Allocate();
	particle->state_id = state_id;
	particle->weight = weight;
	return particle;
}


std::vector<despot::State*> PedPomdp::ConstructParticles(std::vector<PomdpState>& samples) 
{
	int num_particles=samples.size();
	std::vector<despot::State*> particles;
	for(int i=0;i<samples.size();i++) {
		PomdpState* particle = static_cast<PomdpState*>(Allocate(-1, 1.0/num_particles));
		(*particle) = samples[i];
		particle->SetAllocated();
		particle->weight = 1.0/num_particles;
		particles.push_back(particle);
	}
	return particles;
}


despot::State* PedPomdp::Copy(const despot::State* particle) const 
{
	PomdpState* new_particle = memory_pool_.Allocate();
	*new_particle = *static_cast<const PomdpState*>(particle);
	new_particle->SetAllocated();
	return new_particle;
}


void PedPomdp::Free(despot::State* particle) const 
{
	memory_pool_.Free(static_cast<PomdpState*>(particle));
}
/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, const double std[]) {
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    default_random_engine gen;
    double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

    num_particles = 20;

    std_x = std[0];
    std_y = std[1];
    std_theta = std[2];

    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    particles = std::vector<Particle>(num_particles);
    weights = std::vector<double>(num_particles);

    for (auto &particle : particles) {
        double sample_x, sample_y, sample_theta;

        sample_x = dist_x(gen);
        sample_y = dist_y(gen);
        sample_theta = dist_theta(gen);

        particle.x = sample_x;
        particle.y = sample_y;
        particle.theta = sample_theta;
        particle.weight = 1.0;
    }

    for (auto &weight : weights) {
        weight = 1.0;
    }

}

void ParticleFilter::prediction(double delta_t, const double std_pos[],
                                double velocity, double yaw_rate) {
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine gen;

    normal_distribution<double> dist_x(0.0, std_pos[0]);
    normal_distribution<double> dist_y(0.0, std_pos[1]);
    normal_distribution<double> dist_theta(0.0, std_pos[2]);

    const double p1 = (velocity / yaw_rate);
    const double p2 = yaw_rate * delta_t;
    for (auto &particle : particles) {
        particle.x += p1 * (sin(particle.theta + p2) - sin(particle.theta));
        particle.y += p1 * (cos(particle.theta) - cos(particle.theta + p2));
        particle.theta += p2;

        particle.x += dist_x(gen);
        particle.y += dist_y(gen);
        particle.theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations) {

    for (auto &obs : observations) {

        double min = dist(obs.x, obs.y, predicted[0].x, predicted[0].y);
        int id = 0;

        for (int i = 0; i < predicted.size(); i++) {

            double distance = dist(obs.x, obs.y, predicted[i].x, predicted[i].y);
            if (distance < min) {
                min = distance;
                id = i;
            }
        }
        obs.id = id;
    }

}

void dataAssociation2(Map predicted, std::vector<LandmarkObs> &observations) {
    for (auto &obs : observations) {

        double min = dist(obs.x, obs.y, predicted.landmark_list[0].x_f, predicted.landmark_list[0].y_f);
        int id = 0;

        for (int i = 0; i < predicted.landmark_list.size(); i++) {
            double distance = dist(obs.x, obs.y, predicted.landmark_list[i].x_f, predicted.landmark_list[i].y_f);
            if (distance < min) {
                min = distance;
                id = i;
            }
        }
        obs.id = id;
    }
}

/**
 * Multivariate Gaussian Probability Density
 * @param sig_x
 * @param sig_y
 * @param x_obs
 * @param y_obs
 * @param mu_x
 * @param mu_y
 * @return
 */
double multivariateGaussianProbabilityDensity(double sig_x, double sig_y, double x_obs,
                                              double y_obs, double mu_x, double mu_y) {

    // calculate normalization term
    const double gauss_norm = (1 / (2 * M_PI * sig_x * sig_y));

    // calculate exponent
    const double exponent = (pow(x_obs - mu_x, 2)) / (2 * pow(sig_x, 2)) + (pow(y_obs - mu_y, 2)) / (2 * pow(sig_y, 2));

    // calculate weight using normalization terms and exponent
    return gauss_norm * exp(-exponent);
}

/**
 * Homogeneous transformation for X
 * @param x_particle
 * @param x_observation
 * @param y_observation
 * @param theta
 * @return
 */
double transformXCoordToMap(double x_particle, double x_observation,
                            double y_observation, double theta) {
    return x_particle + (cos(theta) * x_observation) - (sin(theta) * y_observation);
}

/**
 * Homogeneous transformation for Y
 * @param y_particle
 * @param x_observation
 * @param y_observation
 * @param theta
 * @return
 */
double transformYCoordToMap(double y_particle, double x_observation,
                            double y_observation, double theta) {
    return y_particle + (sin(theta) * x_observation) + (cos(theta) * y_observation);
}

/**
 * Homogeneous transformation
 *
 * @param obs
 * @param particle
 * @return
 */
LandmarkObs observationTransformation(const LandmarkObs obs, const Particle particle) {
    LandmarkObs observation{};
    observation.id = obs.id;
    observation.x = transformXCoordToMap(particle.x, obs.x, obs.y, particle.theta);
    observation.y = transformYCoordToMap(particle.y, obs.x, obs.y, particle.theta);
    return observation;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html


    vector<LandmarkObs> list_observations_transformed;
    /*
     * For each particle
     */
    for (int i = 0; i < num_particles; i++) {


        /*
         * I)
         * Transform car sensor landmark observations from the car coordinates system to
         * the map coordinates system to match the particles point of view
         */
        for (const auto &car_observation : observations) {
            LandmarkObs observation = observationTransformation(car_observation, particles[i]);
            list_observations_transformed.push_back(observation);
        }


        /*
         * II.b)
         * Associate these transformed observations with the nearest landmark on the map
         */
        dataAssociation2(map_landmarks, list_observations_transformed);


        /*
         * III)
         * Update our particle weight by applying the multivariate density function
         * for each measurement.
         * Then combining the probabilities of all measurements by taking their product.
         * Final weight -> posterior probability
         */

        double w = weights[i];
        for (auto &obs_in_map_coord : list_observations_transformed) {

            const int near_id = obs_in_map_coord.id;
            const double obs_x = obs_in_map_coord.x;
            const double obs_y = obs_in_map_coord.y;

            const double predicted_x = map_landmarks.landmark_list[near_id].x_f;
            const double predicted_y = map_landmarks.landmark_list[near_id].y_f;

            double res = multivariateGaussianProbabilityDensity(std_landmark[0], std_landmark[1],
                                                   obs_x, obs_y, predicted_x, predicted_y);

            w *= res;
        }

        this->particles[i].weight = w;
        this->weights[i] = w;
    }


}

void ParticleFilter::resample() {
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    default_random_engine gen;
    discrete_distribution<> dist_particles(weights.begin(), weights.end());

    vector<Particle> new_particles_distribution(num_particles);

    for (int i = 0; i < num_particles; i++) {
        new_particles_distribution[i] = particles[dist_particles(gen)];
    }
    particles = new_particles_distribution;
}

Particle ParticleFilter::SetAssociations(Particle &particle, const std::vector<int> &associations,
                                         const std::vector<double> &sense_x, const std::vector<double> &sense_y) {
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseX(Particle best) {
    vector<double> v = best.sense_x;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseY(Particle best) {
    vector<double> v = best.sense_y;
    stringstream ss;
    copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

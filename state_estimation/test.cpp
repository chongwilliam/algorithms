/**
 * @file test.cpp
 * @brief Example UKF usage.
 * 
*/

#include "ukf.h"
#include <random>
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

using Vector7d = Eigen::Matrix<double, 7, 1>;

int main()
{
    // Problem setup
    double lambda = 1;
    double alpha = 1e-2;  // 1e-4 to 1e0
    double dt = 1. / 100;  
    Matrix6d P = 1e0 * Matrix6d::Identity();
    Matrix6d Q = 1e-6 * Matrix6d::Identity();
    double accel_noise = 1e-4;
    double gyro_noise = 1e-2;
    Matrix6d R;
    R.block<3, 3>(0, 0) = accel_noise * Matrix3d::Identity();
    R.block<3, 3>(3, 3) = gyro_noise * Matrix3d::Identity();
    double mass = 500;  
    Matrix3d I = Vector3d(50, 50, 50).asDiagonal();
    Matrix6d M;
    M.block<3, 3>(0, 0) = mass * Matrix3d::Identity();
    M.block<3, 3>(3, 3) = I;

    // Simulation 
    int N = 1000;  // number of simulation steps
    // double dt = 1. / 100;  // filter update frequency 
    double sim_dt = 1. / 1000;  // simulation frequency
    Vector7d pose = Vector7d::Zero();  // position; quaternion (w, x, y, z)
    pose(3) = 1;  // set to identity rotation matrix at the start 
    Vector6d velocity = Vector6d::Zero();
    Matrix3d rot_in_world = Matrix3d::Identity();

    // Init
	auto state_estimator = new Filter::UKF(lambda, alpha, dt, velocity, P, Q, R, mass, I);
    vector<Vector6d> filter_state_log;
    vector<Matrix6d> filter_cov_log;

    // White noise generators
    double random_force_noise = 1e1;  
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<double> accel_dist(0, accel_noise);
    std::normal_distribution<double> gyro_dist(0, gyro_noise);
    std::normal_distribution<double> force_dist(0, random_force_noise);

    // Test case #1:  straight line motion with constant force (no moment)
    const int test = 0;

    if (test == 0) {
        vector<Vector6d> F;
        F.reserve(N);
        for (int i = 0; i < N; ++i) {
            Vector6d random_force;
            for (int j = 0; j < 6; ++j) {
                random_force(j) = force_dist(generator);
            }
            random_force.head(3) += mass * Vector3d(0, 0, -9.81);
        }

        for (int i = 0; i < N; ++i) {
            // Compute ground-truth accelerations 
            Vector6d nonlinear;
            nonlinear.head(3).setZero();
            Vector3d w_truth = velocity.tail(3);
            nonlinear.tail(3) = w_truth.cross(I * w_truth);
            Vector6d accel = M.inverse() * (F[i] - nonlinear);

            // Forward integrate to get velocity (at 1000 Hz vs. 100 Hz)
            velocity += accel * sim_dt;

            // Forward integrate to get pose (most important is the orientation)
            pose.head(3) += velocity.head(3) * sim_dt;
            double w_x = velocity(3);
            double w_y = velocity(4);
            double w_z = velocity(5);
            Matrix4d omega;
            omega << 0, -w_x, -w_y, -w_z,
                        w_x, 0, w_z, -w_y,
                        w_y, -w_z, 0, w_x,
                        w_z, w_y, -w_x, 0;
            omega *= 0.5;
            double delta_w_norm = 0.5 * ( (w_x * sim_dt) * (w_x * sim_dt) + 
                                           (w_y * sim_dt) * (w_y * sim_dt) + 
                                           (w_z * sim_dt) * (w_z * sim_dt) );
            pose.tail(4) = (Matrix4d::Identity() * cos(delta_w_norm / 2) + sin(delta_w_norm / 2) * delta_w_norm * omega) * pose.tail(4);
            Quaterniond q;
            q.w() = pose(3);
            q.x() = pose(4);
            q.y() = pose(5);
            q.z() = pose(6);
            rot_in_world = q.normalized().toRotationMatrix();

            std::cout << (i & 10) << std::endl;

            // Update sensor information and filter (only every 100 Hz)
            if (i & 10 == 0) {
                Vector3d accel_sensor;
                Vector3d gyro_sensor;
                for (int j = 0; j < 3; ++j) {
                    accel_sensor(j) = accel(j) + accel_dist(generator);
                    gyro_sensor(j) = velocity(j + 3) + gyro_dist(generator);
                }
                state_estimator->updateSensors(accel_sensor, gyro_sensor, F[i], rot_in_world);
                state_estimator->updateStep();
                filter_state_log.push_back(state_estimator->m_state);
                filter_cov_log.push_back(state_estimator->m_P);
                std::cout << "True State: " << velocity << endl;
                std::cout << "Filter State: " << state_estimator->m_state << endl << endl;
            }
        }
            
    }

    // Verification
    // for (int i = 0; i < filter_state_log.size(); ++i) {
    // }

    // plt.figure_size(1200, 700);
    // plt.plot(time_vector, filter_state_vector);


    std::cout << "Finished Simulation" << std::endl;


}
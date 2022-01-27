/**
 * @file test.cpp
 * @brief Example UKF usage.
 * 
*/

#include "ukf.h"
#include <random>
#include <chrono>
#include <sciplot/sciplot.hpp>

using namespace sciplot;
using Vector7d = Eigen::Matrix<double, 7, 1>;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

int main()
{
    // Filter parameters 
    double alpha = 1e-2;  // 1e-4 to 1e0
    double ki = 0;  // tunable
    int L = 6;  
    double lambda = alpha * alpha * ((double) L + ki) - (double) L;
    double filter_dt = 1. / 1000;  

    // Covariances
    Matrix6d P = 1e0 * Matrix6d::Identity();
    Matrix6d Q = 1e-6 * Matrix6d::Identity();  // to be tuned
    double accel_noise = 1e-4;  // from sensor measurement
    double gyro_noise = 1e-2;  // from sensor measurement 
    Matrix6d R = Matrix6d::Zero();
    R.block<3, 3>(0, 0) = accel_noise * Matrix3d::Identity();
    R.block<3, 3>(3, 3) = gyro_noise * Matrix3d::Identity();

    // Inertial parameters
    double mass = 500;  
    Matrix3d I = Vector3d(50, 50, 50).asDiagonal();
    Matrix6d M = Matrix6d::Zero();
    M.block<3, 3>(0, 0) = mass * Matrix3d::Identity();
    M.block<3, 3>(3, 3) = I;

    // Simulation 
    int N = 10000 * 300;  // number of simulation steps
    // double dt = 1. / 100;  // filter update frequency 
    double sim_dt = 1. / 10000;  // simulation frequency
    Vector7d pose = Vector7d::Zero();  // position; quaternion (w, x, y, z)
    pose(3) = 1;  // set to identity rotation matrix at the start 
    Vector6d velocity = Vector6d::Zero();
    Matrix3d rot_in_world = Matrix3d::Identity();

    // Init
	auto state_estimator = new Filter::UKF(lambda, alpha, filter_dt, velocity, P, Q, R, mass, I);
    vector<Vector6d> ground_truth_log;
    vector<Vector3d> vel_integrated_log;
    vector<Vector6d> filter_state_log;
    vector<Matrix6d> filter_cov_log;

    // White noise generators
    double random_force_noise = 1e2;  
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<double> accel_dist(0, accel_noise);
    std::normal_distribution<double> gyro_dist(0, gyro_noise);
    std::normal_distribution<double> force_dist(2, random_force_noise);

    // Test setup
    const int test = 0;
    auto t1 = high_resolution_clock::now();
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    int n_skip = 10;  // 1 kHz filter update 
    int n_filter_samples = N / n_skip;
    VectorXd timings(n_filter_samples);
    int cnt = 0;
    Vector3d integrated_velocity = Vector3d::Zero();

    if (test == 0) {
        vector<Vector6d> F;
        F.reserve(N);
        for (int i = 0; i < N; ++i) {
            Vector6d random_force;
            for (int j = 0; j < 6; ++j) {
                random_force(j) = force_dist(generator);
            }
            random_force.head(3) += 0 * Vector3d(0, 0, -9.81);  // if gravity force is added
            F.push_back(random_force);
        }

        for (int i = 0; i < N; ++i) {
            // Compute ground-truth accelerations 
            Vector6d nonlinear;
            nonlinear.head(3).setZero();
            Vector3d w_truth = velocity.tail(3);
            nonlinear.tail(3) = w_truth.cross(I * w_truth);
            Vector6d accel = M.inverse() * (F[i] - nonlinear);

            // Forward integrate to get velocity (at 10000 Hz vs. 1000 Hz)
            velocity += accel * sim_dt;
            ground_truth_log.push_back(velocity);

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
            double delta_w_norm = 0.5 * sqrt( (w_x * sim_dt) * (w_x * sim_dt) + 
                                                (w_y * sim_dt) * (w_y * sim_dt) + 
                                                (w_z * sim_dt) * (w_z * sim_dt) );
            pose.tail(4) = (Matrix4d::Identity() * cos(delta_w_norm / 2) + 
                                sin(delta_w_norm / 2) * delta_w_norm * omega) * pose.tail(4);
            Quaterniond q;
            q.w() = pose(3);
            q.x() = pose(4);
            q.y() = pose(5);
            q.z() = pose(6);
            rot_in_world = q.normalized().toRotationMatrix();  // inertial to body frame 

            // Generate sensor measurements in the inertial frame, with added noise  
            Vector3d accel_sensor;
            Vector3d gyro_sensor;
            for (int j = 0; j < 3; ++j) {
                accel_sensor(j) = accel(j) + accel_dist(generator);  // probably needs zero mean
                gyro_sensor(j) = velocity(j + 3) + gyro_dist(generator);
            }

            // Update sensor information and filter (frequency from n_skip)
            if (i % n_skip == 0) {
                // Naive integration
                integrated_velocity += accel_sensor * sim_dt;
                vel_integrated_log.push_back(integrated_velocity);

                // Rotate inertial frame sensor measuremment to body frame
                accel_sensor = rot_in_world * accel_sensor;
                gyro_sensor = rot_in_world * gyro_sensor;

                // Filter update
                t1 = high_resolution_clock::now();
                state_estimator->updateSensors(accel_sensor, gyro_sensor, F[i], rot_in_world);
                state_estimator->updateStep();
                t2 = high_resolution_clock::now();
                ms_double = t2 - t1;
                timings(cnt) = ms_double.count();
                cnt++;
                filter_state_log.push_back(state_estimator->m_state);
                filter_cov_log.push_back(state_estimator->m_P);
                // std::cout << "True State: \n" << velocity.transpose() << endl;
                // std::cout << "Filter State: \n" << state_estimator->m_state.transpose() << endl << endl;
            }
        }
            
    }

    std::cout << "Average filter update time: " << timings.sum() / timings.size() << " ms" << std::endl;
    std::cout << "Number of filter update: " << timings.size() << std::endl;

    // Draw plots
    Vec time_vector = linspace(0, N * sim_dt, N);
    Vec filter_time_vector = linspace(0, N * sim_dt, n_filter_samples);

    // Convert ground truth, state estimate, and covariance values to plot vectors
    Vec vx_true(N), vy_true(N), vz_true(N), wx_true(N), wy_true(N), wz_true(N);
    Vec vx_est(n_filter_samples), vy_est(n_filter_samples), vz_est(n_filter_samples), 
            wx_est(n_filter_samples), wy_est(n_filter_samples), wz_est(n_filter_samples);
    Vec vx_upper_est(n_filter_samples), vy_upper_est(n_filter_samples), vz_upper_est(n_filter_samples), 
            wx_upper_est(n_filter_samples), wy_upper_est(n_filter_samples), wz_upper_est(n_filter_samples);
    Vec vx_lower_est(n_filter_samples), vy_lower_est(n_filter_samples), vz_lower_est(n_filter_samples), 
            wx_lower_est(n_filter_samples), wy_lower_est(n_filter_samples), wz_lower_est(n_filter_samples);
    Vec vx_naive(n_filter_samples), vy_naive(n_filter_samples), vz_naive(n_filter_samples), 
            wx_naive(n_filter_samples), wy_naive(n_filter_samples), wz_naive(n_filter_samples);

    for (int i = 0; i < n_filter_samples; ++i) {
        vx_est[i] = filter_state_log[i](0);
        vy_est[i] = filter_state_log[i](1);
        vz_est[i] = filter_state_log[i](2);
        wx_est[i] = filter_state_log[i](3);
        wy_est[i] = filter_state_log[i](4);
        wz_est[i] = filter_state_log[i](5);

        vx_upper_est[i] = filter_state_log[i](0) + sqrt(filter_cov_log[i](0, 0));
        vy_upper_est[i] = vy_est[i] + sqrt(filter_cov_log[i](1, 1));
        vz_upper_est[i] = vz_est[i] + sqrt(filter_cov_log[i](2, 2));
        wx_upper_est[i] = wx_est[i] + sqrt(filter_cov_log[i](3, 3));
        wy_upper_est[i] = wy_est[i] + sqrt(filter_cov_log[i](4, 4));
        wz_upper_est[i] = wz_est[i] + sqrt(filter_cov_log[i](5, 5));

        vx_lower_est[i] = filter_state_log[i](0) - sqrt(filter_cov_log[i](0, 0));
        vy_lower_est[i] = vy_est[i] - sqrt(filter_cov_log[i](1, 1));
        vz_lower_est[i] = vz_est[i] - sqrt(filter_cov_log[i](2, 2));
        wx_lower_est[i] = wx_est[i] - sqrt(filter_cov_log[i](3, 3));
        wy_lower_est[i] = wy_est[i] - sqrt(filter_cov_log[i](4, 4));
        wz_lower_est[i] = wz_est[i] - sqrt(filter_cov_log[i](5, 5));

        vx_naive[i] = vel_integrated_log[i](0);
        vy_naive[i] = vel_integrated_log[i](1);
        vz_naive[i] = vel_integrated_log[i](2);
        wx_naive[i] = vel_integrated_log[i](3);
        wy_naive[i] = vel_integrated_log[i](4);
        wz_naive[i] = vel_integrated_log[i](5);
    }

    for (int i = 0; i < N; ++i) {
        vx_true[i] = ground_truth_log[i](0);
        vy_true[i] = ground_truth_log[i](1);
        vz_true[i] = ground_truth_log[i](2);
        wx_true[i] = ground_truth_log[i](3);
        wy_true[i] = ground_truth_log[i](4);
        wz_true[i] = ground_truth_log[i](5);
    }

    Plot plot0;
    // plot0.size(1280, 720);
    plot0.fontName("Palatino");
    plot0.fontSize(14);
    plot0.legend()
        .atOutsideBottom()
        .displayHorizontal()
        .displayExpandWidthBy(2);
    // plot0.palette("parula");
    plot0.xlabel("Time (s)");
    plot0.ylabel("Inertial X Velocity (m/s)");
    plot0.drawCurve(filter_time_vector, vx_est).label("vx est").lineWidth(2);
    // plot0.drawCurve(filter_time_vector, vx_upper_est);
    // plot0.drawCurvesFilled(filter_time_vector, vx_lower_est, vx_upper_est).label("vx 1-sigma").lineWidth(4);
    plot0.drawCurve(time_vector, vx_true).label("vx true").lineWidth(2);
    // plot0.drawCurve(time_vector, vx_naive).label("vx integrated").lineWidth(2);

    Plot plot1;
    // plot1.size(1280, 720);
    plot1.fontName("Palatino");
    plot1.fontSize(14);
    plot1.legend()
        .atOutsideBottom()
        .displayHorizontal()
        .displayExpandWidthBy(2);
    // plot0.palette("parula");
    plot1.xlabel("Time (s)");
    plot1.ylabel("Inertial Y Velocity (m/s)");
    plot1.drawCurve(filter_time_vector, vy_est).label("vy est").lineWidth(2);
    // plot0.drawCurve(filter_time_vector, vx_upper_est);
    // plot0.drawCurvesFilled(filter_time_vector, vx_lower_est, vx_upper_est).label("vx 1-sigma").lineWidth(4);
    plot1.drawCurve(time_vector, vy_true).label("vy true").lineWidth(2);
    // plot0.drawCurve(time_vector, vx_naive).label("vx integrated").lineWidth(2);

    Plot plot2;
    // plot2.size(1280, 720);
    plot2.fontName("Palatino");
    plot2.fontSize(14);
    plot2.legend()
        .atOutsideBottom()
        .displayHorizontal()
        .displayExpandWidthBy(2);
    // plot0.palette("parula");
    plot2.xlabel("Time (s)");
    plot2.ylabel("Inertial Z Velocity (m/s)");
    plot2.drawCurve(filter_time_vector, vz_est).label("vz est").lineWidth(2);
    // plot0.drawCurve(filter_time_vector, vx_upper_est);
    // plot0.drawCurvesFilled(filter_time_vector, vx_lower_est, vx_upper_est).label("vx 1-sigma").lineWidth(4);
    plot2.drawCurve(time_vector, vz_true).label("vz true").lineWidth(2);
    // plot0.drawCurve(time_vector, vx_naive).label("vx integrated").lineWidth(2);

    Plot plot3;
    // plot3.size(1280, 720);
    plot3.fontName("Palatino");
    plot3.fontSize(14);
    plot3.legend()
        .atOutsideBottom()
        .displayHorizontal()
        .displayExpandWidthBy(2);
    // plot0.palette("parula");
    plot3.xlabel("Time (s)");
    plot3.ylabel("Inertial X Angular Velocity (rad/s)");
    plot3.drawCurve(filter_time_vector, wx_est).label("wx est").lineWidth(2);
    plot3.drawCurve(time_vector, wx_true).label("wx true").lineWidth(2);

    Plot plot4;
    // plot4.size(1280, 720);
    plot4.fontName("Palatino");
    plot4.fontSize(14);
    plot4.legend()
        .atOutsideBottom()
        .displayHorizontal()
        .displayExpandWidthBy(2);
    // plot0.palette("parula");
    plot4.xlabel("Time (s)");
    plot4.ylabel("Inertial Y Angular Velocity (rad/s)");
    plot4.drawCurve(filter_time_vector, wy_est).label("wy est").lineWidth(2);
    plot4.drawCurve(time_vector, wy_true).label("wy true").lineWidth(2);

    Plot plot5;
    // plot5.size(1280, 720);
    plot5.fontName("Palatino");
    plot5.fontSize(14);
    plot5.legend()
        .atOutsideBottom()
        .displayHorizontal()
        .displayExpandWidthBy(2);
    // plot0.palette("parula");
    plot5.xlabel("Time (s)");
    plot5.ylabel("Inertial Z Angular Velocity (rad/s)");
    plot5.drawCurve(filter_time_vector, wz_est).label("wz est").lineWidth(2);
    plot5.drawCurve(time_vector, wz_true).label("wz true").lineWidth(2);

    Figure fig = {{ plot0, plot1, plot2 },
                  { plot3, plot4, plot5 }};
    fig.title("Filter vs. Truth Comparison");
    fig.palette("dark2");
    fig.size(1280, 720);
    fig.show();

    // fig.save("ukf_results.pdf");
    fig.save("ukf_results.png");

}
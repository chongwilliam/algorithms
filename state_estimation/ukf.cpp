/**
 * @file main.cpp
 * @brief Runs an unscented kalman filter for SRBD (single rigid body dynamics).
*/

#include "ukf.h"

namespace Filter {

UKF::UKF(const double lambda,
			const double alpha,
            const double dt,
            const Vector6d& state,
			const Matrix6d& P, 
			const Matrix6d& Q,
			const Matrix6d& R,
            const double mass,
            const Matrix3d& I)
{
    // Initialize redis and smoothing filter 
    redis_client = new RedisClient();
    filter = new ButterworthFilter(3);  // cutoff frequency needs to be tuned 
    // double sampling_rate = 1. / 1000;  // if needed for filter cutoff computation 
    double cutoff_freq = 0.1;
    filter->setCutoffFrequency(cutoff_freq);

    // UKF parameters 
    m_L = 6;
    m_dt = dt;
    m_alpha = alpha;
    m_beta = 2;  // optimal for gaussian distribution 
    m_eta = sqrt(m_L + lambda);
    m_state = state;  // [linear velocity; angular velocity] in inertial frame 

    // Populate sigma point vector container
    int n_sigma_points = 1 + 2 * m_L;
    for (int i = 0; i < n_sigma_points; ++i) {
        m_state_sigma_points.push_back(Vector6d::Zero());
        m_state_sigma_points_prop.push_back(Vector6d::Zero());
        m_meas_sigma_points.push_back(Vector6d::Zero());
    }

    // Compute weights
    double c = (double) m_L + lambda;
    m_state_weights.push_back( lambda / c );
    m_cov_weights.push_back( lambda / c + (1 - m_alpha * m_alpha + m_beta) );
    for (int i = 0; i < n_sigma_points - 1; ++i) {
        m_state_weights.push_back( 0.5 / c );
        m_cov_weights.push_back( 0.5 / c );
    }

    // Covariance init
    m_P = P;
    m_Q = Q;
    m_R = R;

    // Inertia matrix init
    m_M.setZero();
    m_M.block<3, 3>(0, 0) = mass * Matrix3d::Identity();
    m_M.block<3, 3>(3, 3) = I;
    m_M_inv = m_M.inverse();
    m_I = I;
    m_gravity = Vector3d(0, 0, -9.81);  // replace with gravity estimate below sea level

}

// Replace with redis client calls for O1K
void UKF::updateSensors(const Vector3d& accel, 
                            const Vector3d& gyro, 
                            const Vector6d& tau,
                            const Matrix3d& rot_in_world)
{
    m_accel = accel;  // body frame
    m_gyro = gyro;  // body frame
    m_tau = tau;  // inertial frame
    m_rot_in_world = rot_in_world;  // from inertial frame to body frame 
}

void UKF::filterAccel()
{
    m_accel = filter->update(m_accel);
}

void UKF::updateVelocity()
{
    m_vel = m_rot_in_world * m_state.head(3) + m_accel * m_dt;  // "velocity" body measurement
}

void UKF::computeSigmaPoints(const Vector6d& state, const Matrix6d& P)
{
    m_state_sigma_points[0] = state;
    LLT<Matrix6d> llt(P);
    Matrix6d P_sqrt = llt.matrixL();
    Matrix6d sigma_matrix = m_eta * P_sqrt;
    for (int i = 0; i < m_L; ++i) {
        // m_state_sigma_points[i + 1] = state + sigma_matrix.row(i).transpose();  // 1 - 6
        // m_state_sigma_points[m_L + i + 1] = state - sigma_matrix.row(i).transpose();  // 7 - 12
        m_state_sigma_points[i + 1] = state + sigma_matrix.col(i);  // 1 - 6
        m_state_sigma_points[m_L + i + 1] = state - sigma_matrix.col(i);  // 7 - 12
    }
}

Vector6d UKF::propagateState(const Vector6d& state, 
                                const Vector6d& tau, 
                                const double dt)
{
    // Compute acceleration
    Vector3d w = state.tail(3);
    Vector6d nonlinear;
    nonlinear.head(3).setZero();  
    nonlinear.tail(3) = w.cross(m_I * w);
    Vector6d state_accel = m_M_inv * (tau - nonlinear);

    // Forward time integration
    return state + state_accel * dt;
}

void UKF::unscentedTransformDynamics()
{
    int n_points = m_state_sigma_points.size();
    
    // Propagate through system dynamics
    for (int i = 0; i < n_points; ++i) {
        m_state_sigma_points_prop[i] = propagateState(m_state_sigma_points[i], m_tau, m_dt);
    }

    // Compute apriori mean
    m_state.setZero();
    for (int i = 0; i < n_points; ++i) {
        m_state += m_state_weights[i] * m_state_sigma_points_prop[i];
    }

    // Compute apriori covariance
    m_P = m_Q;
    for (int i = 0; i < n_points; ++i) {
        m_P += m_cov_weights[i] * (m_state_sigma_points_prop[i] - m_state) * 
                                        (m_state_sigma_points_prop[i] - m_state).transpose();
    }
}

void UKF::unscentedTransformSensors()
{
    int n_points = m_meas_sigma_points.size();

    Matrix6d R = Matrix6d::Zero();
    R.block<3, 3>(0, 0) = m_rot_in_world;  // inertial to body frame 
    R.block<3, 3>(3, 3) = m_rot_in_world;

    // Observed body sensor readings from state sigma points 
    for (int i = 0; i < n_points; ++i) {
        m_meas_sigma_points[i] = R * m_state_sigma_points[i];
    }

    // Compute mean
    m_meas.setZero();
    for (int i = 0; i < n_points; ++i) {
        m_meas += m_state_weights[i] * m_meas_sigma_points[i];
    }

    // Compute P_yy
    m_Pyy = m_R;
    for (int i = 0; i < n_points; ++i) {
        m_Pyy += m_cov_weights[i] * (m_meas_sigma_points[i] - m_meas) 
                                            * (m_meas_sigma_points[i] - m_meas).transpose();
    }

    // Compute P_xy
    m_Pxy.setZero();
    for (int i = 0; i < n_points; ++i) {
        m_Pxy += m_cov_weights[i] * (m_state_sigma_points[i] - m_state) 
                                            * (m_meas_sigma_points[i] - m_meas).transpose();
    }
}

void UKF::updatePosterior()
{
    Vector6d y;  // body frame sensor measurements 
    y.head(3) = m_vel;
    y.tail(3) = m_gyro;
    Matrix6d K = m_Pxy * m_Pyy.inverse();
    m_state += K * (y - m_meas);
    m_P -= K * m_Pyy * K.transpose();
}

void UKF::updateStep()
{
    // filterAccel();
    updateVelocity();
    computeSigmaPoints(m_state, m_P);
    unscentedTransformDynamics();
    computeSigmaPoints(m_state, m_P);  // redraw sigma points (optional, but better results)
    unscentedTransformSensors();
    updatePosterior();
}

}  // namespace filter 

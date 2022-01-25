/**
 * @file main.cpp
 * @brief Runs an unscented kalman filter for SRBD (single rigid body dynamics).
*/

#include "ukf.h"

namespace filter {

UKF::UKF(const double lambda,
				const double alpha,
                const double dt,
				const Matrix6d& P, 
				const Matrix6d& Q,
				const Matrix6d& R,
                const double mass,
                const Matrix3d& I)
{
    // Initialize redis client
    redis_client = RedisClient();
    filter = ButterworthFilter(3, 1. / dt, 0.1);  // cutoff frequency needs to be tuned 

    m_L = 6;
    m_dt = dt;
    m_alpha = alpha;
    m_beta = 2;  // optimal for gaussian distribution 
    m_eta = sqrt(L + lambda);

    // Populate sigma point vector container
    int n_sigma_points = 1 + 2 * m_L;
    for (int i = 0; i < n_sigma_points; ++i) {
        m_state_sigma_points.push_back(Vector6d::Zeros());
        m_state_sigma_points_prop.push_back(Vector6d::Zeros());
        m_meas_sigma_points.push_back(Vector6d::Zeros());
    }

    // Compute weights
    m_state_weights.push_back( lambda / (m_L + lambda) );
    m_cov_weights.push_back( (lambda / (m_L + lambda)) + (1 - alpha * alpha + beta) );
    for (int i = 0; i < n_sigma_points - 1; ++i) {
        m_state_weights.push_back( 1 / (2 * (m_L + lambda)) );
        m_cov_weights.push_back( 1 / (2 * (m_L + lambda)) );
    }

    // Covariance init
    m_P = P;
    m_Q = Q;
    m_R = R;

    // Inertia matrix init
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
    m_accel = accel;
    m_gyro = gyro;
    m_tau = tau;
    m_rot_in_world = rot_in_world;  // from inertial frame to body frame 
}

Vector3d UKF::filterAccel()
{
    m_vel = m_state.head(3) + filter.update(m_accel) * m_dt;
}

void UKF::computeSigmaPoints(const Vector6d& state, const Matrix6d& P)
{
    m_sigma_points[0] = state;
    LLT<Matrix6d> llt(P);
    Matrix6d L = llt.matrixL();
    Matrix6d sigma_matrix = eta * L;
    for (int i = 0; i < 7; ++i) {
        m_sigma_points[i + 1] = state + sigma_matrix.col(i);  // 1 - 6
        m_sigma_points[L + i + 1] = state - sigma_matrix.col(i);  // 7 - 12
    }
}

Vector6d UKF::propagateState(const Vector6d& state, 
                                const Vector6d& tau, 
                                const double dt)
{
    // Compute acceleration
    Vector3d w = state.tail(3);
    Vector6d nonlinear;
    nonlinear.head(3) = m_gravity;
    nonlinear.tail(3) = w.cross(m_I * w);
    Vector6d state_accel = m_M_inv * (tau - nonlinear);

    // Forward time integration
    return state + state_accel * dt;
}

void UKF::unscentedTransformDynamics()
{
    // Propagate through system dynamics
    int n_points = m_state_sigma_points.size();
    for (size_t i = 0; i < n_points; ++i) {
        m_state_sigma_points_prop[i] = propagateState(m_state_sigma_points[i], m_tau, m_dt);
    }

    // Compute apriori mean
    m_state.setZero();
    for (size_t i = 0; i < n_points; ++i) {
        m_state += m_state_weights[i] * m_state_sigma_points_prop[i];
    }

    // Compute apriori covariance
    m_P = m_Q;
    for (size_t i = 0; i < n_points; ++i) {
        m_P += m_cov_weights[i] * (m_state_sigma_points_prop[i] - m_state) * 
                                        (m_state_sigma_points_prop[i] - m_state).transpose();
    }
}

void UKF::unscentedTransformSensors()
{
    int n_points = m_meas_sigma_points.size();

    for (size_t i = 0; i < n_points; ++i) {
        m_meas_sigma_points[i] = m_rot_in_world * m_state_sigma_points[i];
    }

    // Compute mean
    m_meas.setZero();
    for (size_t i = 0; i < n_points; ++i) {
        m_meas += m_state_weights[i] * m_meas_sigma_points[i];
    }

    // Compute P_yy
    m_Pyy = m_R;
    for (size_t i = 0; i < n_points; ++i) {
        m_Pyy += m_cov_weights[i] * (m_meas_sigma_points[i] - m_meas) 
                                            * (m_meas_sigma_points[i] - m_meas).transpose();
    }

    // Compute P_xy
    m_Pxy.setZero();
    for (size_t i = 0; i < n_points; ++i) {
        m_Pxy += m_cov_weights[i] * (m_state_sigma_points[i] - m_state) 
                                            * (m_meas_sigma_points[i] - m_meas).transpose();
    }
}

void UKF::updatePosterior()
{
    Vector6d y;
    y.head(3) = m_vel;
    y.tail(3) = m_gyro;
    Matrix6d K = m_Pxy * m_Pxy.inverse();
    m_state += K * (y - m_meas_mean);
    m_P -= K * m_Pyy * K.transpose();
}

}  // namespace filter 

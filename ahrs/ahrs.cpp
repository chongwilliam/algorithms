/**
 * @file ahrs.cpp
 * @brief Class source files.
 * 
 */

#include "ahrs.h"

namespace ahrs {

AHRS::AHRS(const double lambda, const double alpha, const Matrix7d& P, const Matrix7d& Q, const Matrix7d& R)
{
    L = 7;
    m_alpha = alpha;
    m_beta = 2;  // optimal for gaussian distribution 
    m_eta = sqrt(L + lambda);

    // Populate sigma point vector container
    int n_sigma_points = 1 + 2 * L;
    m_sigma_points.reserve(n_sigma_points);
    for (int i = 0; i < n_sigma_points; ++i) {
        m_sigma_points.push_back(Vector7d::Zeros());
        m_sigma_points_prop.push_back(Vector7d::Zeros());
    }

    // Compute weights
    m_sigma_weights.reserve(n_sigma_points);
    m_sigma_weights.push_back( (lambda / (L + lambda)) + (1 - alpha * alpha + beta) );
    for (int i = 0; i < n_sigma_points - 1; ++i) {
        m_sigma_weights.push_back( 1 / (2 * (L + lambda)) );
    }

    // Covariance initialization
    m_P = P;
    m_Q = Q;
    m_R = R;

}

void AHRS::updateSensors(const Vector3d& accel, const Vector3d& gyro, const Vector3d& mag)
{
    m_accel = accel;
    m_gyro = gyro;
    m_mag = mag;
}

void AHRS::updateSigmaPoints(const Vector7d& x, const Matrix7d& P)
{
    m_sigma_points[0] = x;
    LLT<Matrix7d> llt(P);
    Matrix7d L = llt.matrixL();
    Matrix7d sigma_matrix = eta * L;
    for (int i = 0; i < 7; ++i) {
        m_sigma_points[i + 1] = x + sigma_matrix.col(i);  // 1 - 7
        m_sigma_points[L + i + 1] = x - sigma_matrix.col(i);  // 8 - 14
    }

}

Vector7d AHRS::propagateState(const Vector7d& x, const int dt)
{
    Vector7d x_prop = x;
    double w_x = m_gyro(0) - x[4];
    double w_y = m_gyro(1) - x[5];
    double w_z = m_gyro(2) - x[6];
    double w_delta_norm = 0.5 * sqrt( (w_x * dt) * (w_x * dt) + 
                                        (w_y * dt) * (w_y * dt) + 
                                        (w_z * dt) * (w_z * dt) );
    Matrix4d omega;
    omega << 0, - w_x, -w_y, -w_z,
                w_x, 0, w_z, -w_y,
                w_y, -w_z, 0, w_x,
                w_z, w_y, -w_x, 0;

    x_prop.head(4) = (Matrix4d::Identity() * cos(w_delta_norm) + sin(w_delta_norm) * w_delta_norm * omega) * q;
    return x_prop;
}

Vector4d AHRS::unscentedTransform()
{
    // Propagate sigma points 
    for (int i = 0; i < sigma_points.size(); ++i) {
        sigma_points_prop[i] = propagateState(x_sigma[i]);
    }

    // Compute pre-innovation mean
    m_x.setZero();
    for (int i = 0; i < m_sigma_points_prop; ++i) {
        m_x += m_sigma_weights[i] * m_sigma_points_prop[i];
    }

    // Compute pre-innovation covariance
    m_P = m_Q;
    for (int i = 0; i < m_sigma_points_prop; ++i) {
        m_P += m_sigma_weights[i] * (m_sigma_points_prop[i] - m_x) * (m_sigma_points_prop[i] - m_x).transpose(); 
    }
}

Vector4d rotationToQuaternion(const Matrix3d& m)
{
    double tr = m(0,0) + m(1,1) + m(2,2);
    double qw, qx, qy, qz;
    Vector4d q;

    if (tr > 0) {
      double S = sqrt(tr+1.0) * 2; // S=4*qw
      qw = 0.25 * S;
      qx = (m(2,1) - m(1,2)) / S;
      qy = (m(0,2) - m(2,0)) / S;
      qz = (m(1,0) - m(0,1)) / S;
    } else if ((m(0,0) > m(1,1)) && (m(0,0) > m(2,2))) {
      double S = sqrt(1.0 + m(0,0) - m(1,1) - m(2,2)) * 2; // S=4*qx
      qw = (m(2,1) - m(1,2)) / S;
      qx = 0.25 * S;
      qy = (m(0,1) + m(1,0)) / S;
      qz = (m(0,2) + m(2,0)) / S;
    } else if (m(1,1) > m(2,2)) {
      double S = sqrt(1.0 + m(1,1) - m(0,0) - m(2,2)) * 2; // S=4*qy
      qw = (m(0,2) - m(2,0)) / S;
      qx = (m(0,1) + m(1,0)) / S;
      qy = 0.25 * S;
      qz = (m(1,2) + m(2,1)) / S;
    } else {
      double S = sqrt(1.0 + m(2,2) - m(0,0) - m(1,1)) * 2; // S=4*qz
      qw = (m(1,0) - m(0,1)) / S;
      qx = (m(0,2) + m(2,0)) / S;
      qy = (m(1,2) + m(2,1)) / S;
      qz = 0.25 * S;
    }

    q << qw, qx, qy, qz;
    return q.normalized();
}

Vector6d estimateMeasure(const Vector4d& quat) 
{
    Vector4d quat_inv = inverseQuat(quat);
    Matrix3d R = quaternionToRotation(quat_inv);
    Vector6d z_est;
    z_est.head(3) = R * m_accel;
    z_est.tail(3) = R * m_mag;
    return z_est;
}

Vector4d observationModel(const Vector7d& x) 
{
    Vector4d obs;
    obs(0) = 2 * (x(1) * x(3) - x(0) * x(2));
    obs(1) = 2 * (x(2) * x(3) + x(0) * x(1));
    obs(2) = (x(0) * x(0) + x(1) * x(1) + x(2) * x(2) + x(3) * x(3));
    obs(3) = 2 * (x(1) * x(2) + x(0) * x(3));
    return obs;
}

Vector4d AHRS::measurementState(const Vector3d& gravity, const Vector3d& mag_ref)
{
    Matrix3d B = Matrix3d::Zeros();
    B += gravity * m_accel.transpose();
    B += mag_ref * m_mag.transpose();
    JacobiSVD<Matrix3d> svd(B, ComputeThinU | ComputeThinV);
    Vector3d diag_elem = Vector3d(1, 1, svd.matrixU().determinant() * svd.matrixV().determinant());
    Matrix3d M = svd.matrixU() * svd.singularValues().asDiagonal() * svd.matrixV();  // or V'
    Matrix3d R = svd.matrixU() * M * svd.matrixV();
    return rotationToQuaternion(R);
}





}  // namespace ahrs
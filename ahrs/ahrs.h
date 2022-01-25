/**
 * @file ahrs.h
 * @brief Class definitions for AHRS using UKF multiplicative filter.
 * 
*/

#ifndef AHRS_H_
#define AHRS_H_

#include <eigen3/Eigen/Dense>
using Vector4d = Eigen::Matrix<double, 4, 1>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector7d = Eigen::Matrix<double, 7, 1>;
using Matrix4d = Eigen::Matrix<double, 4, 4>;
using Matrix7d = Eigen::Matrix<double, 7, 7>;
using namespace Eigen;

#include "RedisClient.h"

namespace ahrs {

class AHRS
{
	public:
		AHRS(const double& lambda, );

		void updateSensors(const Vector3d& accel, const Vector3d& gyro, const Vector3d& mag);
		void updateSigmaPoints();
		void unscentedTransform();



	private:
		int L;
		Vector7d m_x;  // state expectation 
		Matrix7d m_P;  // state covariance (qw, qx, qy, qz, bx, by, bz)
		Matrix7d m_Q;
		Matrix7d m_Pyy;  // measure covariance matrix
		Matrix7d m_Pxy;  // cross-covariance matrix 
		Matrix6d m_R;  // measurement noise covariance (acc_x, acc_y, acc_z, mag_x, mag_y, mag_z)
		double m_alpha;
		double m_beta;
		double m_eta;  // sigma point scaling 
		vector<Vector7d> m_sigma_points;
		vector<Vector7d> m_sigma_points_prop;  // propagated sigma points 
		vector<double> m_sigma_weights;
		Vector3d m_accel;
		Vector3d m_gyro;
		Vector3d m_mag;


};

}  // namespace ahrs


#endif
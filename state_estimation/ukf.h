/**
 * @file ukf.h
 * @brief UKF class header
 * 
*/

#include <eigen3/Eigen/Dense>
#include "redis/RedisClient.h"
#include "filter/ButterworthFilter.h"
#include <iostream>

using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using namespace Eigen;

namespace filter {

class UKF
{
	public:
		UKF(const double lambda,
				const double alpha,
				const double dt,
				const Matrix6d& P, 
				const Matrix6d& Q,
				const Matrix6d& R,
				const double mass,
				const Matrix3d& I);

		void updateSensors(const Vector3d& accel, 
								const Vector3d& gyro,
								const Vector6d& tau,
								const Matrix3d& rot_in_world);
		void computeSigmaPoints(const Vector6d& state, const Matrix6d& P);
		Vector6d propagateState(const Vector6d& state, const double dt);
		Vector6d unscentedTransformDynamics();
		Vector6d unscentedTransformSensors();







	private:
		// UKF setup
		int m_L;
		double m_dt;
		double m_alpha;
		double m_beta;
		double m_eta;	

		// State and noise stats
		Vector6d m_state;  // E[state]
		Vector6d m_meas;  // E[measurement]
		Matrix6d m_P;
		Matrix6d m_Q;
		Matrix6d m_R;
		Matrix6d m_Pyy;
		Matrix6d m_Pxy;
	
		// Sigma point containers
		vector<Vector6d> m_state_sigma_points;
		vector<Vector6d> m_state_sigma_points_prop;
		vector<Vector6d> m_meas_sigma_points;
		vector<double> m_state_weights;
		vector<double> m_cov_weights;

		// Sensors
		Vector3d m_accel;
		Vector3d m_vel;  // filtered and integrated accel
		Vector3d m_gyro;
		Vector6d m_tau;
		Matrix3d m_rot_in_world;  // from inertial frame to body frame 

		// Inertia
		Matrix6d m_M;
		Matrix6d m_M_inv;
		Matrix3d m_I;

		// Externals 
		RedisClient redis_client;
		ButterworthFilter filter;

};

}  // namespace filter
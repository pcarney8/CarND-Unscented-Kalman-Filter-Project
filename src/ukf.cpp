#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1.2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1.2;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  time_us_ = 0;
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;

  //set vector for weights_
  weights_ = VectorXd(2*n_aug_+1);
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);
  Xsig_pred_.fill(0);

  double weight_0 = lambda_/(lambda_ + n_aug_);
  weights_(0) = weight_0;

  for (int i=1; i<2*n_aug_+1; i++) {
    double weight = 0.5/(n_aug_ + lambda_);
    weights_(i) = weight;
  }

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    if (!is_initialized_) {

      cout << "Initializing UKF..." << endl;
      time_us_ = meas_package.timestamp_;

      if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        cout << "first measurement is RADAR" << endl;

        float rho = meas_package.raw_measurements_[0];
        float phi = meas_package.raw_measurements_[1];
        float rho_dot = meas_package.raw_measurements_[2];

        cout << "convert to cartesian" << endl;
        float px = rho * cos(phi);
        float py = rho * sin(phi);

        x_ << px, py, 5, 0, 0;

      }
      else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        /**
        Initialize state.
        */
        cout << "first measurement is LIDAR" << endl;

        x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 5, 0, 0;
      }

      cout << "px: " << x_[0] << endl;
      cout << "py: " << x_[1] << endl;
      cout << "v: " << x_[2] << endl;
      cout << "yaw: " << x_[3] << endl;
      cout << "yaw_dot: " << x_[4] << endl;

      // done initializing, no need to predict or update
      is_initialized_ = true;
      return;
    }
    /*****************************************************************************
     *  Prediction
     ****************************************************************************/
    float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
    time_us_ = meas_package.timestamp_;
//    cout << "compute the time elapsed between the current and previous measurements, dt: " << dt << endl;

    Prediction(dt);
    /*****************************************************************************
     *  Update
     ****************************************************************************/
    if(meas_package.sensor_type_ == MeasurementPackage::RADAR){
        cout << "UPDATE RADAR!" << endl;
        UpdateRadar(meas_package);
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER){
        cout << "UPDATE LIDAR!" << endl;
        UpdateLidar(meas_package);
    }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  //generate augmented sigma points
  VectorXd x_aug_ = VectorXd(n_aug_);
  x_aug_.fill(0);
  MatrixXd Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug_.fill(0);
  MatrixXd P_aug_ = MatrixXd(n_aug_, n_aug_);
  P_aug_.fill(0);
  P_.fill(0);
  P_ << 0.3, 0, 0, 0, 0,
        0, 0.3, 0, 0, 0,
        0, 0, 0.3, 0, 0,
        0, 0, 0, 0.3, 0,
        0, 0, 0, 0, 0.3;

    //create augmented mean state
    x_aug_ << x_,
             0,
             0;

//    cout << "x_aug_: " << endl << x_aug_ << endl;

    //create augmented covariance matrix
    MatrixXd Q = MatrixXd(2,2);
    Q << std_a_*std_a_, 0,
         0, std_yawdd_*std_yawdd_;

//    cout << "Q: " << endl << Q << endl;

    //create square root matrix
    P_aug_.topLeftCorner(n_x_, n_x_) = P_;
    P_aug_.bottomRightCorner(2,2) = Q;

//    cout << "P_aug_: " << endl << P_aug_ << endl;

    MatrixXd sqrt_P_aug = P_aug_.llt().matrixL();
    double n_aug_lambda = sqrt(lambda_ + n_aug_);

    //create augmented sigma points
    Xsig_aug_.col(0) = x_aug_;

    for(int i = 0; i < n_aug_; i++){
        int first = i+1;
        int second = n_aug_+first;

        Xsig_aug_.col(first) = x_aug_ + n_aug_lambda*sqrt_P_aug.col(i);
        Xsig_aug_.col(second) = x_aug_ - n_aug_lambda*sqrt_P_aug.col(i);
    }

  //predict sigma points
  for(int i = 0; i < Xsig_pred_.cols(); i++){
      double p_x = Xsig_aug_(0,i);
      double p_y = Xsig_aug_(1,i);
      double v = Xsig_aug_(2,i);
      double psi = Xsig_aug_(3,i);
      double psi_d = Xsig_aug_(4,i);
      double nu_a = Xsig_aug_(5,i);
      double nu_psidd = Xsig_aug_(6,i);

      double dt_2 = 0.5*delta_t*delta_t;

      if(psi_d == 0){
          Xsig_pred_(0,i) = p_x + v*cos(psi)*delta_t + dt_2*cos(psi)*nu_a;
          Xsig_pred_(1,i) = p_y + v*sin(psi)*delta_t + dt_2*sin(psi)*nu_a;
      } else {
          Xsig_pred_(0,i) = p_x + (v/psi_d)*(sin(psi + psi_d*delta_t) - sin(psi)) + dt_2*cos(psi)*nu_a;
          Xsig_pred_(1,i) = p_y + (v/psi_d)*(-cos(psi + psi_d*delta_t) + cos(psi)) + dt_2*sin(psi)*nu_a;
      }

	  Xsig_pred_(2,i) = v + delta_t*nu_a;
	  Xsig_pred_(3,i) = psi + psi_d*delta_t + dt_2*nu_psidd;
	  Xsig_pred_(4,i) = psi_d + delta_t*nu_psidd;

  }
//      cout << "Xsig_pred_: " << endl << Xsig_pred_ << endl;

    //predict mean and covariance
    x_.fill(0);
    for(int i = 0; i < Xsig_pred_.cols(); i++){
      x_ = x_ + weights_(i)*Xsig_pred_.col(i);
    }

    for(int i = 0; i < Xsig_pred_.cols(); i++){
      // state difference
      VectorXd x_diff = Xsig_pred_.col(i) - x_;
      //angle normalization
      while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
      while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

      P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
    }

    cout << "prediction step, x: " << endl << x_ << endl;
    cout << "prediction step, P: " << endl << P_ << endl;

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

   //set measurement dimension, lidar can measure px and py
  int n_z = 2;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0);
  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0);
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0);
  VectorXd z = VectorXd(n_z);
  z.fill(0);

  // put px and py into z vector
  z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1];

  /**
  * predict measurement
  **/

  //transform sigma points into measurement space
  for(int i = 0; i < Xsig_pred_.cols(); i++){
      double p_x = Xsig_pred_(0,i);
      double p_y = Xsig_pred_(1,i);

      Zsig(0,i) = p_x;
      Zsig(1,i) = p_y;
  }

  //calculate mean predicted measurement
  for(int i = 0; i < Zsig.cols(); i++){
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //calculate measurement covariance matrix S
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_laspx_*std_laspx_, 0,
       0, std_laspy_*std_laspy_;

  for(int i = 0; i < Zsig.cols(); i++){
      VectorXd z_diff = Zsig.col(i) - z_pred;
      S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  S = S + R;

  /**
  * update measurement
  **/

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0);
  MatrixXd K = MatrixXd(n_x_, n_z);
  K.fill(0);

  //calculate cross correlation matrix
  for(int i = 0; i < Zsig.cols(); i++){
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    VectorXd z_diff = Zsig.col(i) - z_pred;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K;
  K = Tc * S.inverse();

  //update state mean and covariance matrix
  VectorXd z_diff = z - z_pred;

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  cout << "update step, x: " << endl << x_ << endl;
  cout << "update step, P: " << endl << P_ << endl;

  /**
  *calculate NIS
  **/

  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;

  cout << "LIDAR NIS: " << endl << NIS_laser_ << endl;

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
  Zsig.fill(0);
  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0);
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0);
  // actual measurement values
  VectorXd z = VectorXd(n_z);
  z.fill(0);

  // put rho, phi, and rho_dot into z vector
  z << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], meas_package.raw_measurements_[2];

  /**
  * predict measurement
  **/

  //transform sigma points into measurement space
  for(int i = 0; i < Xsig_pred_.cols(); i++){
      double p_x = Xsig_pred_(0,i);
      double p_y = Xsig_pred_(1,i);
      double v = Xsig_pred_(2,i);
      double psi = Xsig_pred_(3,i);

      Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);
      Zsig(1,i) = atan2(p_y,p_x);
      Zsig(2,i) = (p_x*cos(psi)*v + p_y*sin(psi)*v)/(sqrt(p_x*p_x + p_y*p_y));
  }

  //calculate mean predicted measurement
  for(int i = 0; i < Zsig.cols(); i++){
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //calculate measurement covariance matrix S
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_radr_*std_radr_, 0, 0,
       0, std_radphi_*std_radphi_, 0,
       0, 0, std_radrd_*std_radrd_;

  for(int i = 0; i < Zsig.cols(); i++){
      VectorXd z_diff = Zsig.col(i) - z_pred;
          //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
      S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  S = S + R;

  /**
  * update measurement
  **/

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0);
  MatrixXd K = MatrixXd(n_x_, n_z);
  K.fill(0);

  //calculate cross correlation matrix
  for(int i = 0; i < Zsig.cols(); i++){
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K;
  K = Tc * S.inverse();

  //update state mean and covariance matrix
  VectorXd z_diff = z - z_pred;
  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

  cout << "update step, x: " << endl << x_ << endl;
  cout << "update step, P: " << endl << P_ << endl;

  /**
  *calculate NIS
  **/

  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;

  cout << "RADAR NIS: " << endl << NIS_radar_ << endl;

}

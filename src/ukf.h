#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include "tools.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

//#define DEBUG_OUTPUT

class UKF
{
public:

    ///* initially set to false, set to true in first call of ProcessMeasurement
    bool is_initialized_;

    ///* if this is false, laser measurements will be ignored (except for init)
    bool use_laser_;

    ///* if this is false, radar measurements will be ignored (except for init)
    bool use_radar_;

    ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
    VectorXd x_;

    //augmented state vector: [posx, posy, vel_ab, yaw_angle, yaw_rate, longitudinal acceleration, yaw_acceleration;]
    VectorXd x_aug_;

    ///* state covariance matrix
    MatrixXd P_;

    MatrixXd P_aug_;

    ///* predicted sigma points matrix
    MatrixXd Xsig_pred_;

    ///* time when the state is true, in us
    long long time_us_;

    ///* Process noise standard deviation longitudinal acceleration in m/s^2
    double std_a_;

    ///* Process noise standard deviation yaw acceleration in rad/s^2
    double std_yawdd_;

    MatrixXd Q_;//process noise matrix;

    ///* Laser measurement noise standard deviation position1 in m
    double std_laspx_;

    ///* Laser measurement noise standard deviation position2 in m
    double std_laspy_;

    ///* Radar measurement noise standard deviation radius in m
    double std_radr_;

    ///* Radar measurement noise standard deviation angle in rad
    double std_radphi_;

    ///* Radar measurement noise standard deviation radius change in m/s
    double std_radrd_;

    MatrixXd R_radar_;

    MatrixXd R_laser_;

    ///* Weights of sigma points
    VectorXd weights_;

    ///* State dimension
    int n_x_;

    ///* Augmented state dimension
    int n_aug_;

    ///* Sigma point spreading parameter
    double lambda_;

    ///* the current NIS for radar
    double NIS_radar_;

    ///* the current NIS for laser
    double NIS_laser_;

    /**
     * Constructor
     */
    UKF();

    /**
     * Destructor
     */
    virtual ~UKF();

    void ProcessMeasurement(const MeasurementPackage &meas_package);

    void Prediction(const double delta_t);

    void UpdateLidar(const MeasurementPackage &meas_package);

    void UpdateRadar(const MeasurementPackage &meas_package);

    void PredictRadarMeasurement(MatrixXd *Zsig_out, VectorXd *z_pred_out, MatrixXd *S_out);

    void UpdateRadarState(const MatrixXd &Zsig, const VectorXd &z_pred, const MatrixXd &S, const MeasurementPackage &meas_package);

    void GenerateAugmentedSigmaPoints(MatrixXd *Xaug_sig_out);

    void PredictSigmaPoints(const MatrixXd &Xaug_sig, const double delta_t, MatrixXd *Xsig_pred_out);

    void PredictMeanAndCovariance();

    double NormalizeAngle(double &angle)
    {
        //        while (angle>M_PI)
        //        {
        //            angle -= 2*M_PI;
        //        }
        //
        //        while (angle<-M_PI)
        //        {
        //            angle += 2*M_PI;
        //        }

        angle = angle - ceil((angle - M_PI) / (M_PI * 2.0)) * M_PI * 2.0;
        return angle;
    }

    void GenerateWeights()
    {
        weights_ = VectorXd(2 * n_aug_ + 1);
        weights_(0) = lambda_ / (lambda_ + n_aug_);
        for (int i = 1; i < 2 * n_aug_ + 1; ++i)
        {
            weights_(i) = 0.5 / (lambda_ + n_aug_);
        }
    }
};

#endif /* UKF_H */

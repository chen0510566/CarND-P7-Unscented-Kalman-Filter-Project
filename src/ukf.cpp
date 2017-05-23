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
UKF::UKF()
{
    n_x_ = 5;
    n_aug_ = 7;

    is_initialized_ = false;

    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(n_x_);
    x_.fill(0.0);

    x_aug_ = VectorXd(n_aug_);
    x_aug_.fill(0.0);

    // initial covariance matrix
    P_ = MatrixXd(n_x_, n_x_);
    P_.fill(0.0);
    P_(0, 0) = 1.0;
    P_(1, 1) = 1.0;
    P_(2, 2) = 1.0;
    P_(3, 3) = 1.0;
    P_(4, 4) = 1.0;

    P_aug_ = MatrixXd(n_aug_, n_aug_);
    P_aug_.fill(0.0);

    //process covariance matrix;
    /**@todo to be tuned*/
    std_a_ = 2.5;// Process noise standard deviation longitudinal acceleration in m/s^2
    std_yawdd_ = 2.0;// Process noise standard deviation yaw acceleration in rad/s^2
    Q_ = MatrixXd(2, 2);
    Q_.fill(0.0);
    Q_(0, 0) = std_a_ * std_a_;
    Q_(1, 1) = std_yawdd_ * std_yawdd_;

    //laser covariance matrix;
    std_laspx_ = 0.15;// Laser measurement noise standard deviation position1 in m
    std_laspy_ = 0.15;// Laser measurement noise standard deviation position2 in m
    R_laser_ = MatrixXd(2, 2);
    R_laser_.fill(0.0);
    R_laser_(0, 0) = std_laspx_*std_laspx_;
    R_laser_(1, 1) = std_laspy_*std_laspy_;


    //radar covariance matrix;
    std_radr_ = 0.3;// Radar measurement noise standard deviation radius in m
    std_radphi_ = 0.03;// Radar measurement noise standard deviation angle in rad
    std_radrd_ = 0.3;// Radar measurement noise standard deviation radius change in m/s
    R_radar_ = MatrixXd(3, 3);
    R_radar_.fill(0.0);
    R_radar_(0, 0) = std_radr_*std_radr_;
    R_radar_(1, 1) = std_radphi_*std_radphi_;
    R_radar_(2, 2) = std_radrd_*std_radrd_;

    lambda_ = 3 - n_aug_;

    GenerateWeights();
}

UKF::~UKF()
{
}

/**@brief process measurement;
 *
 * process measurement:
 * 1. predict state;
 * 2. update the state by the measurement;
 *
 * @param meas_package [IN]: The latest measurement data of either radar or laser.
 */
void UKF::ProcessMeasurement(const MeasurementPackage& meas_package)
{
    //if donot use laser;
    if (MeasurementPackage::LASER == meas_package.sensor_type_ && !use_laser_)
    {
        return;
    }

    //if donot use radar;
    if (MeasurementPackage::RADAR == meas_package.sensor_type_ && !use_radar_)
    {
        return;
    }

    if (!is_initialized_)
    {
        //set initial state here;
        time_us_ = meas_package.timestamp_;
        is_initialized_ = true;
        if (MeasurementPackage::RADAR == meas_package.sensor_type_)
        {
            const double rho = meas_package.raw_measurements_[0];
            const double phi = meas_package.raw_measurements_[1];
            const double rho_dot = meas_package.raw_measurements_[2];

            x_ << rho * cos(phi), rho * sin(phi), rho_dot, 0.0, 0.0;
        }

        if(MeasurementPackage::LASER == meas_package.sensor_type_)
        {
            x_(0) = meas_package.raw_measurements_[0];
            x_(1) = meas_package.raw_measurements_[1];
        }
        return;
    }

    double delta_t = (meas_package.timestamp_ - time_us_) * 1e-6;//convert to second;

    //predict state and covariance matrix;
    Prediction(delta_t);

    //update state and covariance matrix by sensor measurement;
    if (MeasurementPackage::RADAR == meas_package.sensor_type_)
    {
#ifdef DEBUG_OUTPUT
        std::cerr<<"-----------------Radar Update---------------------\n";
#endif

        UpdateRadar(meas_package);
    }
    if (MeasurementPackage::LASER == meas_package.sensor_type_)
    {
#ifdef DEBUG_OUTPUT
        std::cerr<<"-----------------Lidar Update---------------------\n";
#endif

       UpdateLidar(meas_package);
    }

#ifdef DEBUG_OUTPUT
    std::cerr<<"x:\n"<<x_<<std::endl;
    std::cerr<<"P:\n"<<P_<<std::endl;
#endif

    //update timestamp;
    time_us_ = meas_package.timestamp_;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */

/**@brief predict state and covariance;
 *
 * predict state and covariance:
 * 1. generate augmented sigma points;
 * 2. predict sigma points;
 * 3. calculate mean of predicted sigma points and covariance;
 * @param delta_t [IN]: the change in time (in seconds) between the last measurement and this one;
 */
void UKF::Prediction(const double delta_t)
{
    //generate augmented sigma points;
    MatrixXd X_aug_sig;
    GenerateAugmentedSigmaPoints(&X_aug_sig);


#ifdef DEBUG_OUTPUT
    std::cerr<<"---------------Predicted:-----------------"<<std::endl;
    std::cerr<<"x_aug_sig:"<<std::endl;
    std::cerr<<X_aug_sig<<std::endl;
#endif

    //predict sigma points;
    PredictSigmaPoints(X_aug_sig, delta_t, &Xsig_pred_);

#ifdef DEBUG_OUTPUT
    std::cerr<<"x_sig_pred:"<<std::endl;
    std::cerr<<Xsig_pred_<<std::endl;
#endif

    //predict mean and covariance;
    PredictMeanAndCovariance();

#ifdef DEBUG_OUTPUT
    std::cerr<<"x:\n"<<x_<<std::endl;
    std::cerr<<"P:\n"<<P_<<std::endl;
#endif
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */

/**@brief update the state and the covariance by a laser measurement;
 *
 * update the state and the covariance by a laser measurement:
 * 1. predict measurement;
 * 2. calculate state and covariance by standard kalman filter;
 * 3. calculate NIS;
 * @param meas_package [IN]: the lidar measurement;
 * @note the update of lidar could also use sigma points just the same as radar;
 */
void UKF::UpdateLidar(const MeasurementPackage& meas_package)
{
    //measurement matrix;
    MatrixXd H = MatrixXd(2, 5);
    H.fill(0.0);
    H(0, 0) = 1.0;
    H(1, 1) = 1.0;

    //predict measurement;
    VectorXd z_pred = H*x_;
    VectorXd y = meas_package.raw_measurements_ - z_pred;

    MatrixXd S = H*P_*H.transpose() + R_laser_;
    MatrixXd K = P_*H.transpose()*S.inverse();
    MatrixXd I = MatrixXd::Identity(5, 5);

    //update state;
    x_ = x_ + K*y;

    //update covariance;
    P_ = (I-K*H)*P_;

    NIS_laser_ = (meas_package.raw_measurements_-z_pred).transpose()*S.inverse()*(meas_package.raw_measurements_-z_pred);
}


/**@brief update state and covariance by radar measurement;
 *
 * update state and covariance by radar measurement:
 * 1. predict radar measurement and covariance based on the augmented sigma points;
 * 2. update state and covariance by kalman filter;
 * 3. calculate nis;
 * @param meas_package [IN]: the radar measurement;
 */
void UKF::UpdateRadar(const MeasurementPackage& meas_package)
{
    VectorXd z_pred;
    MatrixXd Zsig;
    MatrixXd S_pred;

    //predict measurement and covariance;
    PredictRadarMeasurement(&Zsig, &z_pred, &S_pred);

    //update state and covariance;
    UpdateRadarState(Zsig, z_pred, S_pred, meas_package);

    NIS_radar_ = (meas_package.raw_measurements_-z_pred).transpose()*S_pred.inverse()*(meas_package.raw_measurements_-z_pred);
}

/**@brief generate augmented sigma points;
 *
 * generate augmented sigma points;
 *
 * @param Xaug_sig_out [OUT]: the augmented sigma points;
 */
void UKF::GenerateAugmentedSigmaPoints(MatrixXd *Xaug_sig_out)
{
    MatrixXd X_sig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

    x_aug_.head(n_x_) = x_;
    x_aug_(5) = 0.0;
    x_aug_(6) = 0.0;

    P_aug_.topLeftCorner(n_x_, n_x_) = P_;
    P_aug_.bottomRightCorner(2, 2) = Q_;

    MatrixXd square_root = P_aug_.llt().matrixL();

    //create augmented sigma points
    X_sig_aug.col(0) = x_aug_;

    double sqrt_lambda_n_aug = sqrt(lambda_+n_aug_);
    VectorXd sqrt_square_root;
    for (int i = 0; i < n_aug_; i++)
    {
        sqrt_square_root = sqrt_lambda_n_aug*square_root.col(i);

        X_sig_aug.col(i + 1) = x_aug_ + sqrt_square_root;
        X_sig_aug.col(i + 1 + n_aug_) = x_aug_ - sqrt_square_root;
    }

    *Xaug_sig_out = X_sig_aug;
}

/**@brief predict sigma points;
 *
 * predict sigma points;
 * @param Xaug_sig [IN]: the augmented sigma points;
 * @param delta_t [IN]: the time between last measurement and current measurement;
 * @param Xaug_pred_out [OUT]: the predicted sigma points;
 */
void UKF::PredictSigmaPoints(const MatrixXd &Xaug_sig, const double delta_t, MatrixXd *Xsig_pred_out)
{
    MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

    for (int p = 0; p < 2 * n_aug_ + 1; ++p)
    {
        double p_x = Xaug_sig(0, p);
        double p_y = Xaug_sig(1, p);
        double v = Xaug_sig(2, p);
        double yaw = Xaug_sig(3, p);
        double yawd = Xaug_sig(4, p);
        double nu_a = Xaug_sig(5, p);
        double nu_yawdd = Xaug_sig(6, p);

        double pred_p_x, pred_p_y, pred_v, pred_yaw, pred_yawd;

        //predict state
        if (fabs(yawd) > 1e-3)
        {
            pred_p_x = p_x + v * (sin(yaw + yawd * delta_t) - sin(yaw)) / yawd;
            pred_p_y = p_y + v * (-cos(yaw + yawd * delta_t) + cos(yaw)) / yawd;
        }
        else
        {
            pred_p_x = p_x + v * cos(yaw) * delta_t;
            pred_p_y = p_y + v * sin(yaw) * delta_t;
        }
        pred_v = v;
        pred_yaw = yaw + yawd * delta_t;
        pred_yawd = yawd;

        //add noise;
        pred_p_x += 0.5 * delta_t * delta_t * cos(yaw) * nu_a;
        pred_p_y += 0.5 * delta_t * delta_t * sin(yaw) * nu_a;
        pred_v += delta_t * nu_a;
        pred_yaw += 0.5 * delta_t * delta_t * nu_yawdd;
        pred_yawd += delta_t * nu_yawdd;

        Xsig_pred(0, p) = pred_p_x;
        Xsig_pred(1, p) = pred_p_y;
        Xsig_pred(2, p) = pred_v;
        Xsig_pred(3, p) = pred_yaw;
        Xsig_pred(4, p) = pred_yawd;
    }

    *Xsig_pred_out = Xsig_pred;
}

/**@brief calculate mean and covariance of predicted sigma points;
 *
 */
void UKF::PredictMeanAndCovariance()
{
    //predict state mean;
    x_.fill(0.0);
    for (int i = 0; i < 2*n_aug_+1; ++i)
    {
        x_ = x_ + weights_(i)*Xsig_pred_.col(i);
    }

    //predict state covariance matrix;
    P_.fill(0.0);
    for (int i = 0; i < 2*n_aug_+1; ++i)
    {
        //state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        NormalizeAngle(x_diff(3));

        P_ = P_ + weights_(i)*x_diff*x_diff.transpose();
    }
}

/**@brief predict radar measurement;
 *
 * predict radar measurement based on the sigma points, calculate the mean and covariance of predicted sigma z;
 * @param Zsig_out [OUT]: predicted sigma points of z;
 * @param z_pred_out [OUT]: mean of predicted sigma points of z;
 * @param S_out [OUT]: covariance of predicted sigma points of z;
 */
void UKF::PredictRadarMeasurement(MatrixXd* Zsig_out, VectorXd* z_pred_out, MatrixXd* S_out)
{
    //predict z based on sigma points;
    MatrixXd Zsig = MatrixXd(3, 2*n_aug_+1);
    for (int i = 0; i < 2*n_aug_+1; ++i)
    {
        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);
        double v = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);

        double v1 = cos(yaw)*v;
        double v2 = sin(yaw)*v;

        //measurement model;
        Zsig(0, i) = sqrt(p_x*p_x+p_y*p_y);//the positon could not be zero;
        Zsig(1, i) = atan2(p_y, p_x);
        Zsig(2, i) = (p_x*v1+p_y*v2)/Zsig(0, i);
    }

    //mean predicted measurement;
    VectorXd z_pred = VectorXd(3);
    z_pred.fill(0.0);
    for (int i = 0; i < 2*n_aug_+1; ++i)
    {
        z_pred = z_pred + weights_(i)*Zsig.col(i);
    }

    //measurement covariance matrix;
    MatrixXd S = MatrixXd(3, 3);
    S.fill(0.0);
    for (int i = 0; i < 2*n_aug_+1; ++i)
    {
        VectorXd z_diff = Zsig.col(i) - z_pred;
        NormalizeAngle(z_diff(1));
        S = S + weights_(i)*z_diff*z_diff.transpose();
    }
    S = S+R_radar_;

    *z_pred_out = z_pred;
    *S_out = S;
    *Zsig_out = Zsig;
}


/**@brief update state and covariance based on radar measurement;
 *
 * update state and covariance based on radar measurement;
 * @param Zsig [IN]: the sigma z points;
 * @param z_pred [IN]: the mean of sigma z points;
 * @param S [IN]: the covariance of sigma z points;
 * @param meas_package [IN]: the radar measurement;
 */
void UKF::UpdateRadarState(const MatrixXd& Zsig, const VectorXd& z_pred, const MatrixXd& S, const MeasurementPackage& meas_package)
{
    //calculate T;
    MatrixXd Tc = MatrixXd(n_x_, 3);
    Tc.fill(0.0);
    for (int i = 0; i < 2*n_aug_+1; ++i)
    {
        VectorXd z_diff = Zsig.col(i) - z_pred;

        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        NormalizeAngle(x_diff(3));

        Tc = Tc + weights_(i)*x_diff*z_diff.transpose();
    }

    //kalman gain K;
    MatrixXd K = Tc*S.inverse();

    //residual z;
    VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
    NormalizeAngle(z_diff(1));

    //calculate state and covariance;
    x_ = x_ + K*z_diff;
    P_ = P_ - K*S*K.transpose();
}
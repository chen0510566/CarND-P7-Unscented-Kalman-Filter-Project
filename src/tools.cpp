#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

//#define DEBUG_OUTPUT

Tools::Tools()
{
}

Tools::~Tools()
{
}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth)
{
    VectorXd rmse(4);
    rmse.fill(0.0);

    if (estimations.size() == 0 || estimations.size() != ground_truth.size())
    {
        std::cerr << "invalid number of estimations and ground truth" << std::endl;
        return rmse;
    }

    for (int i = 0; i < estimations.size(); ++i)
    {
        VectorXd residual = estimations[i] - ground_truth[i];
        rmse.array() += residual.array() * residual.array();
#ifdef DEBUG_OUTPUT
        std::cerr<<"---------------------"<<i<<"------------------------"<<std::endl;
        std::cerr<<"estimation:\n"<<estimations[i]<<std::endl;
        std::cerr<<"ground_truth:\n"<<ground_truth[i]<<std::endl;
        std::cerr<<"residual:\n"<<residual<<std::endl;
        std::cerr<<"estimated velocity:"<<sqrt(estimations[i](2)*estimations[i](2)+estimations[i](3)*estimations[i](3))<<std::endl;
        std::cerr<<"true velocity:"<<sqrt(ground_truth[i](2)*ground_truth[i](2)+ground_truth[i](3)*ground_truth[i](3))<<std::endl;
#endif

    }
    rmse /= estimations.size();
    rmse = rmse.array().sqrt();

    return rmse;
}

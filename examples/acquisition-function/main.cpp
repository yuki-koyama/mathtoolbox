#include <Eigen/Core>
#include <iostream>
#include <mathtoolbox/gaussian-process-regression.hpp>
#include <mathtoolbox/acquisition-functions.hpp>
#include <mathtoolbox/probability-distributions.hpp>
#include <timer.hpp>

using Eigen::MatrixXd;
using Eigen::Matrix2d;
using Eigen::VectorXd;
using Eigen::Vector2d;

double CalcFunc(const Vector2d& x)
{
    return mathtoolbox::GetNormalDist(3.0 * x, Vector2d::Zero(), Matrix2d::Identity(), 1.0);
}

int main(int argc, char** argv)
{
    // Define the scene setting
    constexpr int    number_of_samples = 50;
    constexpr double noise_intensity = 1e-04;

    // Generate (and export) scattered data
    MatrixXd X(2, number_of_samples);
    VectorXd y(number_of_samples);
    for (int i = 0; i < number_of_samples; ++i)
    {
        X.col(i) = Vector2d::Random();
        y(i)    = CalcFunc(X.col(i)) + noise_intensity * (VectorXd::Random(1))(0);
    }

    // Define the kernel type
    const auto kernel_type = mathtoolbox::GaussianProcessRegressor::KernelType::ArdMatern52;

    // Instantiate the interpolation object
    mathtoolbox::GaussianProcessRegressor regressor(X, y, kernel_type);

    // Perform hyperparameter estimation
    const Eigen::Vector3d default_kernel_hyperparams{0.50, 0.50, 0.50};
    {
        timer::Timer t("maximum likelihood estimation");
        regressor.PerformMaximumLikelihood(default_kernel_hyperparams, 1e-04);
    }

    return 0;
}

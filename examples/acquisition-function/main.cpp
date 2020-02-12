#include <Eigen/Core>
#include <iostream>
#include <mathtoolbox/acquisition-functions.hpp>
#include <mathtoolbox/gaussian-process-regression.hpp>
#include <mathtoolbox/probability-distributions.hpp>
#include <timer.hpp>

using Eigen::Matrix2d;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::VectorXd;

double CalcFunc(const Vector2d& x)
{
    return mathtoolbox::GetNormalDist(3.0 * x, Vector2d::Zero(), Matrix2d::Identity(), 1.0);
}

int main(int argc, char** argv)
{
    // Define the scene setting
    constexpr int    num_samples     = 10;
    constexpr double noise_intensity = 1e-04;

    // Generate scattered data
    MatrixXd X(2, num_samples);
    VectorXd y(num_samples);
    for (int i = 0; i < num_samples; ++i)
    {
        X.col(i) = Vector2d::Random();
        y(i)     = CalcFunc(X.col(i)) + noise_intensity * (VectorXd::Random(1))(0);
    }

    // Define the kernel type
    const auto kernel_type = mathtoolbox::GaussianProcessRegressor::KernelType::ArdMatern52;

    // Instantiate the interpolation object
    mathtoolbox::GaussianProcessRegressor regressor(X, y, kernel_type);

    // Perform hyperparameter estimation
    const Eigen::Vector3d default_kernel_hyperparams(0.50, 0.50, 0.50);
    regressor.PerformMaximumLikelihood(default_kernel_hyperparams, 1e-04);

    // Calculate EI values and their derivatives
    for (int i = 0; i < 100; ++i)
    {
        constexpr int    num_dims = 2;
        constexpr double epsilon  = 1e-06;

        const VectorXd x_plus = [&]() {
            int index;
            y.maxCoeff(&index);
            return X.col(index);
        }();

        const VectorXd x = Vector2d::Random();

        const VectorXd acquisition_deriv = mathtoolbox::GetExpectedImprovementDerivative(
            x,
            [&](const VectorXd& x) { return regressor.PredictMean(x); },
            [&](const VectorXd& x) { return regressor.PredictStdev(x); },
            x_plus,
            [&](const VectorXd& x) { return regressor.PredictMeanDeriv(x); },
            [&](const VectorXd& x) { return regressor.PredictStdevDeriv(x); });

        VectorXd acquisition_numerical_deriv(num_dims);
        for (int d = 0; d < num_dims; ++d)
        {
            VectorXd delta = VectorXd::Zero(num_dims);
            delta(d)       = epsilon;

            const double value_plus = mathtoolbox::GetExpectedImprovement(
                x + delta,
                [&](const VectorXd& x) { return regressor.PredictMean(x); },
                [&](const VectorXd& x) { return regressor.PredictStdev(x); },
                x_plus);

            const double value_minus = mathtoolbox::GetExpectedImprovement(
                x - delta,
                [&](const VectorXd& x) { return regressor.PredictMean(x); },
                [&](const VectorXd& x) { return regressor.PredictStdev(x); },
                x_plus);

            acquisition_numerical_deriv(d) = value_plus - value_minus;
        }
        acquisition_numerical_deriv /= 2.0 * epsilon;

        const auto scale     = acquisition_deriv.norm();
        const auto abs_error = (acquisition_deriv - acquisition_numerical_deriv).norm();
        const auto rel_error = abs_error / scale;

        if (scale > 1e-06 && rel_error > 1e-02)
        {
            std::cout << "point location: " << x.transpose() << std::endl;
            std::cout << "analytic      : " << acquisition_deriv.transpose() << std::endl;
            std::cout << "numerical     : " << acquisition_numerical_deriv.transpose() << std::endl;
            std::cout << "error         : " << abs_error << std::endl;

            exit(1);
        }
    }

    // Calculate GP-UCB values and their derivatives
    for (int i = 0; i < 100; ++i)
    {
        constexpr int    num_dims = 2;
        constexpr double epsilon  = 1e-06;

        const double hyperparam = 1.0 + (Eigen::VectorXd::Random(1))(0);

        const VectorXd x = Vector2d::Random();

        const VectorXd acquisition_deriv = mathtoolbox::GetGaussianProcessUpperConfidenceBoundDerivative(
            x,
            [&](const VectorXd& x) { return regressor.PredictMean(x); },
            [&](const VectorXd& x) { return regressor.PredictStdev(x); },
            hyperparam,
            [&](const VectorXd& x) { return regressor.PredictMeanDeriv(x); },
            [&](const VectorXd& x) { return regressor.PredictStdevDeriv(x); });

        VectorXd acquisition_numerical_deriv(num_dims);
        for (int d = 0; d < num_dims; ++d)
        {
            VectorXd delta = VectorXd::Zero(num_dims);
            delta(d)       = epsilon;

            const double value_plus = mathtoolbox::GetGaussianProcessUpperConfidenceBound(
                x + delta,
                [&](const VectorXd& x) { return regressor.PredictMean(x); },
                [&](const VectorXd& x) { return regressor.PredictStdev(x); },
                hyperparam);

            const double value_minus = mathtoolbox::GetGaussianProcessUpperConfidenceBound(
                x - delta,
                [&](const VectorXd& x) { return regressor.PredictMean(x); },
                [&](const VectorXd& x) { return regressor.PredictStdev(x); },
                hyperparam);

            acquisition_numerical_deriv(d) = value_plus - value_minus;
        }
        acquisition_numerical_deriv /= 2.0 * epsilon;

        const auto scale     = acquisition_deriv.norm();
        const auto abs_error = (acquisition_deriv - acquisition_numerical_deriv).norm();
        const auto rel_error = abs_error / scale;

        if (scale > 1e-06 && rel_error > 1e-02)
        {
            std::cout << "point location: " << x.transpose() << std::endl;
            std::cout << "analytic      : " << acquisition_deriv.transpose() << std::endl;
            std::cout << "numerical     : " << acquisition_numerical_deriv.transpose() << std::endl;
            std::cout << "error         : " << abs_error << std::endl;

            exit(1);
        }
    }

    return 0;
}

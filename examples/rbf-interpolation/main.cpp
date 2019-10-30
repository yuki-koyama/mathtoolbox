#include <Eigen/Core>
#include <iostream>
#include <mathtoolbox/rbf-interpolation.hpp>
#include <random>
#include <vector>

using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::VectorXd;

namespace
{
    std::random_device                     seed;
    std::default_random_engine             engine(seed());
    std::uniform_real_distribution<double> uniform_dist(-1.0, 1.0);

    double CalculateFunction(const Vector2d& x) { return std::sin(10.0 * x(0)) + std::sin(10.0 * x(1)); }
} // namespace

int main()
{
    // Generate scattered data (in this case, 500 data points in a 2-dimensional space)
    constexpr int    number_of_samples = 500;
    constexpr double noise_intensity   = 0.1;
    Eigen::MatrixXd  X(2, number_of_samples);
    Eigen::VectorXd  y(number_of_samples);
    for (int i = 0; i < number_of_samples; ++i)
    {
        X.col(i) = Vector2d(uniform_dist(engine), uniform_dist(engine));
        y(i)     = CalculateFunction(X.col(i)) + noise_intensity * uniform_dist(engine);
    }

    // Instantiate an interpolator
    constexpr mathtoolbox::RbfType rbf_type = mathtoolbox::RbfType::ThinPlateSpline;
    mathtoolbox::RbfInterpolation  rbf_interpolation(rbf_type);

    // Set data
    rbf_interpolation.SetData(X, y);

    // Compute internal weights with or without regularization
    constexpr bool use_regularization = true;
    rbf_interpolation.ComputeWeights(use_regularization);

    // Calculate and print interpolated values on randomly sampled points in CSV format
    constexpr int number_of_test_samples = 100;
    std::cout << "x(0),x(1),y" << std::endl;
    for (int i = 0; i < number_of_test_samples; ++i)
    {
        const Vector2d x = Vector2d(uniform_dist(engine), uniform_dist(engine));
        const double   y = rbf_interpolation.GetValue(x);

        std::cout << x(0) << "," << x(1) << "," << y << std::endl;
    }

    return 0;
}

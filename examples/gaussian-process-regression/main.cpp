#include <vector>
#include <random>
#include <iostream>
#include <Eigen/Core>
#include <mathtoolbox/gaussian-process-regression.hpp>

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace
{
    std::random_device seed;
    std::default_random_engine engine(seed());
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    
    double CalculateFunction(double x)
    {
        return x * std::sin(10.0 * x);
    }
}

int main()
{
    // Generate scattered data
    constexpr int    number_of_samples = 10;
    constexpr double noise_intensity   = 0.10;
    Eigen::MatrixXd X(1, number_of_samples);
    Eigen::VectorXd y(number_of_samples);
    for (int i = 0; i < number_of_samples; ++ i)
    {
        X(0, i) = uniform_dist(engine);
        y(i)    = CalculateFunction(X(0, i)) + noise_intensity * uniform_dist(engine);
    }
    
    // Instantiate the interpolation object
    mathtoolbox::GaussianProcessRegression regressor(X, y);
    
    // Calculate and print estimated values
    std::cout << "x,y,s" << std::endl;
    for (int i = 0; i <= 100; ++ i)
    {
        const double x = (1.0 / 100.0) * i;
        const double y = regressor.EstimateY(Eigen::VectorXd::Constant(1, x));
        const double s = regressor.EstimateS(Eigen::VectorXd::Constant(1, x));

        std::cout << x << "," << y << "," << s << std::endl;
    }
    
    return 0;
}

#include <vector>
#include <random>
#include <iostream>
#include <Eigen/Core>
#include <mathtoolbox/gaussian-process-regression.hpp>

using Eigen::VectorXd;
using Eigen::Vector2d;
using Eigen::MatrixXd;

namespace
{
    std::random_device seed;
    std::default_random_engine engine(seed());
    std::uniform_real_distribution<double> uniform_dist(-1.0, 1.0);
    
    double CalculateFunction(const Vector2d& x)
    {
        return std::sin(10.0 * x(0)) + std::sin(10.0 * x(1));
    }
}

int main()
{
    // Generate scattered data (in this case, 500 data points in a 2-dimensional space)
    constexpr int    number_of_samples = 500;
    constexpr double noise_intensity   = 0.1;
    Eigen::MatrixXd X(2, number_of_samples);
    Eigen::VectorXd y(number_of_samples);
    for (int i = 0; i < number_of_samples; ++ i)
    {
        X.col(i) = Vector2d(uniform_dist(engine), uniform_dist(engine));
        y(i)     = CalculateFunction(X.col(i)) + noise_intensity * uniform_dist(engine);
    }
    
    // Instantiate the interpolation object
    mathtoolbox::GaussianProcessRegression regressor(X, y);
    
    // Calculate and print interpolated values on randomly sampled points
    std::cout << "x(0),x(1),y,s" << std::endl;
    for (int i = 0; i < 100; ++ i)
    {
        const Vector2d x = Vector2d(uniform_dist(engine), uniform_dist(engine));
        const double   y = regressor.EstimateY(x);
        const double   s = regressor.EstimateS(x);

        std::cout << x(0) << "," << x(1) << "," << y << "," << s << std::endl;
    }
    
    return 0;
}

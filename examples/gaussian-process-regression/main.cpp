#include <vector>
#include <random>
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <Eigen/Core>
#include <mathtoolbox/gaussian-process-regression.hpp>

using Eigen::VectorXd;
using Eigen::MatrixXd;

namespace
{
    std::random_device seed;
    std::default_random_engine engine(seed());
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    double CalculateFunction(double x)
    {
        return x * std::sin(10.0 * x);
    }
}

int main(int argc, char** argv)
{
    // Set a output directory path
    const std::string output_directory_path = (argc < 2) ? "." : argv[1];

    // Generate (and export) scattered data
    constexpr int    number_of_samples = 50;
    constexpr double noise_intensity   = 0.010;
    std::ofstream scattered_data_stream(output_directory_path + "/scattered_data.csv");
    scattered_data_stream << "x,y" << std::endl;
    Eigen::MatrixXd X(1, number_of_samples);
    Eigen::VectorXd y(number_of_samples);
    for (int i = 0; i < number_of_samples; ++ i)
    {
        X(0, i) = uniform_dist(engine);
        y(i)    = CalculateFunction(X(0, i)) + noise_intensity * normal_dist(engine);

        scattered_data_stream << X(0, i) << "," << y(i) << std::endl;
    }
    scattered_data_stream.close();

    // Instantiate the interpolation object
    mathtoolbox::GaussianProcessRegression regressor(X, y);
    regressor.PerformMaximumLikelihood(0.50, 0.010, Eigen::VectorXd::Constant(1, 0.50));

    // Define constants for export
    constexpr int    resolution       = 300;
    constexpr double percentile_point = 1.95996398454005423552;

    // Calculate (and export) predictive distribution
    std::ofstream estimated_data_stream(output_directory_path + "/estimated_data.csv");
    estimated_data_stream << "x,mean,standard deviation,95-percent upper,95-percent lower" << std::endl;
    for (int i = 0; i <= resolution; ++ i)
    {
        const double x = (1.0 / static_cast<double>(resolution)) * i;
        const double y = regressor.EstimateY(Eigen::VectorXd::Constant(1, x));
        const double s = std::sqrt(regressor.EstimateVariance(Eigen::VectorXd::Constant(1, x)));

        estimated_data_stream << x << "," << y << "," << s << "," << y + percentile_point * s << "," << y - percentile_point * s << std::endl;
    }
    estimated_data_stream.close();

    return 0;
}

#include <cmath>
#include <functional>
#include <mathtoolbox/probability-distributions.hpp>
#include <random>

double CalculateNumericalDifferentiation(const std::function<double(double)>& f, const double x, const double h = 1e-04)
{
    return (-f(x + 2.0 * h) + 8.0 * f(x + h) - 8.0 * f(x - h) + f(x - 2.0 * h)) / (12.0 * h);
}

bool CheckConsistency(const std::function<double(double)>& f, const std::function<double(double)>& g, const double x)
{
    const double dfdx_numerical = CalculateNumericalDifferentiation(f, x);
    const double dfdx_analytic  = g(x);

    const double scale = std::max(std::abs(dfdx_numerical), std::abs(dfdx_analytic));
    const double diff  = std::abs(dfdx_numerical - dfdx_analytic);

    constexpr double relative_threshold = 1e-02;
    constexpr double absolute_threshold = 1e-08;

    const bool relative_check = diff <= relative_threshold * scale;
    const bool absolute_check = diff <= absolute_threshold;

    return relative_check || absolute_check;
}

int main(int argc, char** argv)
{
    constexpr int num_tests = 100;

    std::random_device                     seed;
    std::default_random_engine             engine(seed());
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

    // NormalDist
    for (int i = 0; i < num_tests; ++i)
    {
        const double mu      = 20.0 * uniform_dist(engine) - 10.0;
        const double sigma_2 = 10.0 * uniform_dist(engine) + 1e-16;

        const auto f = [&mu, &sigma_2](const double x) { return mathtoolbox::GetNormalDist(x, mu, sigma_2); };
        const auto g = [&mu, &sigma_2](const double x) { return mathtoolbox::GetNormalDistDerivative(x, mu, sigma_2); };

        const double x = 10.0 * uniform_dist(engine) - 5.0;

        if (!CheckConsistency(f, g, x))
        {
            return 1;
        }
    }

    // LogNormalDist
    for (int i = 0; i < num_tests; ++i)
    {
        const double mu      = 2.0 * uniform_dist(engine) - 1.0;
        const double sigma_2 = 2.0 * uniform_dist(engine) + 1e-16;

        const auto f = [&mu, &sigma_2](const double x) { return mathtoolbox::GetLogNormalDist(x, mu, sigma_2); };
        const auto g = [&mu, &sigma_2](const double x) {
            return mathtoolbox::GetLogNormalDistDerivative(x, mu, sigma_2);
        };

        const double x = 10.0 * uniform_dist(engine) + 1e-03;

        if (!CheckConsistency(f, g, x))
        {
            return 1;
        }
    }

    // Log of LogNormalDist
    for (int i = 0; i < num_tests; ++i)
    {
        const double mu      = 2.0 * uniform_dist(engine) - 1.0;
        const double sigma_2 = 2.0 * uniform_dist(engine) + 1e-16;

        const auto f = [&mu, &sigma_2](const double x) { return mathtoolbox::GetLogOfLogNormalDist(x, mu, sigma_2); };
        const auto g = [&mu, &sigma_2](const double x) {
            return mathtoolbox::GetLogOfLogNormalDistDerivative(x, mu, sigma_2);
        };

        const double x = 10.0 * uniform_dist(engine) + 1e-03;

        if (!CheckConsistency(f, g, x))
        {
            return 1;
        }
    }

    return 0;
}

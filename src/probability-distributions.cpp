#include <cmath>
#include <mathtoolbox/constants.hpp>
#include <mathtoolbox/probability-distributions.hpp>

double mathtoolbox::GetNormalDist(const double x, const double mu, const double sigma_2)
{
    const double r = x - mu;
    return (1.0 / std::sqrt(2.0 * constants::pi * sigma_2)) * std::exp(-0.5 * (r * r) / sigma_2);
}

double mathtoolbox::GetNormalDistDerivative(const double x, const double mu, const double sigma_2)
{
    const double r = x - mu;
    const double N = (1.0 / std::sqrt(2.0 * constants::pi * sigma_2)) * std::exp(-0.5 * (r * r) / sigma_2);
    return -r * N / sigma_2;
}

double mathtoolbox::GetLogNormalDist(const double x, const double mu, const double sigma_2)
{
    const double r = std::log(x) - mu;
    return (1.0 / (x * std::sqrt(2.0 * constants::pi * sigma_2))) * std::exp(-0.5 * (r * r) / sigma_2);
}

double mathtoolbox::GetLogNormalDistDerivative(const double x, const double mu, const double sigma_2)
{
    const double LN = GetLogNormalDist(x, mu, sigma_2);
    return LN * (mu - std::log(x) - sigma_2) / (x * sigma_2);
}

double mathtoolbox::GetLogOfLogNormalDist(const double x, const double mu, const double sigma_2)
{
    const double log_x = std::log(x);
    const double r     = log_x - mu;

    return - log_x - 0.5 * std::log(2.0 * constants::pi * sigma_2) - 0.5 * (r * r) / sigma_2;
}

double mathtoolbox::GetLogOfLogNormalDistDerivative(const double x, const double mu, const double sigma_2)
{
    return (mu - std::log(x) - sigma_2) / (x * sigma_2);
}

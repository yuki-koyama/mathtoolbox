#include <cmath>
#include <mathtoolbox/constants.hpp>
#include <mathtoolbox/distributions.hpp>

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

#include <cmath>
#include <mathtoolbox/constants.hpp>
#include <mathtoolbox/probability-distributions.hpp>

double mathtoolbox::GetStandardNormalDist(const double x)
{
    return (1.0 / std::sqrt(2.0 * constants::pi)) * std::exp(-0.5 * x * x);
}

double mathtoolbox::GetStandardNormalDistDerivative(const double x)
{
    const double N = GetStandardNormalDist(x);
    return -x * N;
}

double mathtoolbox::GetStandardNormalDistCdf(const double x)
{
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

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

    return -log_x - 0.5 * std::log(2.0 * constants::pi * sigma_2) - 0.5 * (r * r) / sigma_2;
}

double mathtoolbox::GetLogOfLogNormalDistDerivative(const double x, const double mu, const double sigma_2)
{
    return (mu - std::log(x) - sigma_2) / (x * sigma_2);
}

double mathtoolbox::GetNormalDist(const Eigen::VectorXd& x,
                                  const Eigen::VectorXd& mu,
                                  const Eigen::MatrixXd& Sigma_inv,
                                  const double           Sigma_det)
{
    const double          coeff = std::pow(2.0 * constants::pi, -0.5 * mu.size()) * std::pow(Sigma_det, -0.5);
    const Eigen::VectorXd r     = x - mu;

    return coeff * std::exp(-0.5 * (r.transpose() * Sigma_inv * r)(0, 0));
}

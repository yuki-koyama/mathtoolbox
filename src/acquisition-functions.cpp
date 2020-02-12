#include <cmath>
#include <mathtoolbox/acquisition-functions.hpp>
#include <mathtoolbox/probability-distributions.hpp>

using Eigen::VectorXd;

double mathtoolbox::GetExpectedImprovement(const VectorXd&                               x,
                                           const std::function<double(const VectorXd&)>& mu,
                                           const std::function<double(const VectorXd&)>& sigma,
                                           const VectorXd&                               x_best)
{
    const double sigma_x = sigma(x);
    const double f_best  = mu(x_best);
    const double diff    = mu(x) - f_best;
    const double Z       = diff / sigma_x;
    const double Phi     = GetStandardNormalDistCdf(Z);
    const double phi     = GetStandardNormalDist(Z);
    const double EI      = diff * Phi + sigma_x * phi;

    constexpr double epsilon = 1e-16;

    return (sigma_x < epsilon || std::isnan(EI)) ? 0.0 : EI;
}

VectorXd mathtoolbox::GetExpectedImprovementDerivative(const VectorXd&                                 x,
                                                       const std::function<double(const VectorXd&)>&   mu,
                                                       const std::function<double(const VectorXd&)>&   sigma,
                                                       const VectorXd&                                 x_best,
                                                       const std::function<VectorXd(const VectorXd&)>& mu_derivative,
                                                       const std::function<VectorXd(const VectorXd&)>& sigma_derivative)
{
    const VectorXd mu_x_derivative    = mu_derivative(x);
    const VectorXd sigma_x_derivative = sigma_derivative(x);

    const double f_best  = mu(x_best);
    const double sigma_x = sigma(x);
    const double mu_x    = mu(x);
    const double diff    = mu_x - f_best;

    const double Z = diff / sigma_x;

    const double Phi              = GetStandardNormalDistCdf(Z);
    const double phi              = GetStandardNormalDist(Z);
    const double phi_Z_derivative = GetStandardNormalDistDerivative(Z);

    const VectorXd Z_x_derivative = (mu_x_derivative - Z * sigma_x_derivative) / sigma_x;

    const VectorXd EI_derivative = mu_x_derivative * Phi + diff * Z_x_derivative * phi + sigma_x_derivative * phi +
                                   sigma_x * Z_x_derivative * phi_Z_derivative;

    constexpr double epsilon = 1e-16;

    return (sigma_x < epsilon || EI_derivative.hasNaN()) ? VectorXd::Zero(x.size()) : EI_derivative;
}

double mathtoolbox::GetGaussianProcessUpperConfidenceBound(const VectorXd&                               x,
                                                           const std::function<double(const VectorXd&)>& mu,
                                                           const std::function<double(const VectorXd&)>& sigma,
                                                           const double                                  hyperparam)
{
    assert(hyperparam >= 0);

    return mu(x) + hyperparam * sigma(x);
}

VectorXd mathtoolbox::GetGaussianProcessUpperConfidenceBoundDerivative(
    const VectorXd&                                 x,
    const std::function<double(const VectorXd&)>&   mu,
    const std::function<double(const VectorXd&)>&   sigma,
    const double                                    hyperparam,
    const std::function<VectorXd(const VectorXd&)>& mu_derivative,
    const std::function<VectorXd(const VectorXd&)>& sigma_derivative)
{
    assert(hyperparam >= 0);

    return mu_derivative(x) + hyperparam * sigma_derivative(x);
}

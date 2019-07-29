#include <cmath>
#include <mathtoolbox/acquisition-functions.hpp>
#include <mathtoolbox/probability-distributions.hpp>

using Eigen::VectorXd;

double mathtoolbox::GetExpectedImprovement(const VectorXd&                               x,
                                           const std::function<double(const VectorXd&)>& mu,
                                           const std::function<double(const VectorXd&)>& sigma,
                                           const VectorXd&                               x_best)
{
    const double y_best = mu(x_best);
    const double s_x    = sigma(x);
    const double diff   = mu(x) - y_best;
    const double u      = diff / s_x;
    const double Phi    = GetStandardNormalDistCdf(u);
    const double phi    = GetStandardNormalDist(u);
    const double EI     = diff * Phi + s_x * phi;

    constexpr double epsilon = 1e-16;

    return (s_x < epsilon || std::isnan(EI)) ? 0.0 : EI;
}

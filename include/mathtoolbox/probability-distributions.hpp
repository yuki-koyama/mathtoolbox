#ifndef MATHTOOLBOX_PROBABILITY_DISTRIBUTIONS_HPP
#define MATHTOOLBOX_PROBABILITY_DISTRIBUTIONS_HPP

#include <Eigen/Core>

namespace mathtoolbox
{
    // N(x | 0, 1)
    double GetStandardNormalDist(const double x);

    // d/dx N(x | 0, 1)
    double GetStandardNormalDistDerivative(const double x);

    // integral_{- inf, x} N(x' | 0, 1) dx'
    double GetStandardNormalDistCdf(const double x);

    // N(x | mu, sigma^2)
    double GetNormalDist(const double x, const double mu, const double sigma_2);

    // d/dx N(x | mu, sigma^2)
    double GetNormalDistDerivative(const double x, const double mu, const double sigma_2);

    // LogNormal(x | mu, sigma^2)
    double GetLogNormalDist(const double x, const double mu, const double sigma_2);

    // d/dx LogNormal(x | mu, sigma^2)
    double GetLogNormalDistDerivative(const double x, const double mu, const double sigma_2);

    // log{ LogNormal(x | mu, sigma^2) }
    double GetLogOfLogNormalDist(const double x, const double mu, const double sigma_2);

    // d/dx log{ LogNormal(x | mu, sigma^2) }
    double GetLogOfLogNormalDistDerivative(const double x, const double mu, const double sigma_2);

    // N(x | mu, Sigma)
    double GetNormalDist(const Eigen::VectorXd& x,
                         const Eigen::VectorXd& mu,
                         const Eigen::MatrixXd& Sigma_inv,
                         const double           Sigma_det);
} // namespace mathtoolbox

#endif // MATHTOOLBOX_PROBABILITY_DISTRIBUTIONS_HPP

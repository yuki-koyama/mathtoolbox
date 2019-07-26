#ifndef DISTRIBUTIONS_HPP
#define DISTRIBUTIONS_HPP

namespace mathtoolbox
{
    // N(x | mu, sigma^2)
    double GetNormalDist(const double x, const double mu, const double sigma_2);

    // d/dx N(x | mu, sigma^2)
    double GetNormalDistDerivative(const double x, const double mu, const double sigma_2);

    // log{ LogNormal(x | mu, sigma^2) }
    double GetLogOfLogNormalDist(const double x, const double mu, const double sigma_2);

    // d/dx log{ LogNormal(x | mu, sigma^2) }
    double GetLogOfLogNormalDistDerivative(const double x, const double mu, const double sigma_2);
} // namespace mathtoolbox

#endif /* DISTRIBUTIONS_HPP */

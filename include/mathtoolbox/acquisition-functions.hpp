#ifndef ACQUISITION_FUNCTIONS_HPP
#define ACQUISITION_FUNCTIONS_HPP

#include <Eigen/Core>
#include <functional>

namespace mathtoolbox
{
    double GetExpectedImprovement(const Eigen::VectorXd&                               x,
                                  const std::function<double(const Eigen::VectorXd&)>& mu,
                                  const std::function<double(const Eigen::VectorXd&)>& sigma,
                                  const Eigen::VectorXd&                               x_best);
} // namespace mathtoolbox

#endif /* ACQUISITION_FUNCTIONS_HPP */

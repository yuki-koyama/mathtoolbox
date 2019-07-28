#ifndef KERNEL_FUNCTIONS_HPP
#define KERNEL_FUNCTIONS_HPP

#include <Eigen/Core>

namespace mathtoolbox
{
    double GetArdSquaredExponentialKernel(const Eigen::VectorXd& x_a,
                                          const Eigen::VectorXd& x_b,
                                          const Eigen::VectorXd& hyperparameters);

    Eigen::VectorXd GetArdSquaredExponentialKernelHyperparametersDerivative(const Eigen::VectorXd& x_a,
                                                                            const Eigen::VectorXd& x_b,
                                                                            const Eigen::VectorXd& hyperparameters);
} // namespace mathtoolbox

#endif /* KERNEL_FUNCTIONS_HPP */

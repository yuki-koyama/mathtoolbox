#ifndef KERNEL_FUNCTIONS_HPP
#define KERNEL_FUNCTIONS_HPP

#include <Eigen/Core>
#include <functional>

namespace mathtoolbox
{
    using Kernel = std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&)>;
    using KernelThetaDerivative =
        std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&)>;
    using KernelThetaIDerivative =
        std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const int)>;

    double GetArdSquaredExpKernel(const Eigen::VectorXd& x_a, const Eigen::VectorXd& x_b, const Eigen::VectorXd& theta);

    Eigen::VectorXd GetArdSquaredExpKernelThetaDerivative(const Eigen::VectorXd& x_a,
                                                          const Eigen::VectorXd& x_b,
                                                          const Eigen::VectorXd& theta);

    double GetArdSquaredExpKernelThetaIDerivative(const Eigen::VectorXd& x_a,
                                                  const Eigen::VectorXd& x_b,
                                                  const Eigen::VectorXd& theta,
                                                  const int              index);

    double GetArdMatern52Kernel(const Eigen::VectorXd& x_a, const Eigen::VectorXd& x_b, const Eigen::VectorXd& theta);

    Eigen::VectorXd GetArdMatern52KernelThetaDerivative(const Eigen::VectorXd& x_a,
                                                        const Eigen::VectorXd& x_b,
                                                        const Eigen::VectorXd& theta);

    double GetArdMatern52KernelThetaIDerivative(const Eigen::VectorXd& x_a,
                                                const Eigen::VectorXd& x_b,
                                                const Eigen::VectorXd& theta,
                                                const int              index);
} // namespace mathtoolbox

#endif /* KERNEL_FUNCTIONS_HPP */

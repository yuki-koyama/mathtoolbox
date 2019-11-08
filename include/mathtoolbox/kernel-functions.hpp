#ifndef MATHTOOLBOX_KERNEL_FUNCTIONS_HPP
#define MATHTOOLBOX_KERNEL_FUNCTIONS_HPP

#include <Eigen/Core>
#include <functional>

namespace mathtoolbox
{
    using Kernel = std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&)>;
    using KernelThetaDerivative =
        std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&)>;
    using KernelThetaIDerivative =
        std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&, const int)>;
    using KernelFirstArgDerivative =
        std::function<Eigen::VectorXd(const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&)>;

    double GetArdSquaredExpKernel(const Eigen::VectorXd& x_a, const Eigen::VectorXd& x_b, const Eigen::VectorXd& theta);

    Eigen::VectorXd GetArdSquaredExpKernelThetaDerivative(const Eigen::VectorXd& x_a,
                                                          const Eigen::VectorXd& x_b,
                                                          const Eigen::VectorXd& theta);

    double GetArdSquaredExpKernelThetaIDerivative(const Eigen::VectorXd& x_a,
                                                  const Eigen::VectorXd& x_b,
                                                  const Eigen::VectorXd& theta,
                                                  const int              index);

    Eigen::VectorXd GetArdSquaredExpKernelFirstArgDerivative(const Eigen::VectorXd& x_a,
                                                             const Eigen::VectorXd& x_b,
                                                             const Eigen::VectorXd& theta);

    double GetArdMatern52Kernel(const Eigen::VectorXd& x_a, const Eigen::VectorXd& x_b, const Eigen::VectorXd& theta);

    Eigen::VectorXd GetArdMatern52KernelThetaDerivative(const Eigen::VectorXd& x_a,
                                                        const Eigen::VectorXd& x_b,
                                                        const Eigen::VectorXd& theta);

    double GetArdMatern52KernelThetaIDerivative(const Eigen::VectorXd& x_a,
                                                const Eigen::VectorXd& x_b,
                                                const Eigen::VectorXd& theta,
                                                const int              index);

    Eigen::VectorXd GetArdMatern52KernelFirstArgDerivative(const Eigen::VectorXd& x_a,
                                                           const Eigen::VectorXd& x_b,
                                                           const Eigen::VectorXd& theta);
} // namespace mathtoolbox

#endif // MATHTOOLBOX_KERNEL_FUNCTIONS_HPP

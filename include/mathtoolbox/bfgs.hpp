#ifndef MATHTOOLBOX_BFGS_HPP
#define MATHTOOLBOX_BFGS_HPP

#include <Eigen/Core>
#include <functional>

namespace mathtoolbox
{
    namespace optimization
    {
        void RunBfgs(const Eigen::VectorXd&                                        x_init,
                     const std::function<double(const Eigen::VectorXd&)>&          f,
                     const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& g,
                     const double                                                  epsilon,
                     const unsigned int                                            max_num_iterations,
                     Eigen::VectorXd&                                              x_star,
                     unsigned int&                                                 num_iterations);
    }
} // namespace mathtoolbox

#endif // MATHTOOLBOX_BFGS_HPP

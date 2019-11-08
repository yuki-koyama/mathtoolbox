#ifndef MATHTOOLBOX_GRADIENT_DESCENT_HPP
#define MATHTOOLBOX_GRADIENT_DESCENT_HPP

#include <Eigen/Core>

namespace mathtoolbox
{
    namespace optimization
    {
        void RunGradientDescent(const Eigen::VectorXd&                                        x_init,
                                const std::function<double(const Eigen::VectorXd&)>&          f,
                                const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& g,
                                const Eigen::VectorXd&                                        lower_bound,
                                const Eigen::VectorXd&                                        upper_bound,
                                const double                                                  epsilon,
                                const unsigned int                                            max_num_iterations,
                                Eigen::VectorXd&                                              x_star,
                                unsigned int&                                                 num_iterations);
    }
} // namespace mathtoolbox

#endif // MATHTOOLBOX_GRADIENT_DESCENT_HPP

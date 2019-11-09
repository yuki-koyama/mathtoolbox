#ifndef MATHTOOLBOX_GRADIENT_DESCENT_HPP
#define MATHTOOLBOX_GRADIENT_DESCENT_HPP

#include <Eigen/Core>

namespace mathtoolbox
{
    namespace optimization
    {
        /// \brief Run a simple gradient descent to find a local minimizer of the specified function
        ///
        /// \details This algorithm uses the Backtracking Line Search algorithm to determine an appropriate step size.
        ///
        /// \param lower_bound The lower bound values. If this is a zero-length empty vector, the algorithm just ignores
        /// the lower bound condition.
        ///
        /// \param upper_bound The upper bound values. If this is a zero-length empty vector, the algorithm just ignores
        /// the upper bound condition.
        ///
        /// \param default_alpha The default step size that the algorithm first tries.
        void RunGradientDescent(const Eigen::VectorXd&                                        x_init,
                                const std::function<double(const Eigen::VectorXd&)>&          f,
                                const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& g,
                                const Eigen::VectorXd&                                        lower_bound,
                                const Eigen::VectorXd&                                        upper_bound,
                                const double                                                  epsilon,
                                const double                                                  default_alpha,
                                const unsigned int                                            max_num_iters,
                                Eigen::VectorXd&                                              x_star,
                                unsigned int&                                                 num_iters);
    } // namespace optimization
} // namespace mathtoolbox

#endif // MATHTOOLBOX_GRADIENT_DESCENT_HPP

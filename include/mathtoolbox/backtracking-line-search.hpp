#ifndef MATHTOOLBOX_BACKTRACKING_LINE_SEARCH_HPP
#define MATHTOOLBOX_BACKTRACKING_LINE_SEARCH_HPP

#include <Eigen/Core>
#include <functional>
#include <iostream>

namespace mathtoolbox
{
    namespace optimization
    {
        // Algorithm 3.1: Backtracking Line Search
        //
        // This algoritmh tries to find an appropriate step size that satisfies the Armijo condition (i.e., the
        // safficient decreasing condition). This algorithm runs faster than the line search algorithm for the strong
        // Wolfe conditions, but it does not guarantee the curvature condition, which is required to stabilize the
        // overall optimization.
        inline double RunBacktrackingLineSearch(const std::function<double(const Eigen::VectorXd&)>& f,
                                                const Eigen::VectorXd&                               grad,
                                                const Eigen::VectorXd&                               x,
                                                const Eigen::VectorXd&                               p,
                                                const double                                         alpha_init,
                                                const double                                         rho,
                                                const double                                         c = 1e-04)
        {
            constexpr unsigned int max_num_iters = 50;

            double alpha = alpha_init;

            for (int iter = 0; iter < max_num_iters; ++iter)
            {
                // Equation 3.4
                const bool armijo_condition = f(x + alpha * p) <= f(x) + c * alpha * grad.transpose() * p;

                if (armijo_condition)
                {
                    return alpha;
                }

                alpha *= rho;
            }

            std::cerr << "Warning: The line search did not converge." << std::endl;
            return alpha;
        }
    } // namespace optimization
} // namespace mathtoolbox

#endif // MATHTOOLBOX_BACKTRACKING_LINE_SEARCH_HPP

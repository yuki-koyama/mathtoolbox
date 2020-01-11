#ifndef MATHTOOLBOX_BACKTRACKING_LINE_SEARCH_HPP
#define MATHTOOLBOX_BACKTRACKING_LINE_SEARCH_HPP

#include <Eigen/Core>
#include <functional>
#include <iostream>

namespace mathtoolbox
{
    namespace optimization
    {
        /// \brief Run the backtracking line search.
        ///
        /// \details The algorithm is described in the book (Algorithm 3.1: Backtracking Line Search).
        ///
        /// This algorithm tries to find an appropriate step size that satisfies the Armijo condition (i.e., the
        /// safficient decreasing condition). This algorithm runs faster than the line search algorithm for the strong
        /// Wolfe conditions, but it does not guarantee the curvature condition, which is required to stabilize the
        /// overall optimization.
        inline double RunBacktrackingLineSearch(const std::function<double(const Eigen::VectorXd&)>& f,
                                                const Eigen::VectorXd&                               grad,
                                                const Eigen::VectorXd&                               x,
                                                const Eigen::VectorXd&                               p,
                                                const double                                         alpha_init,
                                                const double                                         rho,
                                                const double                                         c = 1e-04)
        {
            constexpr unsigned int max_num_iters = 50;

            assert(p.size() == x.size());

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

        /// \brief Run the backtracking line search with bound constraints
        ///
        /// \details This function should be slightly slower than the unbounded one since it evaluates bound conditions
        /// every step.
        ///
        /// \param lower_bound Lower bound of the search space. It can also be a zero-sized vector when there is no
        /// lower bound condition.
        ///
        /// \param upper_bound Upper bound of the search space. It can also be a zero-sized vector when there is no
        /// upper bound condition.
        inline double RunBacktrackingBoundedLineSearch(const std::function<double(const Eigen::VectorXd&)>& f,
                                                       const Eigen::VectorXd&                               grad,
                                                       const Eigen::VectorXd&                               x,
                                                       const Eigen::VectorXd&                               p,
                                                       const Eigen::VectorXd&                               lower_bound,
                                                       const Eigen::VectorXd&                               upper_bound,
                                                       const double                                         alpha_init,
                                                       const double                                         rho,
                                                       const double                                         c = 1e-04)
        {
            constexpr unsigned int max_num_iters = 50;

            assert(p.size() == x.size());

            double alpha = alpha_init;

            for (int iter = 0; iter < max_num_iters; ++iter)
            {
                const Eigen::VectorXd x_new = x + alpha * p;

                // Equation 3.4
                const bool armijo_condition = f(x_new) <= f(x) + c * alpha * grad.transpose() * p;

                // Bound conditions
                const bool lower_bound_condition = lower_bound.size() == 0 || (x_new - lower_bound).minCoeff() >= 0.0;
                const bool upper_bound_condition = upper_bound.size() == 0 || (upper_bound - x_new).minCoeff() >= 0.0;

                if (armijo_condition && lower_bound_condition && upper_bound_condition)
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

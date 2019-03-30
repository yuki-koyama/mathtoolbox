#ifndef BACKTRACKING_LINE_SEARCH_HPP
#define BACKTRACKING_LINE_SEARCH_HPP

#include <functional>
#include <Eigen/Core>

namespace mathtoolbox
{
    namespace optimization
    {
        // Procedure 3.1: Backtracking Line Search
        //
        // This algoritmh tries to find an appropriate step size that satisfies
        // the Armijo condition (i.e., the safficient decreasing condition).
        // This algorithm runs faster than the line search algorithm for the
        // strong Wolfe conditions, but it does not guarantee the curvature
        // condition, which is required to stabilize the overall optimization.
        inline double RunBacktrackingLineSearch(const std::function<double(const Eigen::VectorXd&)>& f,
                                                const Eigen::VectorXd& grad,
                                                const Eigen::VectorXd& x,
                                                const Eigen::VectorXd& p,
                                                const double alpha_init,
                                                const double rho,
                                                const double c)
        {
            constexpr unsigned int num_max_iterations = 50;

            unsigned counter = 0;
            double alpha = alpha_init;
            while (true)
            {
                // Equation 3.6a
                const bool armijo_condition = f(x + alpha * p) <= f(x) + c * alpha * grad.transpose() * p;

                if (armijo_condition || counter == num_max_iterations) { break; }

                alpha *= rho;

                ++ counter;
            }
            return alpha;
        }
    }
}

#endif /* BACKTRACKING_LINE_SEARCH_HPP */

#include <iostream>
#include <mathtoolbox/backtracking-line-search.hpp>
#include <mathtoolbox/gradient-descent.hpp>

using Eigen::VectorXd;

void mathtoolbox::optimization::RunGradientDescent(const VectorXd&                                 x_init,
                                                   const std::function<double(const VectorXd&)>&   f,
                                                   const std::function<VectorXd(const VectorXd&)>& g,
                                                   const VectorXd&                                 lower_bound,
                                                   const VectorXd&                                 upper_bound,
                                                   const double                                    epsilon,
                                                   const double                                    default_alpha,
                                                   const unsigned int                              max_num_iters,
                                                   VectorXd&                                       x_star,
                                                   unsigned int&                                   num_iters)
{
    assert(x_init.size() > 0);

    const int  num_dims         = x_init.size();
    const bool is_lower_bounded = (lower_bound.size() != 0);
    const bool is_upper_bounded = (upper_bound.size() != 0);

    assert(!is_lower_bounded || x_init.size() == lower_bound.size());
    assert(!is_upper_bounded || x_init.size() == upper_bound.size());

    // Initialize the solution
    x_star = x_init;

    // Calculate the initial value
    double y_star = f(x_star);

    for (int iter = 0; iter < max_num_iters; ++iter)
    {
        // Calculate the next candidate position
        const VectorXd grad = g(x_star);
        const VectorXd p    = [&]() {
            // Initialize the direction
            VectorXd p = -grad;

            // Freeze dimensions that are on the boundary and are directed outside
            for (int dim = 0; dim < num_dims; ++dim)
            {
                if ((is_lower_bounded && p(dim) - lower_bound(dim) < epsilon && grad(dim) > 0.0) ||
                    (is_upper_bounded && upper_bound(dim) - p(dim) < epsilon && grad(dim) < 0.0))
                {
                    p(dim) = 0.0;
                }
            }

            return p;
        }();
        const double step_size =
            RunBacktrackingBoundedLineSearch(f, grad, x_star, p, lower_bound, upper_bound, default_alpha, 0.5);
        VectorXd x_new = x_star + step_size * p;

        // Enforce bounding-box conditions by simple projection
        if (is_lower_bounded)
        {
            x_new = x_new.cwiseMax(lower_bound);
        }
        if (is_upper_bounded)
        {
            x_new = x_new.cwiseMin(upper_bound);
        }

        const double y_new = f(x_new);

        const double absolute_diff = std::abs(y_new - y_star);
        const double relative_diff = absolute_diff / std::max({std::abs(y_new), std::abs(y_star), 1.0});

        // Update the state parameters
        x_star    = x_new;
        y_star    = y_new;
        num_iters = iter + 1;

        // Check the termination condition
        if (absolute_diff < epsilon || relative_diff < epsilon)
        {
            return;
        }
    }

    std::cerr << "Warning: the gradient descent did not converge." << std::endl;
}

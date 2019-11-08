#include <mathtoolbox/gradient-descent.hpp>

using Eigen::VectorXd;

void mathtoolbox::optimization::RunGradientDescent(const VectorXd&                                 x_init,
                                                   const std::function<double(const VectorXd&)>&   f,
                                                   const std::function<VectorXd(const VectorXd&)>& g,
                                                   const VectorXd&                                 lower_bound,
                                                   const VectorXd&                                 upper_bound,
                                                   const double                                    epsilon,
                                                   const unsigned int                              max_num_iters,
                                                   VectorXd&                                       x_star,
                                                   unsigned int&                                   num_iters)
{
    constexpr double gamma = 1e-02;

    assert(x_init.size() > 0);

    const int num_dims = x_init.size();

    const bool is_lower_bounded = (lower_bound.size() != 0);
    const bool is_upper_bounded = (upper_bound.size() != 0);

    assert(!is_lower_bounded || x_init.size() == lower_bound.size());
    assert(!is_upper_bounded || x_init.size() == upper_bound.size());

    for (int iter = 0; iter < max_num_iters; ++iter)
    {
        VectorXd x_new = x_star - gamma * g(x_star);

        if (is_lower_bounded)
        {
            x_new = x_new.cwiseMax(lower_bound);
        }
        if (is_upper_bounded)
        {
            x_new = x_new.cwiseMin(upper_bound);
        }

        // TODO: Check the termination condition

        x_star    = x_new;
        num_iters = iter + 1;
    }
}

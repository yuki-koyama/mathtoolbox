#include <mathtoolbox/gradient-descent.hpp>

using Eigen::VectorXd;

void mathtoolbox::optimization::RunGradientDescent(const VectorXd&                                 x_init,
                                                   const std::function<double(const VectorXd&)>&   f,
                                                   const std::function<VectorXd(const VectorXd&)>& g,
                                                   const VectorXd&                                 lower_bound,
                                                   const VectorXd&                                 upper_bound,
                                                   const double                                    epsilon,
                                                   const unsigned int                              max_num_iterations,
                                                   VectorXd&                                       x_star,
                                                   unsigned int&                                   num_iterations)
{
    // TODO
    assert(false);

    x_star         = x_init;
    num_iterations = 0;
}

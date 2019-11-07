#include <mathtoolbox/bayesian-optimization.hpp>

using Eigen::VectorXd;

void mathtoolbox::optimization::RunBayesianOptimization(const std::function<double(const VectorXd&)>& f,
                                                        const unsigned int max_num_iterations,
                                                        VectorXd&          x_star,
                                                        unsigned int&      num_iterations)
{
    // TODO
    assert(false);
}

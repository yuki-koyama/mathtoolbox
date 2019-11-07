#include <mathtoolbox/bayesian-optimization.hpp>
#include <mathtoolbox/gaussian-process-regression.hpp>

using Eigen::VectorXd;

void mathtoolbox::optimization::RunBayesianOptimization(const std::function<double(const VectorXd&)>& f,
                                                        const unsigned int max_num_iterations,
                                                        const KernelType&  kernel_type,
                                                        VectorXd&          x_star,
                                                        unsigned int&      num_iterations)
{
    // TODO
    assert(false);
}

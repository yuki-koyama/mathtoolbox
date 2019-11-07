#ifndef MATHTOOLBOX_BAYESIAN_OPTIMIZATION_HPP
#define MATHTOOLBOX_BAYESIAN_OPTIMIZATION_HPP

#include <Eigen/Core>

namespace mathtoolbox
{
    namespace optimization
    {
        void RunBayesianOptimization(const std::function<double(const Eigen::VectorXd&)>& f,
                                     const unsigned int                                   max_num_iterations,
                                     Eigen::VectorXd&                                     x_star,
                                     unsigned int&                                        num_iterations);
    }
} // namespace mathtoolbox

#endif // MATHTOOLBOX_BAYESIAN_OPTIMIZATION_HPP

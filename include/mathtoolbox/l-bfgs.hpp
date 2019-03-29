#ifndef L_BFGS_HPP
#define L_BFGS_HPP

#include <functional>
#include <Eigen/Core>

namespace mathtoolbox
{
    namespace optimization
    {
        void RunLBfgs(const Eigen::VectorXd& x_init,
                      const std::function<double(const Eigen::VectorXd&)>& f,
                      const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& g,
                      const double epsilon,
                      const unsigned int max_num_iterations,
                      Eigen::VectorXd& x_star,
                      unsigned int& num_iterations);
    }
}

#endif // L_BFGS_HPP

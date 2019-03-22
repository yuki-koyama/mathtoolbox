#ifndef BFGS_HPP
#define BFGS_HPP

#include <Eigen/Core>

namespace mathtoolbox
{
    enum class Type
    {
        Min, Max
    };

    struct Setting
    {
        Eigen::VectorXd x_init;
        std::function<double(const Eigen::VectorXd&)> f;
        std::function<Eigen::VectorXd(const Eigen::VectorXd&)> f_grad;
        Type type = Type::Min;
    };

    struct Result
    {
        Eigen::VectorXd x_star;
        double y_star;
        unsigned num_iterations;
    };

    Result RunOptimization(const Setting& input);
}

#endif // BFGS_HPP

#include <mathtoolbox/bfgs.hpp>

namespace mathtoolbox
{
    Result RunOptimization(const Setting& input)
    {
        constexpr double epsilon = 1e-05;

        const unsigned dim = input.x_init.rows();

        const Eigen::MatrixXd H_init = Eigen::MatrixXd::Identity(dim, dim);

        Eigen::MatrixXd H = H_init;
        Eigen::VectorXd x = input.x_init;

        while (true)
        {
            break;
        }

        return Result
        {
            x,
            input.f(x)
        };
    }
}

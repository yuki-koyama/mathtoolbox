#include <mathtoolbox/bfgs.hpp>

namespace mathtoolbox
{
    Result RunOptimization(const Setting& input)
    {
        constexpr double epsilon = 1e-05;
        constexpr unsigned num_max_iterations = 10000;

        const unsigned dim = input.x_init.rows();

        const Eigen::MatrixXd H_init = Eigen::MatrixXd::Identity(dim, dim);

        Eigen::MatrixXd H = H_init;
        Eigen::VectorXd x = input.x_init;

        unsigned counter = 0;
        while (true)
        {
            const Eigen::VectorXd grad = input.f_grad(x);

            if (grad.norm() < epsilon || counter == num_max_iterations)
            {
                break;
            }

            ++ counter;
        }

        return Result
        {
            x,
            input.f(x)
        };
    }
}

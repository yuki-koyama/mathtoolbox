#include <mathtoolbox/bfgs.hpp>
#include <cmath>

namespace mathtoolbox
{
    Result RunOptimization(const Setting& input)
    {
        constexpr double epsilon = 1e-05;
        constexpr unsigned num_max_iterations = 1000;

        const unsigned dim = input.x_init.rows();

        const Eigen::MatrixXd H_init = Eigen::MatrixXd::Identity(dim, dim); // TODO

        Eigen::MatrixXd H = H_init;
        Eigen::VectorXd x = input.x_init;
        Eigen::VectorXd grad = input.f_grad(x);

        unsigned counter = 0;
        while (true)
        {

            if (grad.norm() < epsilon || counter == num_max_iterations)
            {
                break;
            }

            const Eigen::VectorXd p = - H * grad;
            const double alpha = 0.0; // TODO
            const Eigen::VectorXd x_next = x + alpha * p;
            const Eigen::VectorXd s = x_next - x;
            const Eigen::VectorXd grad_next = input.f_grad(x_next);
            const Eigen::VectorXd y = grad_next - grad;

            const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dim, dim);
            const double rho = 1.0 / (y.transpose() * s);

            assert(!std::isnan(rho));

            H = (I - rho * s * y.transpose()) * H * (I - rho * y * s.transpose()) + rho * s * s.transpose();

            x = x_next;
            grad = grad_next;

            ++ counter;
        }

        return Result
        {
            x,
            input.f(x),
            counter
        };
    }
}

#include <mathtoolbox/bfgs.hpp>
#include <cassert>
#include <cmath>

namespace mathtoolbox
{
    namespace internal
    {
        // Procedure 3.1: "Backtracking Line Search"
        double RunBacktrackingLineSearch(const std::function<double(const Eigen::VectorXd&)>& f,
                                         const Eigen::VectorXd& grad,
                                         const Eigen::VectorXd& x,
                                         const Eigen::VectorXd& p,
                                         const double alpha_init,
                                         const double rho,
                                         const double c)
        {
            double alpha = alpha_init;
            while (true)
            {
                // Equation 3.6a
                const bool sufficient_decrease_condition = f(x + alpha * p) <= f(x) + c * alpha * grad.transpose() * p;

                if (sufficient_decrease_condition) { break; }

                alpha *= rho;
            }
            return alpha;
        }
    }

    // Algorithm 8.1 "BFGS Method" with Backtracking Line Search
    Result RunOptimization(const Setting& input)
    {
        constexpr double epsilon = 1e-05;
        constexpr unsigned num_max_iterations = 1000;

        const unsigned dim = input.x_init.rows();

        const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dim, dim);
        const Eigen::MatrixXd H_init = I; // TODO

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

            // Equation 8.18
            const Eigen::VectorXd p = - H * grad;

            // Procedure 3.1
            const double alpha = internal::RunBacktrackingLineSearch(input.f, grad, x, p, 1.0, 0.5, 1e-04);

            const Eigen::VectorXd x_next = x + alpha * p;
            const Eigen::VectorXd s = x_next - x;
            const Eigen::VectorXd grad_next = input.f_grad(x_next);
            const Eigen::VectorXd y = grad_next - grad;

            // Equation 8.17
            const double rho = 1.0 / (y.transpose() * s);

            assert(!std::isnan(rho));

            // Equation 8.16
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

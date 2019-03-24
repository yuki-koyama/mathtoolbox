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
            constexpr unsigned num_max_iterations = 50;

            unsigned counter = 0;
            double alpha = alpha_init;
            while (true)
            {
                // Equation 3.6a
                const bool sufficient_decrease_condition = f(x + alpha * p) <= f(x) + c * alpha * grad.transpose() * p;

                if (sufficient_decrease_condition || counter == num_max_iterations) { break; }

                alpha *= rho;

                ++ counter;
            }
            return alpha;
        }
    }

    // Algorithm 8.1 "BFGS Method" with Backtracking Line Search
    Result RunOptimization(const Setting& input)
    {
        constexpr double epsilon = 1e-05;
        constexpr unsigned num_max_iterations = 1000;

        const auto f = (input.type == Type::Min) ? input.f : [&input](const Eigen::VectorXd& x) { return - input.f(x); };
        const auto g = (input.type == Type::Min) ? input.g : [&input](const Eigen::VectorXd& x) { return - input.g(x); };

        const unsigned dim = input.x_init.rows();

        const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dim, dim);
        const Eigen::MatrixXd H_init = I;

        Eigen::MatrixXd H = H_init;
        Eigen::VectorXd x = input.x_init;
        Eigen::VectorXd grad = g(x);

        bool is_first_step = true;

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
            const double alpha = internal::RunBacktrackingLineSearch(f, grad, x, p, 1.0, 0.5, 1e-04);

            const Eigen::VectorXd x_next = x + alpha * p;
            const Eigen::VectorXd s = x_next - x;
            const Eigen::VectorXd grad_next = g(x_next);
            const Eigen::VectorXd y = grad_next - grad;

            const double yts = y.transpose() * s;
            const double yty = y.transpose() * y;

            // Equation 8.17
            const double rho = 1.0 / yts;

            // As we do not search the step, alpha, using backtracking line
            // search without the curvature condition, the condition may be
            // violated. In that case, we need to correct the Hessian
            // approximation somehow. It is mentioned that damped BFGS is useful
            // for this purpose (p.201), which is a little complicated to
            // implement. Here, we take a simple solution, that is, just
            // skipping the Hessian approximation update when the condition is
            // violated, though this approach is not recommended (p.201).
            const bool is_curvature_condition_satisfied = yts > 0 && !std::isnan(rho);

            if (is_curvature_condition_satisfied)
            {
                // Equation 8.20
                if (is_first_step)
                {
                    const double scale = yts / yty;
                    H = scale * I;
                    is_first_step = false;
                }

                // Equation 8.16
                H = (I - rho * s * y.transpose()) * H * (I - rho * y * s.transpose()) + rho * s * s.transpose();
            }

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

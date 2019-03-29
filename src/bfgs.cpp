#include <mathtoolbox/numerical-optimization.hpp>
#include <cassert>
#include <cmath>

namespace mathtoolbox
{
    namespace optimization
    {
        // Algorithm 8.1 "BFGS Method" with Backtracking Line Search
        void RunBfgs(const Eigen::VectorXd& x_init,
                     const std::function<double(const Eigen::VectorXd&)>& f,
                     const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& g,
                     const double epsilon,
                     const unsigned max_num_iterations,
                     Eigen::VectorXd& x_star,
                     unsigned int& num_iterations)
        {
            const unsigned int dim = x_init.rows();

            const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(dim, dim);
            const Eigen::MatrixXd H_init = I;

            Eigen::MatrixXd H = H_init;
            Eigen::VectorXd x = x_init;
            Eigen::VectorXd grad = g(x);

            bool is_first_step = true;

            unsigned int counter = 0;
            while (true)
            {
                if (grad.norm() < epsilon || counter == max_num_iterations)
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

                    const Eigen::MatrixXd V = I - rho * y * s.transpose();

                    // Equation 8.16
                    H = V.transpose() * H * V + rho * s * s.transpose();
                }

                x = x_next;
                grad = grad_next;

                ++ counter;
            }

            // Output
            x_star = x;
            num_iterations = counter;
        }
    }
}

#include <cassert>
#include <cmath>
#include <mathtoolbox/bfgs.hpp>
#include <mathtoolbox/strong-wolfe-conditions-line-search.hpp>

// Algorithm 6.1: BFGS Method
void mathtoolbox::optimization::RunBfgs(const Eigen::VectorXd&                                        x_init,
                                        const std::function<double(const Eigen::VectorXd&)>&          f,
                                        const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& g,
                                        const double                                                  epsilon,
                                        const unsigned   max_num_iterations,
                                        Eigen::VectorXd& x_star,
                                        unsigned int&    num_iterations)
{
    const unsigned int dim = x_init.rows();

    const Eigen::MatrixXd I      = Eigen::MatrixXd::Identity(dim, dim);
    const Eigen::MatrixXd H_init = I;

    Eigen::MatrixXd H    = H_init;
    Eigen::VectorXd x    = x_init;
    Eigen::VectorXd grad = g(x);

    bool is_first_step = true;

    unsigned int counter = 0;
    while (true)
    {
        if (grad.norm() < epsilon || counter == max_num_iterations)
        {
            break;
        }

        // Equation 6.18
        const Eigen::VectorXd p = -H * grad;

        // Algorithm 3.5: Line Search Algorithm
        const double alpha = RunStrongWolfeConditionsLineSearch(f, g, x, p, 1.0);

        const Eigen::VectorXd x_next    = x + alpha * p;
        const Eigen::VectorXd s         = x_next - x;
        const Eigen::VectorXd grad_next = g(x_next);
        const Eigen::VectorXd y         = grad_next - grad;

        const double yts = y.transpose() * s;
        const double yty = y.transpose() * y;

        // Equation 6.14
        const double rho = 1.0 / yts;

        // Equation 6.20
        if (is_first_step)
        {
            const double scale = yts / yty;
            H                  = scale * I;
            is_first_step      = false;
        }

        const Eigen::MatrixXd V = I - rho * y * s.transpose();

        // Equation 6.17
        H = V.transpose() * H * V + rho * s * s.transpose();

        x    = x_next;
        grad = grad_next;

        ++counter;
    }

    // Output
    x_star         = x;
    num_iterations = counter;
}

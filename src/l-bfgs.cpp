#include <cassert>
#include <cmath>
#include <list>
#include <mathtoolbox/l-bfgs.hpp>
#include <mathtoolbox/strong-wolfe-conditions-line-search.hpp>

namespace
{
    struct LBfgsEntry
    {
        const Eigen::VectorXd s;
        const Eigen::VectorXd y;
        const double          gamma;
        const double          rho;
    };

    // Algotihm 7.4: L-BFGS Two-Loop Recursion
    Eigen::VectorXd RunLBfgsTwoLoopRecursion(const Eigen::DiagonalMatrix<double, Eigen::Dynamic>& H_0,
                                             const Eigen::VectorXd&                               grad,
                                             const std::list<LBfgsEntry>&                         data)
    {
        Eigen::VectorXd q = grad;

        std::list<double> alphas;

        // Visit in the newer-to-older order
        for (auto iter = data.begin(); iter != data.end(); ++iter)
        {
            const Eigen::VectorXd& s   = iter->s;
            const Eigen::VectorXd& y   = iter->y;
            const double&          rho = iter->rho;

            const double alpha = rho * s.transpose() * q;

            alphas.push_back(alpha);

            q -= alpha * y;
        }

        Eigen::VectorXd r = H_0 * q;

        // Visit in the older-to-newer order
        for (auto iter = data.rbegin(); iter != data.rend(); ++iter)
        {
            const Eigen::VectorXd& s   = iter->s;
            const Eigen::VectorXd& y   = iter->y;
            const double&          rho = iter->rho;

            const double& alpha = alphas.back();

            const double beta = rho * y.transpose() * r;

            r += s * (alpha - beta);

            alphas.pop_back();
        }

        assert(alphas.empty());

        return r;
    }
} // namespace

// Algorithm 7.5: L-BFGS
void mathtoolbox::optimization::RunLBfgs(const Eigen::VectorXd&                                        x_init,
                                         const std::function<double(const Eigen::VectorXd&)>&          f,
                                         const std::function<Eigen::VectorXd(const Eigen::VectorXd&)>& g,
                                         const double                                                  epsilon,
                                         const unsigned int max_num_iterations,
                                         Eigen::VectorXd&   x_star,
                                         unsigned int&      num_iterations)
{
    const unsigned int     dim = x_init.rows();
    constexpr unsigned int m   = 5;

    const Eigen::DiagonalMatrix<double, Eigen::Dynamic> I = Eigen::VectorXd::Ones(dim).asDiagonal();

    Eigen::VectorXd x             = x_init;
    Eigen::VectorXd grad          = g(x);
    double          absolute_diff = std::numeric_limits<double>::max();

    std::list<LBfgsEntry> data;

    unsigned int counter = 0;
    while (true)
    {
        if (absolute_diff < epsilon || grad.norm() < epsilon || counter == max_num_iterations)
        {
            break;
        }

        // Choose H_k^0
        const Eigen::DiagonalMatrix<double, Eigen::Dynamic> H_0 = (data.empty() ? 1.0 : data.front().gamma) * I;

        // Get the descent direction by the two-loop recursion algorithm
        // Algotihm 7.4: L-BFGS Two-Loop Recursion
        const Eigen::VectorXd p = -RunLBfgsTwoLoopRecursion(H_0, grad, data);

        // Algorithm 3.5: Line Search Algorithm
        const double alpha = RunStrongWolfeConditionsLineSearch(f, g, x, p, 1.0);

        const Eigen::VectorXd x_next    = x + alpha * p;
        const Eigen::VectorXd s         = x_next - x;
        const Eigen::VectorXd grad_next = g(x_next);
        const Eigen::VectorXd y         = grad_next - grad;

        const double sty = s.transpose() * y;
        const double yty = y.transpose() * y;

        // Equation 7.20
        const double gamma = sty / yty;

        // Equation 7.17
        const double rho = 1.0 / sty;

        // Manage the data in the newer-to-older order
        data.push_front({s, y, gamma, rho});
        if (data.size() > m)
        {
            data.pop_back();
        }

        assert(data.size() == counter + 1 || data.size() <= m);

        x             = x_next;
        grad          = grad_next;
        absolute_diff = s.norm();

        ++counter;
    }

    // Output
    x_star         = x;
    num_iterations = counter;
}

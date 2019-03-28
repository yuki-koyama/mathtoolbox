#ifndef NUMERICAL_OPTIMIZATION_HPP
#define NUMERICAL_OPTIMIZATION_HPP

#include <mathtoolbox/bfgs.hpp>
#include <functional>
#include <stdexcept>
#include <Eigen/Core>

namespace mathtoolbox
{
    namespace optimization
    {
        enum class Algorithm
        {
            Bfgs
        };

        enum class Type
        {
            Min, Max
        };

        struct Setting
        {
            Algorithm algorithm                                      = Algorithm::Bfgs;
            Eigen::VectorXd x_init                                   = Eigen::VectorXd(0);
            std::function<double(const Eigen::VectorXd&)> f          = nullptr;
            std::function<Eigen::VectorXd(const Eigen::VectorXd&)> g = nullptr;
            double epsilon                                           = 1e-05;
            unsigned max_num_iterations                              = 1000;
            Type type                                                = Type::Min;
        };

        struct Result
        {
            Eigen::VectorXd x_star;
            unsigned        num_iterations;
        };

        inline Result RunOptimization(const Setting& input)
        {
            const auto f = (input.type == Type::Min) ? input.f : [&input](const Eigen::VectorXd& x) { return - input.f(x); };
            const auto g = (input.type == Type::Min) ? input.g : [&input](const Eigen::VectorXd& x) { return - input.g(x); };

            switch (input.algorithm) {
                case Algorithm::Bfgs:
                {
                    if (!input.f || !input.g || input.x_init.rows() == 0)
                    {
                        throw std::invalid_argument("Invalid setting.");
                    }

                    Result result;
                    RunBfgs(input.x_init, f, g, input.epsilon, input.max_num_iterations, result.x_star, result.num_iterations);
                    return result;
                }

        namespace internal
        {
            // Procedure 3.1: "Backtracking Line Search"
            inline double RunBacktrackingLineSearch(const std::function<double(const Eigen::VectorXd&)>& f,
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
    }
}

#endif // NUMERICAL_OPTIMIZATION_HPP

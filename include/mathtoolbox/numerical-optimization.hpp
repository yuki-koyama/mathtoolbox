#ifndef NUMERICAL_OPTIMIZATION_HPP
#define NUMERICAL_OPTIMIZATION_HPP

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
            Algorithm algorithm = Algorithm::Bfgs;
            Eigen::VectorXd x_init;
            std::function<double(const Eigen::VectorXd&)> f;
            std::function<Eigen::VectorXd(const Eigen::VectorXd&)> g;
            double epsilon = 1e-05;
            unsigned max_num_iterations = 1000;
            Type type = Type::Min;
        };

        struct Result
        {
            Eigen::VectorXd x_star;
            double y_star;
            unsigned num_iterations;
        };

        Result RunBfgs(const Eigen::VectorXd& x_init,
                       const std::function<double(const Eigen::VectorXd&)> f,
                       const std::function<Eigen::VectorXd(const Eigen::VectorXd&)> g,
                       const double epsilon,
                       const unsigned max_num_iterations);

        inline Result RunOptimization(const Setting& input)
        {
            const auto f = (input.type == Type::Min) ? input.f : [&input](const Eigen::VectorXd& x) { return - input.f(x); };
            const auto g = (input.type == Type::Min) ? input.g : [&input](const Eigen::VectorXd& x) { return - input.g(x); };

            switch (input.algorithm) {
                case Algorithm::Bfgs:
                {
                    return RunBfgs(input.x_init, f, g, input.epsilon, input.max_num_iterations);
                }
            }
        }
    }
}

#endif // NUMERICAL_OPTIMIZATION_HPP

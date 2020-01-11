#ifndef MATHTOOLBOX_NUMERICAL_OPTIMIZATION_HPP
#define MATHTOOLBOX_NUMERICAL_OPTIMIZATION_HPP

#include <Eigen/Core>
#include <functional>
#include <mathtoolbox/bfgs.hpp>
#include <mathtoolbox/l-bfgs.hpp>
#include <stdexcept>

namespace mathtoolbox
{
    namespace optimization
    {
        enum class Algorithm
        {
            Bfgs,
            LBfgs
        };

        enum class Type
        {
            Min,
            Max
        };

        struct Setting
        {
            Algorithm                                              algorithm          = Algorithm::Bfgs;
            Eigen::VectorXd                                        x_init             = Eigen::VectorXd(0);
            std::function<double(const Eigen::VectorXd&)>          f                  = nullptr;
            std::function<Eigen::VectorXd(const Eigen::VectorXd&)> g                  = nullptr;
            double                                                 epsilon            = 1e-05;
            unsigned int                                           max_num_iterations = 1000;
            Type                                                   type               = Type::Min;
        };

        struct Result
        {
            Eigen::VectorXd x_star;
            unsigned        num_iterations;
        };

        /// \brief Run optimization with the specified setting
        ///
        /// \details This is a utility function to run various optimization types in a consitent interface.
        inline Result RunOptimization(const Setting& input)
        {
            const auto f = (input.type == Type::Min) ? input.f : [&input](const Eigen::VectorXd& x) -> double {
                return -input.f(x);
            };
            const auto g = (input.type == Type::Min) ? input.g : [&input](const Eigen::VectorXd& x) -> Eigen::VectorXd {
                return -input.g(x);
            };

            switch (input.algorithm)
            {
                case Algorithm::Bfgs:
                {
                    if (!input.f || !input.g || input.x_init.rows() == 0)
                    {
                        throw std::invalid_argument("Invalid setting.");
                    }

                    Result result;
                    RunBfgs(input.x_init,
                            f,
                            g,
                            input.epsilon,
                            input.max_num_iterations,
                            result.x_star,
                            result.num_iterations);
                    return result;
                }
                case Algorithm::LBfgs:
                {
                    if (!input.f || !input.g || input.x_init.rows() == 0)
                    {
                        throw std::invalid_argument("Invalid setting.");
                    }

                    Result result;
                    RunLBfgs(input.x_init,
                             f,
                             g,
                             input.epsilon,
                             input.max_num_iterations,
                             result.x_star,
                             result.num_iterations);
                    return result;
                }
            }
        }
    } // namespace optimization
} // namespace mathtoolbox

#endif // MATHTOOLBOX_NUMERICAL_OPTIMIZATION_HPP

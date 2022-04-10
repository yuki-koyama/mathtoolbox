#include <Eigen/Core>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mathtoolbox/gradient-descent.hpp>
#include <optimization-test-functions.hpp>

int main()
{
    constexpr otf::FunctionType type     = otf::FunctionType::Rosenbrock;
    constexpr int               num_dims = 10;

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    const Eigen::VectorXd x_init = Eigen::VectorXd::Random(num_dims);
    const auto            f      = [&type](const Eigen::VectorXd& x) { return otf::GetValue(x, type); };
    const auto            g      = [&type](const Eigen::VectorXd& x) { return otf::GetGrad(x, type); };

    const Eigen::VectorXd lower_bound; // Empty vector indicates no lower bound constraint
    const Eigen::VectorXd upper_bound; // Empty vector indicates no upper bound constraint

    constexpr double epsilon       = 1e-12;
    constexpr double default_alpha = 1e-01;
    constexpr int    max_num_iters = 100000;

    Eigen::VectorXd x_star;
    unsigned        num_iters;
    mathtoolbox::optimization::RunGradientDescent(
        x_init, f, g, lower_bound, upper_bound, epsilon, default_alpha, max_num_iters, x_star, num_iters);

    const Eigen::VectorXd expected_solution = otf::GetSolution(num_dims, type);
    const double          expected_value    = otf::GetValue(expected_solution, type);
    const double          initial_value     = otf::GetValue(x_init, type);
    const double          found_value       = otf::GetValue(x_star, type);

    std::cout << "#iterations: " << num_iters << std::endl;
    std::cout << "Initial solution: " << x_init.transpose() << " (" << initial_value << ")" << std::endl;
    std::cout << "Found solution: " << x_star.transpose() << " (" << found_value << ")" << std::endl;
    std::cout << "Expected solution: " << expected_solution.transpose() << " (" << expected_value << ")" << std::endl;

    return 0;
}

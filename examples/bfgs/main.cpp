#include <cstdlib>
#include <ctime>
#include <iostream>
#include <Eigen/Core>
#include <mathtoolbox/numerical-optimization.hpp>
#include <optimization-test-functions.hpp>

int main()
{
    constexpr otf::FunctionType type = otf::FunctionType::Rosenbrock;
    constexpr int dimensions = 100;

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    mathtoolbox::optimization::Setting setting;
    setting.algorithm = mathtoolbox::optimization::Algorithm::Bfgs;
    setting.x_init = Eigen::VectorXd::Random(dimensions);
    setting.f = [](const Eigen::VectorXd& x) { return otf::GetValue(x, type); };
    setting.g = [](const Eigen::VectorXd& x) { return otf::GetGrad(x, type); };
    setting.type = mathtoolbox::optimization::Type::Min;
    setting.max_num_iterations = 1000;

    const mathtoolbox::optimization::Result result = mathtoolbox::optimization::RunOptimization(setting);

    const Eigen::VectorXd expected_solution = otf::GetSolution(dimensions, type);

    std::cout << "#iterations: " << result.num_iterations << std::endl;
    std::cout << "Initial solution: " << setting.x_init.transpose() << " (" << otf::GetValue(setting.x_init, type) << ")" << std::endl;
    std::cout << "Found solution: " << result.x_star.transpose() << " (" << otf::GetValue(result.x_star, type) << ")" << std::endl;
    std::cout << "Expected solution: " << expected_solution.transpose() << " (" << otf::GetValue(expected_solution, type) << ")" << std::endl;

    return 0;
}

#include <Eigen/Core>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <mathtoolbox/bayesian-optimization.hpp>
#include <optimization-test-functions.hpp>

void ExportMatrixToCsv(const std::string& file_path, const Eigen::MatrixXd& X)
{
    std::ofstream   file(file_path);
    Eigen::IOFormat format(Eigen::StreamPrecision, Eigen::DontAlignCols, ",");
    file << X.format(format);
}

int main()
{
    constexpr otf::FunctionType type       = otf::FunctionType::Rosenbrock;
    constexpr int               num_dims   = 5;
    constexpr int               num_iters  = 80;
    constexpr int               num_trials = 5;

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    const auto            objective_func = [&](const Eigen::VectorXd& x) { return -otf::GetValue(x, type); };
    const Eigen::VectorXd lower_bound    = Eigen::VectorXd::Constant(num_dims, -3.0);
    const Eigen::VectorXd upper_bound    = Eigen::VectorXd::Constant(num_dims, 3.0);

    Eigen::MatrixXd result(num_iters, num_trials);

    for (int trial = 0; trial < num_trials; ++trial)
    {
        std::cout << "#trial: " << std::to_string(trial) << std::endl;

        mathtoolbox::optimization::BayesianOptimizer optimizer(objective_func, lower_bound, upper_bound);

        for (int iter = 0; iter < num_iters; ++iter)
        {
            const auto new_point = optimizer.Step();

            const Eigen::VectorXd current_solution = optimizer.GetCurrentOptimizer();

            std::cout << current_solution.transpose().format(Eigen::IOFormat(2));
            std::cout << " (" << optimizer.EvaluatePoint(current_solution) << ")" << std::endl;

            result(iter, trial) = optimizer.EvaluatePoint(current_solution);
        }
    }

    const Eigen::VectorXd expected_solution = otf::GetSolution(num_dims, type);
    const double          expected_value    = otf::GetValue(expected_solution, type);

    std::cout << "Expected solution: " << expected_solution.transpose() << " (" << expected_value << ")" << std::endl;

    ExportMatrixToCsv("./result.csv", result);

    return 0;
}

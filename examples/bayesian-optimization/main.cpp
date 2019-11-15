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
    constexpr otf::FunctionType type       = otf::FunctionType::Sphere;
    constexpr int               num_dims   = 5;
    constexpr int               num_iters  = 15;
    constexpr int               num_trials = 2;

    std::srand(static_cast<unsigned>(std::time(nullptr)));

    const auto            objective_func = [&](const Eigen::VectorXd& x) { return -otf::GetValue(x, type); };
    const Eigen::VectorXd lower_bound    = Eigen::VectorXd::Constant(num_dims, -1.0);
    const Eigen::VectorXd upper_bound    = Eigen::VectorXd::Constant(num_dims, 1.0);

    Eigen::MatrixXd bo_result(num_iters, num_trials);

    for (int trial = 0; trial < num_trials; ++trial)
    {
        std::cout << "#trial: " << std::to_string(trial + 1) << std::endl;

        mathtoolbox::optimization::BayesianOptimizer optimizer(objective_func, lower_bound, upper_bound);

        for (int iter = 0; iter < num_iters; ++iter)
        {
            const auto new_point = optimizer.Step();

            const Eigen::VectorXd current_solution      = optimizer.GetCurrentOptimizer();
            const double          current_optimal_value = optimizer.EvaluatePoint(current_solution);

            std::cout << current_solution.transpose().format(Eigen::IOFormat(2));
            std::cout << " (" << current_optimal_value << ")" << std::endl;

            bo_result(iter, trial) = current_optimal_value;
        }
    }

    Eigen::MatrixXd rand_result(num_iters, num_trials);

    for (int trial = 0; trial < num_trials; ++trial)
    {
        std::cout << "#trial: " << std::to_string(trial + 1) << std::endl;

        Eigen::VectorXd current_solution;
        double          current_optimal_value;

        for (int iter = 0; iter < num_iters; ++iter)
        {
            const auto new_point = [&]() {
                const Eigen::VectorXd normalized_sample =
                    0.5 * (Eigen::VectorXd::Random(num_dims) + Eigen::VectorXd::Ones(num_dims));
                const Eigen::VectorXd sample =
                    (normalized_sample.array() * (upper_bound - lower_bound).array()).matrix() + lower_bound;

                return sample;
            }();

            const double new_value = objective_func(new_point);

            if (current_solution.size() == 0 || current_optimal_value < new_value)
            {
                current_solution      = new_point;
                current_optimal_value = new_value;
            }

            std::cout << current_solution.transpose().format(Eigen::IOFormat(2));
            std::cout << " (" << current_optimal_value << ")" << std::endl;

            rand_result(iter, trial) = current_optimal_value;
        }
    }

    const Eigen::VectorXd expected_solution = otf::GetSolution(num_dims, type);
    const double          expected_value    = objective_func(expected_solution);

    std::cout << "Expected solution: " << expected_solution.transpose() << " (" << expected_value << ")" << std::endl;

    ExportMatrixToCsv("./bo_result.csv", bo_result);
    ExportMatrixToCsv("./rand_result.csv", rand_result);

    return 0;
}

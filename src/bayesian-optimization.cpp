#include <mathtoolbox/acquisition-functions.hpp>
#include <mathtoolbox/bayesian-optimization.hpp>
#include <mathtoolbox/gaussian-process-regression.hpp>

using Eigen::VectorXd;

mathtoolbox::optimization::BayesianOptimizer::BayesianOptimizer(const std::function<double(const VectorXd&)>& f,
                                                                const VectorXd&   lower_bound,
                                                                const VectorXd&   upper_bound,
                                                                const KernelType& kernel_type)
    : m_f(f), m_lower_bound(lower_bound), m_upper_bound(upper_bound), m_kernel_type(kernel_type)
{
    // TODO
    assert(false);
}

VectorXd mathtoolbox::optimization::BayesianOptimizer::Step()
{
    const GaussianProcessRegression::KernelType kernel_type = [&]() {
        switch (m_kernel_type)
        {
            case KernelType::ArdSquaredExp:
                return GaussianProcessRegression::KernelType::ArdSquaredExp;
            case KernelType::ArdMatern52:
                return GaussianProcessRegression::KernelType::ArdMatern52;
            default:
                assert(false);
        }
    }();

    m_regressor = std::make_shared<GaussianProcessRegression>(m_X, m_y, kernel_type);

    const Eigen::VectorXd x_plus = GetCurrentOptimizer();

    const auto acquisition_func = [&](const Eigen::VectorXd& x) {
        return GetExpectedImprovement(
            x,
            [&](const Eigen::VectorXd& x) { return m_regressor->PredictMean(x); },
            [&](const Eigen::VectorXd& x) { return m_regressor->PredictStdev(x); },
            x_plus);
    };

    const auto acquisition_func_deriv = [&](const Eigen::VectorXd& x) {
        return GetExpectedImprovementDerivative(
            x,
            [&](const Eigen::VectorXd& x) { return m_regressor->PredictMean(x); },
            [&](const Eigen::VectorXd& x) { return m_regressor->PredictStdev(x); },
            x_plus,
            [&](const Eigen::VectorXd& x) { return m_regressor->PredictMeanDeriv(x); },
            [&](const Eigen::VectorXd& x) { return m_regressor->PredictStdevDeriv(x); });
    };

    // TODO
    assert(false);
}

VectorXd mathtoolbox::optimization::BayesianOptimizer::GetCurrentOptimizer() const
{
    // TODO
    assert(false);
}

void mathtoolbox::optimization::RunBayesianOptimization(const std::function<double(const VectorXd&)>& f,
                                                        const VectorXd&                               lower_bound,
                                                        const VectorXd&                               upper_bound,
                                                        const unsigned int max_num_iterations,
                                                        const KernelType&  kernel_type,
                                                        VectorXd&          x_star,
                                                        unsigned int&      num_iterations)
{
    // TODO
    assert(false);
}

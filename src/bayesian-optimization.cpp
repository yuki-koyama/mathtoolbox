#include <algorithm>
#include <iostream>
#include <limits>
#include <mathtoolbox/acquisition-functions.hpp>
#include <mathtoolbox/bayesian-optimization.hpp>
#include <mathtoolbox/gaussian-process-regression.hpp>
#include <mathtoolbox/gradient-descent.hpp>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

mathtoolbox::optimization::BayesianOptimizer::BayesianOptimizer(const std::function<double(const VectorXd&)>& f,
                                                                const VectorXd&   lower_bound,
                                                                const VectorXd&   upper_bound,
                                                                const KernelType& kernel_type)
    : m_f(f), m_lower_bound(lower_bound), m_upper_bound(upper_bound), m_kernel_type(kernel_type)
{
}

std::pair<VectorXd, double> mathtoolbox::optimization::BayesianOptimizer::Step()
{
    // Check if this is the first step, and if so, use a randomly sampled point as the initial solution
    if (m_X.cols() == 0)
    {
        assert(m_y.size() == 0);
        assert(m_lower_bound.size() == m_upper_bound.size());

        const int num_dims = m_lower_bound.size();

        const VectorXd x_new = [&]() {
            const VectorXd normalized_sample = 0.5 * (VectorXd::Random(num_dims) + VectorXd::Ones(num_dims));
            const VectorXd sample =
                (normalized_sample.array() * (m_upper_bound - m_lower_bound).array()).matrix() + m_lower_bound;

            return sample;
        }();
        const double y_new = EvaluatePoint(x_new);

        AddDataEntry(x_new, y_new);
        ConstructSurrogateFunction();

        return {x_new, y_new};
    }

    const VectorXd x_plus   = GetCurrentOptimizer();
    const int      num_dims = x_plus.size();

    const auto acquisition_func = [&](const VectorXd& x) {
        return GetExpectedImprovement(
            x,
            [&](const VectorXd& x) { return m_regressor->PredictMean(x); },
            [&](const VectorXd& x) { return m_regressor->PredictStdev(x); },
            x_plus);
    };

    const auto acquisition_func_deriv = [&](const VectorXd& x) {
        return GetExpectedImprovementDerivative(
            x,
            [&](const VectorXd& x) { return m_regressor->PredictMean(x); },
            [&](const VectorXd& x) { return m_regressor->PredictStdev(x); },
            x_plus,
            [&](const VectorXd& x) { return m_regressor->PredictMeanDeriv(x); },
            [&](const VectorXd& x) { return m_regressor->PredictStdevDeriv(x); });
    };

    // Perform global search
    // ----
    // Currently, this code relies on a highly naive algorithm: generate samples uniformly and then pick the best one (I
    // am not sure if this algorithm has its name). A possible better approach is to use DIRECT, a more sophisticated
    // global search algorithm, to obtain an initial solution and then refine it by using L-BFGS-B, a local bounded
    // gradient-based algorithm.
    const VectorXd x_new = [&]() {
        constexpr int num_samples = 1000;

        std::vector<VectorXd> samples(num_samples);
        std::for_each(std::begin(samples), std::end(samples), [&](VectorXd& sample) {
            const VectorXd normalized_sample = 0.5 * (VectorXd::Random(num_dims) + VectorXd::Ones(num_dims));
            sample = (normalized_sample.array() * (m_upper_bound - m_lower_bound).array()).matrix() + m_lower_bound;
        });

        std::vector<double> values(num_samples);
        std::transform(std::begin(samples), std::end(samples), std::begin(values), [&](const VectorXd& sample) {
            return acquisition_func(sample);
        });

        const int max_index = std::distance(std::begin(values), std::max_element(std::begin(values), std::end(values)));

        const VectorXd x_init = samples[max_index];

        VectorXd x_star;
        unsigned num_iters;
        RunGradientDescent(
            x_init,
            [&](const VectorXd& x) -> double { return -acquisition_func(x); },
            [&](const VectorXd& x) -> VectorXd { return -acquisition_func_deriv(x); },
            m_lower_bound,
            m_upper_bound,
            1e-12,
            1e+02,
            1000,
            x_star,
            num_iters);

        return x_star;
    }();

    // Calculate the value of the new data point, whose cost is probably heavy
    const double y_new = EvaluatePoint(x_new);

    // Add the new data point
    AddDataEntry(x_new, y_new);

    // Update the surrogate function
    ConstructSurrogateFunction();

    // Return the new data point
    return {x_new, y_new};
}

VectorXd mathtoolbox::optimization::BayesianOptimizer::GetCurrentOptimizer() const
{
    assert(m_X.cols() != 0);
    assert(m_y.size() != 0);

    int index;
    m_y.maxCoeff(&index);

    return m_X.col(index);
}

double mathtoolbox::optimization::BayesianOptimizer::EvaluatePoint(const VectorXd& x) const
{
    return m_f(x);
}

double mathtoolbox::optimization::BayesianOptimizer::PredictMean(const VectorXd& x) const
{
    return m_regressor == nullptr ? std::numeric_limits<double>::quiet_NaN() : m_regressor->PredictMean(x);
}

double mathtoolbox::optimization::BayesianOptimizer::PredictStdev(const VectorXd& x) const
{
    return m_regressor == nullptr ? std::numeric_limits<double>::quiet_NaN() : m_regressor->PredictStdev(x);
}

double mathtoolbox::optimization::BayesianOptimizer::CalcAcquisitionValue(const VectorXd& x) const
{
    const VectorXd x_plus = GetCurrentOptimizer();

    return GetExpectedImprovement(
        x,
        [&](const VectorXd& x) { return m_regressor->PredictMean(x); },
        [&](const VectorXd& x) { return m_regressor->PredictStdev(x); },
        x_plus);
}

std::pair<MatrixXd, VectorXd> mathtoolbox::optimization::BayesianOptimizer::GetData() const
{
    return {m_X, m_y};
}

void mathtoolbox::optimization::BayesianOptimizer::AddDataEntry(const VectorXd& x_new, const double y_new)
{
    if (m_X.cols() == 0)
    {
        m_X = x_new;
        m_y = VectorXd::Constant(1, y_new);

        return;
    }

    const int num_dims = x_new.size();

    const MatrixXd X_old = m_X;
    const VectorXd y_old = m_y;

    m_X.resize(num_dims, X_old.cols() + 1);
    m_y.resize(m_y.size() + 1);
    m_X.block(0, 0, num_dims, X_old.cols()) = X_old;
    m_X.col(X_old.cols())                   = x_new;
    m_y.segment(0, y_old.size())            = y_old;
    m_y(y_old.size())                       = y_new;
}

void mathtoolbox::optimization::BayesianOptimizer::ConstructSurrogateFunction()
{
    assert(m_X.cols() != 0);
    assert(m_y.size() != 0);

    const GaussianProcessRegressor::KernelType kernel_type = [&]() {
        switch (m_kernel_type)
        {
            case KernelType::ArdSquaredExp:
                return GaussianProcessRegressor::KernelType::ArdSquaredExp;
            case KernelType::ArdMatern52:
                return GaussianProcessRegressor::KernelType::ArdMatern52;
            default:
                assert(false);
        }
    }();

    const int      num_dims = m_X.rows();
    const VectorXd default_kernel_hyperparams =
        (VectorXd(num_dims + 1) << 0.20, VectorXd::Constant(num_dims, 0.20)).finished();

    m_regressor = std::make_shared<GaussianProcessRegressor>(m_X, m_y, kernel_type);
#if true
    m_regressor->SetHyperparams(default_kernel_hyperparams, 2e-05);
#else
    m_regressor->PerformMaximumLikelihood(default_kernel_hyperparams, 2e-05);
#endif
}

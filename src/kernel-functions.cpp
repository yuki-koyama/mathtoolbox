#include <mathtoolbox/kernel-functions.hpp>

using Eigen::VectorXd;

double
mathtoolbox::GetArdSquaredExponentialKernel(const VectorXd& x_a, const VectorXd& x_b, const VectorXd& hyperparameters)
{
    assert(x_a.size() == x_b.size());
    assert(x_a.size() == hyperparameters.size() + 1);

    const int              dim             = x_a.size();
    const double&          sigma_squared_f = hyperparameters[0];
    const Eigen::VectorXd& length_scales   = hyperparameters.segment(1, dim);

    const double sum = [&]() {
        double sum = 0.0;
        for (int i = 0; i < dim; ++i)
        {
            const double r = x_a(i) - x_b(i);
            sum += (r * r) / (length_scales(i) * length_scales(i));
        }
        return sum;
    }();

    return sigma_squared_f * std::exp(-0.5 * sum);
}

VectorXd mathtoolbox::GetArdSquaredExponentialKernelHyperparametersDerivative(const VectorXd& x_a,
                                                                              const VectorXd& x_b,
                                                                              const VectorXd& hyperparameters)
{
    assert(x_a.size() == x_b.size());
    assert(x_a.size() == hyperparameters.size() + 1);

    const int              dim           = x_a.size();
    const Eigen::VectorXd& length_scales = hyperparameters.segment(1, dim);

    VectorXd derivative(hyperparameters.size());

    derivative(0) = [&]() {
        double sum = 0.0;
        for (int i = 0; i < dim; ++i)
        {
            const double r = x_a(i) - x_b(i);
            sum += (r * r) / (length_scales(i) * length_scales(i));
        }
        return std::exp(-0.5 * sum);
    }();

    const double k = GetArdSquaredExponentialKernel(x_a, x_b, hyperparameters);

    for (int i = 0; i < dim; ++i)
    {
        const double r    = x_a(i) - x_b(i);
        derivative(i + 1) = k * (r * r) / (length_scales(i) * length_scales(i) * length_scales(i));
    }

    return derivative;
}
